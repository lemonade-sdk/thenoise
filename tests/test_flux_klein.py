"""Flux.2 (Flux Klein): token/position pack-unpack, a small end-to-end DiT forward
(eager — see ``conftest``), the reference-token packing and the reference KV cache.

The schedule contract, detection, defaults and the latent-format registration are
catalog-wide (``test_schedules.py`` / ``test_detect.py`` / ``test_catalog.py``).
"""
from __future__ import annotations

import pytest
import torch

from thenoise.dit.flux2.models import Flux2, Flux2Params
from thenoise.dit.flux2.sampling import prc_img, prc_txt, scatter_ids
from thenoise.dit.kvcache import KVCache
from thenoise.models import FluxKleinModel


def _tiny_params() -> Flux2Params:
    """The smallest config the Flux2 accepts (1 block each, 2 heads of 8)."""
    return Flux2Params(
        in_channels=8,
        context_in_dim=24,
        hidden_size=16,
        num_heads=2,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[2, 2, 2, 2],
        mlp_ratio=1.5,
        use_guidance_embed=False,
    )


@pytest.fixture
def tiny_flux2():
    """A random-init Flux2 with the smallest sensible config."""
    torch.manual_seed(0)
    return Flux2(_tiny_params()).eval().requires_grad_(False)


def _tiny_inputs(tiny_flux2, seq=4):
    x = torch.randn(1, seq, 8)
    x_ids = torch.zeros(1, seq, 4, dtype=torch.long)
    ctx = torch.randn(1, 8, 24)
    ctx_ids = torch.zeros(1, 8, 4, dtype=torch.long)
    # Precompute the RoPE frequencies via the model's RopeCache, as the adapter does.
    tiny_flux2.pe_embedder.clear()
    tiny_flux2.pe_embedder.store("img", x_ids)
    tiny_flux2.pe_embedder.store("txt", ctx_ids)
    return {
        "x": x,
        "pe_x": tiny_flux2.pe_embedder["img"],
        "timesteps": torch.tensor([0.5]),
        "ctx": ctx,
        "pe_ctx": tiny_flux2.pe_embedder["txt"],
    }


def test_prc_img_and_scatter_roundtrip():
    torch.manual_seed(0)
    latent = torch.randn(1, 8, 4, 4)
    x, x_ids = prc_img(latent)
    assert x.shape == (1, 16, 8)
    assert x_ids.shape == (1, 16, 4)
    # scatter back reconstructs the same grid.
    back = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
    assert back.shape == (1, 8, 4, 4)
    assert torch.allclose(back, latent)


def test_prc_txt_ids_shape():
    assert prc_txt(torch.randn(1, 512, 24))[1].shape == (1, 512, 4)


def test_flux2_forward_slices_the_reference_tokens_off_the_output(tiny_flux2):
    """End-to-end forward (no weights, random init) over the tiny config.

    The reference tokens are concatenated into the image stream and sliced back off,
    so the output is exactly the target tokens — and a reference-conditioned pass
    must differ, otherwise the conditioning is silently ignored.
    """
    torch.manual_seed(0)
    inputs = _tiny_inputs(tiny_flux2)
    with torch.no_grad():
        base = tiny_flux2(**inputs)
    assert base.shape == (1, 4, 8)  # a flow velocity, and finite
    assert torch.isfinite(base).all()

    ref_ids = torch.full((1, 6, 4), FluxKleinModel.REF_INDEX, dtype=torch.long)
    # pe_x must cover the concatenated image stream (base tokens + refs).
    tiny_flux2.pe_embedder.store("img", torch.cat([torch.zeros(1, 4, 4, dtype=torch.long), ref_ids], dim=1))
    inputs["pe_x"] = tiny_flux2.pe_embedder["img"]

    with torch.no_grad():
        edit = tiny_flux2(**inputs, ref_tokens=torch.randn(1, 6, 8))
    assert edit.shape == base.shape
    assert torch.isfinite(edit).all()
    assert not torch.allclose(edit, base)


def test_flux2_vae_decode_shape(flux2_vae):
    latents = torch.randn(1, 128, 4, 4)
    pixels = flux2_vae.decode_to_pixels(latents)
    # 16x spatial compression in packed space -> 4 -> 64 px.
    assert pixels.shape == (1, 3, 64, 64)
    assert pixels.min() >= -1.0 and pixels.max() <= 1.0


def test_flux2_vae_encode_shape(flux2_vae):
    """Encoder: pixels [-1,1] -> canonical packed latent (16x compression)."""
    pixels = torch.randn(1, 3, 64, 64)
    latents = flux2_vae.encode_pixels_to_latents(pixels)
    assert latents.shape == (1, 128, 4, 4)
    assert latents.dtype == torch.float32


@pytest.mark.parametrize("t_coord", [FluxKleinModel.REF_INDEX, 2 * FluxKleinModel.REF_INDEX])
def test_reference_ids_go_on_their_own_t_axis(t_coord):
    """Reference packing puts the index on the t-axis, not the still-image 0;
    multi-ref packing uses successive ones (10, 20, ...) per ComfyUI.
    """
    _, ids = prc_img(torch.randn(1, 8, 4, 4), t_coord=torch.tensor([t_coord]))
    assert ids.shape == (1, 16, 4)
    assert torch.all(ids[0, :, 0] == t_coord)


def test_pack_reference_latent_rejects_unsupported_method():
    """An unsupported ``ref_latents_method`` raises rather than being ignored."""
    model = FluxKleinModel.__new__(FluxKleinModel)  # no __init__ (no weights)
    model.device = "cpu"
    model.dtype = torch.float32
    with pytest.raises(ValueError, match="unsupported ref_latents_method"):
        model.pack_reference_latent(torch.randn(1, 8, 4, 4), method="crop")


def _tiny_edit_inputs(model, seq=4, refseq=6, txtlen=8):
    """Build the target-only / reference / text pe + tokens for a tiny Flux2."""
    x = torch.randn(1, seq, 8)
    ref = torch.randn(1, refseq, 8)
    ctx = torch.randn(1, txtlen, 24)
    model.pe_embedder.clear()
    model.pe_embedder.store("img", torch.zeros(1, seq, 4, dtype=torch.long))
    model.pe_embedder.store("ref", torch.full((1, refseq, 4), FluxKleinModel.REF_INDEX, dtype=torch.long))
    model.pe_embedder.store("txt", torch.zeros(1, txtlen, 4, dtype=torch.long))
    return {
        "x": x,
        "pe_x": model.pe_embedder["img"],
        "pe_ref": model.pe_embedder["ref"],
        "pe_txt": model.pe_embedder["txt"],
        "ref": ref,
        "ctx": ctx,
        "t": torch.tensor([0.5]),
    }


def test_kv_cache_fill_and_read_are_exact(tiny_flux2):
    """Fill (refs present) and read (refs dropped + cached K/V) == full sequence.

    The reference K/V are frozen on the first step; at the same t/x the read path
    must reproduce the full-sequence forward exactly, which pins the RoPE/pe
    slicing and the trailing-suffix cache.
    """
    torch.manual_seed(0)
    inp = _tiny_edit_inputs(tiny_flux2)
    with torch.no_grad():
        full = tiny_flux2(
            inp["x"], inp["pe_x"], inp["t"], inp["ctx"], inp["pe_txt"],
            ref_tokens=inp["ref"], ref_pe=inp["pe_ref"],
        )
        kv = KVCache("run")
        filled = tiny_flux2(
            inp["x"], inp["pe_x"], inp["t"], inp["ctx"], inp["pe_txt"],
            ref_tokens=inp["ref"], ref_pe=inp["pe_ref"], kv=kv,
        )
        # After the fill forward the cache is marked filled -> read mode.
        assert kv.filled
        read = tiny_flux2(
            inp["x"], inp["pe_x"], inp["t"], inp["ctx"], inp["pe_txt"],
            ref_tokens=None, kv=kv,
        )
    assert torch.allclose(filled, full, atol=1e-6), (filled - full).abs().max().item()
    assert torch.allclose(read, full, atol=1e-6), (read - full).abs().max().item()


def test_zero_cond_t_changes_the_edit_output_but_not_t2i(tiny_flux2):
    """``zero_cond_t`` modulates the reference slice with vec(0).

    It is a per-forward flag (the adapter derives it from the resolved
    ``ref_method``), so one set of weights serves both: with references the output
    differs, with no references (plain t2i) the flag cannot matter.
    """
    torch.manual_seed(0)
    m = tiny_flux2
    inp = _tiny_edit_inputs(m)
    with torch.no_grad():
        # t2i: identical (no refs -> no t=0 split).
        out_plain_t2i = m(inp["x"], inp["pe_x"], inp["t"], inp["ctx"], inp["pe_txt"],
                          zero_cond_t=False)
        out_zero_t2i = m(inp["x"], inp["pe_x"], inp["t"], inp["ctx"], inp["pe_txt"],
                         zero_cond_t=True)
        assert torch.allclose(out_plain_t2i, out_zero_t2i, atol=1e-6)
        # edit: the refs are modulated differently -> outputs differ.
        out_plain = m(inp["x"], inp["pe_x"], inp["t"], inp["ctx"], inp["pe_txt"],
                      ref_tokens=inp["ref"], ref_pe=inp["pe_ref"], zero_cond_t=False)
        out_zero = m(inp["x"], inp["pe_x"], inp["t"], inp["ctx"], inp["pe_txt"],
                     ref_tokens=inp["ref"], ref_pe=inp["pe_ref"], zero_cond_t=True)
        assert not torch.allclose(out_plain, out_zero)


def test_zero_cond_t_keeps_refs_step_independent(tiny_flux2):
    """The cache's premise, measured: over a short run the cached trajectory drifts
    far less with the references conditioned at t=0 than following the timestep."""
    torch.manual_seed(0)
    m = tiny_flux2
    inp = _tiny_edit_inputs(m)

    def drift(zero_cond_t):
        torch.manual_seed(0)
        x_full = torch.randn(1, 4, 8)
        x_cache = x_full.clone()
        kv = KVCache("c")
        for i, t_i in enumerate([0.9, 0.7, 0.5]):
            ts = torch.tensor([t_i])
            with torch.no_grad():
                x_full = x_full - 0.2 * m(x_full, inp["pe_x"], ts, inp["ctx"], inp["pe_txt"],
                                          ref_tokens=inp["ref"], ref_pe=inp["pe_ref"],
                                          zero_cond_t=zero_cond_t)
                if not kv.filled:
                    x_cache = x_cache - 0.2 * m(x_cache, inp["pe_x"], ts, inp["ctx"], inp["pe_txt"],
                                                ref_tokens=inp["ref"], ref_pe=inp["pe_ref"],
                                                kv=kv, zero_cond_t=zero_cond_t)
                else:
                    x_cache = x_cache - 0.2 * m(x_cache, inp["pe_x"], ts, inp["ctx"], inp["pe_txt"],
                                                ref_tokens=None, kv=kv, zero_cond_t=zero_cond_t)
        return (x_full - x_cache).abs().max().item()

    drift_zero = drift(True)
    drift_index = drift(False)
    assert drift_zero < drift_index


def test_kv_cache_freed_in_finalize():
    """``finalize_latent`` drops the run-scoped KV cache (no stale cache across runs)."""
    from thenoise.models.config import SamplingParams

    model = FluxKleinModel.__new__(FluxKleinModel)
    model._kv_caches = {"cond": KVCache("cond"), "uncond": KVCache("uncond")}
    model._img_ids = torch.zeros(1, 4, 4, dtype=torch.long)
    params = SamplingParams(height=64, width=64, steps=4, seed=0, guidance_scale=1.0, sampler="euler")
    model.finalize_latent(torch.randn(1, 4, 8), params)
    assert model._kv_caches is None
    assert model.kv_cache("cond") is None  # the DiT's plain path from here on


def test_kv_cache_multi_ref_caches_the_sum(tiny_flux2):
    """Two references are one stream: the cached slice is their combined length."""
    torch.manual_seed(0)
    m = tiny_flux2
    inp = _tiny_edit_inputs(m, seq=4, refseq=3, txtlen=8)
    ref_all = torch.randn(1, 8, 8)
    m.pe_embedder.store("ref", torch.zeros(1, 8, 4, dtype=torch.long))
    with torch.no_grad():
        full = m(inp["x"], inp["pe_x"], inp["t"], inp["ctx"], inp["pe_txt"],
                 ref_tokens=ref_all, ref_pe=m.pe_embedder["ref"])
        kv = KVCache("r")
        filled = m(inp["x"], inp["pe_x"], inp["t"], inp["ctx"], inp["pe_txt"],
                   ref_tokens=ref_all, ref_pe=m.pe_embedder["ref"], kv=kv)
        read = m(inp["x"], inp["pe_x"], inp["t"], inp["ctx"], inp["pe_txt"],
                 ref_tokens=None, kv=kv)
    assert kv.filled
    assert torch.allclose(filled, full, atol=1e-6)
    assert torch.allclose(read, full, atol=1e-6)


def test_adapter_denoise_step_kv_fill_then_read():
    """The adapter drives the KV cache through ``denoise_step``: step 0 fills,
    step 1 reads, and the two velocities differ (different timestep)."""
    model = FluxKleinModel.__new__(FluxKleinModel)
    model.device = "cpu"
    model.dtype = torch.float32
    model.dit = Flux2(_tiny_params()).eval().requires_grad_(False)
    model.dit.pe_embedder.clear()
    model.dit.pe_embedder.store("img", torch.zeros(1, 4, 4, dtype=torch.long))
    model.dit.pe_embedder.store("ref", torch.full((1, 6, 4), FluxKleinModel.REF_INDEX, dtype=torch.long))
    model.dit.pe_embedder.store("txt", torch.zeros(1, 8, 4, dtype=torch.long))
    model._kv_caches = {"cond": KVCache("cond")}
    model._txt = torch.randn(1, 8, 24)
    model._un_txt = None
    model._ref_tokens = torch.randn(1, 6, 8)
    model._zero_cond_t = True  # what ``prepare_latent`` stashes for index_timestep_zero

    from thenoise.models.base import Conditioning
    cond = Conditioning(cond=model._txt)
    latents = torch.randn(1, 4, 8)
    with torch.no_grad():
        v0 = model.denoise_step(latents, 0.5, cond, 1.0, 0)
        assert model.kv_cache("cond").filled  # step 0 filled the cache
        v1 = model.denoise_step(latents, 0.4, cond, 1.0, 1)
    assert not torch.allclose(v0, v1)  # different t -> different velocity
