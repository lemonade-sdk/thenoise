"""Qwen-Image adapter tests (no real weights / no GPU needed).

Covers the flow schedule, the token pack/unpack helpers, the vendored
tokenizer config directory, the Qwen-Image latent format, the reference-latent
packing rejection, a small end-to-end DiT forward with the reference-latent KV
cache (eager — see ``conftest``) and the adapter's fill/read driving of it.

Detection and the per-model defaults live in the catalog-wide tables of
``test_detect.py`` / ``test_catalog.py``.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from thenoise.dit.kvcache import KVCache
from thenoise.dit.qwen_image import sampling as qwen_sampling
from thenoise.dit.qwen_image import utils as qwen_utils
from thenoise.dit.qwen_image.models import (
    QwenImageTransformer2DModel,
    build_txt_positions,
    build_video_positions,
)
from thenoise.models.qwen_image import QwenImageModel
from thenoise.models.config import SamplingParams
from thenoise.utils.image_tensor import resize_to_area
from thenoise.utils.latents import pack_latents, unpack_latents
from thenoise.utils.math import calculate_shift
from thenoise.utils.text_encoder import QWEN25_TOKENIZER_CONFIG_DIR
from thenoise.upscale import make_wan21
from thenoise.vae import AutoencoderKLQwenImage


# ---------------------------------------------------------------- flow schedule


def test_schedule_is_flow_grid_1_to_0():
    ts = qwen_sampling.get_schedule(8, 4096)  # 1024x1024 -> 64x64 packed -> 4096 tokens
    assert len(ts) == 9  # num_steps + 1
    assert ts[0] == 1.0
    assert ts[-1] == 0.0
    # Strictly decreasing.
    assert all(ts[i] > ts[i + 1] for i in range(len(ts) - 1))


def test_schedule_depends_on_token_count():
    ts_small = qwen_sampling.get_schedule(8, 256)
    ts_large = qwen_sampling.get_schedule(8, 4096)
    # A larger image token count gets a larger empirical shift (steeper early steps).
    assert ts_small[1] < ts_large[1]


def test_single_step_schedule_is_finite():
    """``steps == 1`` is the degenerate case: the terminal stretch divides by zero.

    The guard in ``get_sigmas`` leaves the grid untouched so a 1-step denoise
    starts at exactly pure noise (t=1.0) instead of producing ``nan``.
    """
    sigmas = qwen_sampling.get_sigmas(1, 256, qwen_sampling.compute_mu(256))
    assert torch.isfinite(sigmas).all()
    assert sigmas[0] == 1.0


def test_calculate_shift_increases_with_token_count():
    assert calculate_shift(256) < calculate_shift(4096)


# ------------------------------------------------------------------ latents


def test_pack_unpack_latents_roundtrip():
    torch.manual_seed(0)
    latent = torch.randn(1, 16, 4, 4)
    packed = pack_latents(latent)
    # 4x4 grid -> 2x2 patch blocks -> 4 tokens, 16ch * 4 = 64 features each.
    assert packed.shape == (1, 4, 64)
    back = unpack_latents(packed, 4, 4)
    assert back.shape == (1, 16, 4, 4)
    assert torch.allclose(back, latent)


def test_pack_latents_accepts_frame_axis():
    """The edit path feeds a ``[B, C, 1, H, W]`` reference latent."""
    latent = torch.randn(1, 16, 1, 4, 4)
    packed = pack_latents(latent)
    assert packed.shape == (1, 4, 64)


# ------------------------------------------------------------- image resize


def test_resize_to_area():
    from PIL import Image

    big = Image.new("RGB", (500, 1000))
    out = resize_to_area(big, area=384 * 384)
    # Scaled down substantially (area-based), aspect preserved (1:2).
    assert out.width * out.height < big.width * big.height
    assert abs(out.width / out.height - 0.5) < 0.01

# ------------------------------------------------------------- vendored config


def test_vendored_tokenizer_config_dir_exists():
    # The tokenizer config files are checked into the package so the tokenizer and
    # processor load offline without fetching from the Hub (mirrors the anima/zimage
    # ``configs/`` pattern).
    from pathlib import Path

    d = Path(QWEN25_TOKENIZER_CONFIG_DIR)
    assert d.is_dir()
    for required in ("tokenizer.json", "tokenizer_config.json"):
        assert (d / required).is_file(), f"missing vendored tokenizer file {required}"


# ---------------------------------------------------------------- latent format


def test_qwen_upscale_format_is_wan21():
    # Qwen-Image uses the shared Wan21 z-score latent format; the adaptor's
    # per-channel mean/std must match the VAE's own encode/decode normalization.
    adaptor = make_wan21()
    vae = AutoencoderKLQwenImage()
    assert torch.allclose(adaptor.mean.view(-1), torch.tensor(vae.latents_mean))
    assert torch.allclose(adaptor.std.view(-1), torch.tensor(vae.latents_std))


# ----------------------------------------------------------------- reference


def test_pack_reference_latent_rejects_unsupported_method():
    """An unsupported ``ref_latents_method`` raises rather than being ignored."""
    model = QwenImageModel.__new__(QwenImageModel)  # no __init__ (no weights)
    model.device = "cpu"
    model.dtype = torch.float32
    with pytest.raises(ValueError, match="unsupported ref_latents_method"):
        model.pack_reference_latent(torch.randn(1, 16, 4, 4), method="crop")


def test_pack_reference_latent_accepts_both_methods():
    """``index_timestep_zero`` packs like ``index`` (only the modulation differs).

    Both reference methods must be accepted: an edit checkpoint carrying the
    ``__index_timestep_zero__`` marker now auto-resolves to it, so rejecting it
    would break the automatic preference layer.
    """
    model = QwenImageModel.__new__(QwenImageModel)
    model.device = "cpu"
    model.dtype = torch.float32
    ref = torch.randn(1, 16, 4, 4)
    index_tokens, _ = model.pack_reference_latent(ref, method="index")
    zero_tokens, _ = model.pack_reference_latent(ref, method="index_timestep_zero")
    assert torch.equal(index_tokens, zero_tokens)


# --------------------------------------------------------------- DiT forward
#
# The tiny config is the smallest shape the DiT accepts: 8-channel packed tokens,
# two heads of 8 dims (so the 3-axis RoPE dims sum to 8), one block, and a 12-wide
# text stream. ``out_channels`` 2 with the 2x2 patchify makes ``proj_out`` 8-wide.

TINY_QWEN = dict(
    patch_size=2,
    in_channels=8,
    out_channels=2,
    num_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=12,
    axes_dims_rope=(2, 2, 4),
)


@pytest.fixture
def tiny_qwen():
    """A random-init Qwen-Image DiT with the smallest sensible config."""
    torch.manual_seed(0)
    return QwenImageTransformer2DModel(**TINY_QWEN).eval()


def _tiny_inputs(model, target=(2, 2), refs=(), txt_len=5):
    """Target/reference tokens plus the RoPE positions an adapter would store.

    ``refs`` are ``(h, w)`` reference grids: their tokens are appended after the
    target's and their positions stored apart (under ``ref``), which is what lets a
    cached run drop them from the sequence once their K/V are frozen.
    """
    th, tw = target
    num_img_tokens = th * tw
    shapes = [(1, th, tw)] + [(1, h, w) for h, w in refs]
    pos = build_video_positions(shapes, device="cpu")
    model.pe_embedder.clear()
    model.pe_embedder.store("img", pos[:, :num_img_tokens], dtype=torch.float32)
    model.pe_embedder.store("ref", pos[:, num_img_tokens:], dtype=torch.float32)
    max_vid_index = max(max(h // 2, w // 2) for _, h, w in shapes)
    model.pe_embedder.store(
        "txt", build_txt_positions(max_vid_index, txt_len, device="cpu"), dtype=torch.float32
    )
    ref_tokens = [torch.randn(1, h * w, 8) for h, w in refs]
    return {
        "x": torch.randn(1, num_img_tokens, 8),
        "ref": torch.cat(ref_tokens, dim=1) if ref_tokens else None,
        "ctx": torch.randn(1, txt_len, 12),
        "t": torch.tensor([0.5]),
        "num_img_tokens": num_img_tokens,
    }


def _forward(model, inp, *, refs=True, kv=None, zero_cond_t=True, x=None, t=None):
    """One DiT forward, as the adapter drives it on an ``index_timestep_zero`` edit."""
    return model(
        inp["x"] if x is None else x,
        inp["ctx"],
        inp["t"] if t is None else t,
        img_pe=model.pe_embedder["img"],
        txt_pe=model.pe_embedder["txt"],
        ref_tokens=inp["ref"] if refs else None,
        ref_pe=model.pe_embedder["ref"],
        kv=kv,
        timestep_zero_index=inp["num_img_tokens"] if zero_cond_t else None,
    )


def test_forward_without_references_is_plain_t2i(tiny_qwen):
    inp = _tiny_inputs(tiny_qwen)
    with torch.no_grad():
        out = _forward(tiny_qwen, inp, refs=False)
    assert out.shape == (1, inp["num_img_tokens"], 8)
    assert torch.isfinite(out).all()
    # Timestep-zero conditioning cannot matter when there is nothing to condition.
    with torch.no_grad():
        index_edit = _forward(tiny_qwen, inp, refs=False, zero_cond_t=False)
    assert torch.allclose(out, index_edit, atol=1e-6)


def test_forward_slices_the_reference_tokens_off_the_output(tiny_qwen):
    """References are appended to the image stream and dropped before the output head."""
    inp = _tiny_inputs(tiny_qwen, refs=((2, 3),))
    with torch.no_grad():
        edit = _forward(tiny_qwen, inp)
        t2i = _forward(tiny_qwen, inp, refs=False)
    assert edit.shape == t2i.shape == (1, inp["num_img_tokens"], 8)
    assert torch.isfinite(edit).all()
    # The differential: a reference-conditioned pass must differ, otherwise the
    # conditioning is silently ignored.
    assert not torch.allclose(edit, t2i)


def test_kv_cache_fill_and_read_are_exact(tiny_qwen):
    """Fill (refs present) then read (refs dropped + cached K/V) == the full sequence.

    The reference K/V are frozen by the fill pass, so at the same t/x the read path
    must reproduce the full-sequence forward exactly — which pins the joint
    ``text, target, references`` token layout, the RoPE slicing and the trailing
    suffix the buffers keep for the references.
    """
    inp = _tiny_inputs(tiny_qwen, refs=((2, 3),))
    with torch.no_grad():
        full = _forward(tiny_qwen, inp)
        kv = KVCache("run")
        filled = _forward(tiny_qwen, inp, kv=kv)
        assert kv.filled
        read = _forward(tiny_qwen, inp, refs=False, kv=kv)
    assert torch.allclose(filled, full, atol=1e-6), (filled - full).abs().max().item()
    assert torch.allclose(read, full, atol=1e-6), (read - full).abs().max().item()
    # One pair of buffers for the single block, sized to the whole joint sequence.
    assert len(kv._buffers) == 1
    assert kv._buffers[0].capacity == 5 + inp["num_img_tokens"] + 6


def test_kv_cache_multi_ref_caches_the_sum(tiny_qwen):
    """Several references are one stream: the cached slice is their combined length."""
    inp = _tiny_inputs(tiny_qwen, refs=((1, 3), (1, 5)))
    assert inp["ref"].shape[1] == 8
    with torch.no_grad():
        full = _forward(tiny_qwen, inp)
        kv = KVCache("r")
        filled = _forward(tiny_qwen, inp, kv=kv)
        read = _forward(tiny_qwen, inp, refs=False, kv=kv)
    assert kv.filled
    assert torch.allclose(filled, full, atol=1e-6)
    assert torch.allclose(read, full, atol=1e-6)


def test_kv_cache_rejects_a_read_longer_than_the_fill():
    """The capacity guard is what keeps the block's prefix copy from overrunning."""
    kv = KVCache("run")
    kv.allocate(0, (1, 2, 6, 8), torch.float32, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="different sequence length"):
        kv.get(0, 7)
    with pytest.raises(KeyError, match="must fill the cache before reading"):
        KVCache("other").get(0, 1)


def test_timestep_zero_keeps_the_reference_kv_step_invariant(tiny_qwen):
    """Only under timestep-zero conditioning are the reference K/V step-invariant.

    The cache's premise, measured: over a short run the cached-vs-full trajectories
    must stay far closer when the references are conditioned at t=0 than when they
    follow the timestep like the target tokens do.
    """
    inp = _tiny_inputs(tiny_qwen, refs=((2, 3),))

    def drift(zero_cond_t):
        torch.manual_seed(0)
        x_full = torch.randn(1, inp["num_img_tokens"], 8)
        x_cache = x_full.clone()
        kv = KVCache("c")
        for t_i in (0.9, 0.7, 0.5):
            ts = torch.tensor([t_i])
            with torch.no_grad():
                x_full = x_full - 0.2 * _forward(
                    tiny_qwen, inp, x=x_full, t=ts, zero_cond_t=zero_cond_t
                )
                x_cache = x_cache - 0.2 * _forward(
                    tiny_qwen, inp, refs=not kv.filled, kv=kv, x=x_cache, t=ts,
                    zero_cond_t=zero_cond_t,
                )
        return (x_full - x_cache).abs().max().item()

    assert drift(True) < drift(False)


# ------------------------------------------------------- adapter KV cache wiring


def _bare_adapter(dit=None):
    """A :class:`QwenImageModel` without ``__init__``, wired for the denoise kernels."""
    from types import SimpleNamespace

    model = QwenImageModel.__new__(QwenImageModel)
    model.device = "cpu"
    model.dtype = torch.float32
    model.dit = dit
    model.vae = SimpleNamespace(spatial_compression=8, z_dim=2)
    model._txt = None
    model._null_txt = None
    model._ref_tokens = None
    model._timestep_zero_index = None
    return model


def _params(**overrides):
    params = SamplingParams(
        height=16, width=16, steps=4, seed=0, guidance_scale=2.5, sampler="euler"
    )
    return replace(params, **overrides) if overrides else params


def test_kv_cache_freed_in_finalize():
    """``finalize_latent`` drops the run-scoped caches (no stale cache across runs)."""
    model = _bare_adapter()
    model._kv_caches = {"cond": KVCache("cond"), "uncond": KVCache("uncond")}
    # 16x16 pixels on an 8x VAE + 2x2 patchify -> one packed token of 8 channels.
    model.finalize_latent(torch.randn(1, 1, 8), _params())
    assert model._kv_caches is None
    assert model.kv_cache("cond") is None  # the DiT's plain path from here on


def test_kv_caches_are_started_only_for_a_cached_edit_run():
    """``start_kv_caches`` gates on the request, the reference and CFG (base protocol)."""
    model = _bare_adapter()

    model.start_kv_caches(_params(kv_cache=True), has_reference=True, has_uncond=True)
    assert set(model._kv_caches) == {"cond", "uncond"}
    assert model.kv_cache("cond") is model._kv_caches["cond"]

    # CFG off: no uncond branch, so no second cache to allocate buffers in.
    model.start_kv_caches(_params(kv_cache=True), has_reference=True, has_uncond=False)
    assert set(model._kv_caches) == {"cond"}

    # No reference latent -> nothing step-invariant to freeze.
    model.start_kv_caches(_params(kv_cache=True), has_reference=False, has_uncond=True)
    assert model._kv_caches is None

    # ``kv_cache`` off (the default) never starts one either.
    model.start_kv_caches(_params(), has_reference=True, has_uncond=True)
    assert model.kv_cache("cond") is None


def test_denoise_step_fills_then_reads(tiny_qwen):
    """The adapter drives the cache through ``denoise_step``: step 0 fills (refs in
    the sequence), step 1 reads (refs dropped) and the velocities differ."""
    model = _bare_adapter(dit=tiny_qwen)
    inp = _tiny_inputs(tiny_qwen, refs=((2, 3),))
    model._txt = inp["ctx"]
    model._ref_tokens = inp["ref"]
    model._timestep_zero_index = inp["num_img_tokens"]  # what ``prepare_latent`` stashes
    model._kv_caches = {"cond": KVCache("cond")}

    from thenoise.models.base import Conditioning

    cond = Conditioning(cond=model._txt)
    with torch.no_grad():
        v0 = model.denoise_step(inp["x"], 0.5, cond, 1.0, 0)
        assert model.kv_cache("cond").filled  # step 0 ran the full sequence
        v1 = model.denoise_step(inp["x"], 0.4, cond, 1.0, 1)
    assert v0.shape == (1, inp["num_img_tokens"], 8)
    assert not torch.allclose(v0, v1)  # different t -> different velocity
