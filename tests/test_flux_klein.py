"""Flux.2 (Flux Klein) adapter tests (no real weights / no GPU needed).

Covers the flow schedule, the token-position pack/unpack helpers, a small
end-to-end DiT forward (eager — see ``conftest``), the Flux.2 VAE shapes, the
reference-token packing and the Flux2 latent-upscaler round-trip.

Detection and the per-model defaults live in the catalog-wide tables of
``test_detect.py`` / ``test_catalog.py``.
"""
from __future__ import annotations

import pytest
import torch

from thenoise.dit.flux2.models import Flux2, Flux2Params
from thenoise.dit.flux2.sampling import get_schedule, prc_img, prc_txt, scatter_ids
from thenoise.models import FluxKleinModel


@pytest.fixture
def tiny_flux2():
    """A random-init Flux2 with the smallest sensible config."""
    torch.manual_seed(0)
    return Flux2(
        Flux2Params(
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
    ).eval()


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


def test_schedule_is_flow_grid_1_to_0():
    ts = get_schedule(8, 4096)  # 1024x1024 -> 64x64 packed -> 4096 tokens
    assert len(ts) == 9  # num_steps + 1
    assert ts[0] == 1.0
    assert ts[-1] == 0.0
    # Strictly decreasing.
    assert all(ts[i] > ts[i + 1] for i in range(len(ts) - 1))


def test_schedule_depends_on_token_count():
    ts_small = get_schedule(8, 256)
    ts_large = get_schedule(8, 4096)
    # Larger image token counts get a larger empirical shift (steeper early steps).
    assert ts_small[1] < ts_large[1]


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
    txt = torch.randn(1, 512, 24)
    _, ids = prc_txt(txt)
    assert ids.shape == (1, 512, 4)


def test_flux2_forward_small_model(tiny_flux2):
    """End-to-end Flux2 forward with a tiny config (no weights, random init)."""
    out = tiny_flux2(**_tiny_inputs(tiny_flux2))
    assert out.shape == (1, 4, 8)
    # A flow-velocity output must be finite.
    assert torch.isfinite(out).all()


def test_flux2_forward_with_reference_tokens(tiny_flux2):
    """Flux2 forward consumes ref tokens and slices them off the output."""
    torch.manual_seed(0)
    inputs = _tiny_inputs(tiny_flux2)
    ref_tokens = torch.randn(1, 6, 8)
    ref_ids = torch.full((1, 6, 4), FluxKleinModel.REF_INDEX, dtype=torch.long)
    # pe_x must cover the concatenated image stream (base tokens + refs).
    img_ids = torch.cat([torch.zeros(1, 4, 4, dtype=torch.long), ref_ids], dim=1)
    tiny_flux2.pe_embedder.store("img", img_ids)
    inputs["pe_x"] = tiny_flux2.pe_embedder["img"]

    with torch.no_grad():
        out = tiny_flux2(**inputs, ref_tokens=ref_tokens)
    # The reference tokens are concatenated in and sliced back off, so the
    # output is exactly the image tokens (seq), not seq + ref tokens.
    assert out.shape == (1, 4, 8)
    assert torch.isfinite(out).all()

    # The differential: a reference-conditioned pass must differ from the
    # un-conditioned one (otherwise conditioning is silently ignored).
    with torch.no_grad():
        base = tiny_flux2(**_tiny_inputs(tiny_flux2))
    assert not torch.allclose(out, base)


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


def test_flux2_reference_ids_use_the_ref_index_t_coord():
    """Reference packing puts ``REF_INDEX`` on the t-axis, not the still-image 0."""
    ref = torch.randn(1, 8, 4, 4)
    _, ids = prc_img(ref, t_coord=torch.tensor([FluxKleinModel.REF_INDEX]))
    assert ids.shape == (1, 16, 4)
    assert torch.all(ids[0, :, 0] == FluxKleinModel.REF_INDEX)


def test_multi_ref_packing_assigns_successive_indices():
    """Multi-ref packing uses successive t-axes (10, 20, ...) per ComfyUI."""
    ref = torch.randn(1, 8, 4, 4)
    index = FluxKleinModel.REF_INDEX
    _, ids1 = prc_img(ref, t_coord=torch.tensor([index]))
    _, ids2 = prc_img(ref, t_coord=torch.tensor([2 * index]))
    assert torch.all(ids1[0, :, 0] == index)
    assert torch.all(ids2[0, :, 0] == 2 * index)
    # Concatenated along the token dimension keeps both refs distinct.
    assert torch.cat([ids1, ids2], dim=1).shape == (1, 32, 4)


def test_pack_reference_latent_rejects_unsupported_method():
    """An unsupported ``ref_latents_method`` raises rather than being ignored."""
    model = FluxKleinModel.__new__(FluxKleinModel)  # no __init__ (no weights)
    model.device = "cpu"
    model.dtype = torch.float32
    with pytest.raises(ValueError, match="unsupported ref_latents_method"):
        model.pack_reference_latent(torch.randn(1, 8, 4, 4), method="crop")


def test_resize_to_cover_center_crop_keeps_target_size():
    """ComfyUI-style ref resize: cover the target, center-crop; no padding."""
    from PIL import Image

    from thenoise.utils.image_tensor import resize_to_cover_center_crop

    # Wide source into a square target: scale height to 100, width overflows, crop.
    assert resize_to_cover_center_crop(Image.new("RGB", (200, 50), "red"), 100, 100).size == (100, 100)
    # Same aspect ratio: only resized, no crop.
    assert resize_to_cover_center_crop(Image.new("RGB", (200, 100), "blue"), 100, 50).size == (100, 50)
    # Already at target size: returned unchanged.
    img = Image.new("RGB", (64, 64), "green")
    assert resize_to_cover_center_crop(img, 64, 64) is img


def test_flux2_upscaler_loads_and_runs():
    from thenoise.upscale import load_latent_upscaler

    model, adaptor = load_latent_upscaler("flux2", device="cpu", dtype=torch.bfloat16)
    # Canonical Flux Klein latent [B, 128, H//16, W//16] -> raw 32ch VAE latent.
    z = torch.randn(1, 128, 8, 8)
    raw = adaptor.to_vae_latent(z).to(torch.bfloat16)
    assert raw.shape == (1, 32, 16, 16)
    # 2x upscale in external coords -> the raw latent target goes through the
    # adaptor's spatial scale (2), i.e. raw 16x16 -> 32x32.
    target = adaptor.vae_target_size((16, 16))
    out = model(raw, target)
    z_up = adaptor.from_vae_latent(out.float())
    assert z_up.shape == (1, 128, 16, 16)
