"""Flux.2 (Flux Klein) adapter tests (no real weights / no GPU needed).

Covers detection, the flow schedule, the token-position pack/unpack helpers, a
small end-to-end DiT forward, the Flux.2 VAE decode, the variant picker, and the
Flux2 latent-upscaler registration.
"""
from __future__ import annotations

import torch

from thenoise.dit.flux2.models import Flux2, Flux2Params
from thenoise.dit.flux2.sampling import get_schedule, prc_img, prc_txt, scatter_ids
from thenoise.dit.flux2.utils import QWEN3_8B_CONFIG
from thenoise.models import FluxKleinModel, resolve
from thenoise.vae import AutoencoderKLFlux2


class _FakeHandle:
    def __init__(self, keys):
        self._keys = keys

    def keys(self):
        return self._keys


_FLUX_KLEIN_KEYS = [
    "double_stream_modulation_img.lin.weight",
    "double_stream_modulation_txt.lin.weight",
    "single_stream_modulation.lin.weight",
    "img_in.weight",
    "txt_in.weight",
    "final_layer.adaLN_modulation.1.weight",
]

_FLUX_KLEIN_WRAPPED_KEYS = [
    "model.diffusion_model.double_stream_modulation_img.lin.weight",
    "model.diffusion_model.double_stream_modulation_txt.lin.weight",
    "model.diffusion_model.single_stream_modulation.lin.weight",
    "model.diffusion_model.img_in.weight",
    "model.diffusion_model.final_layer.adaLN_modulation.1.weight",
]


def test_detect_flux_klein():
    assert FluxKleinModel.detect(_FakeHandle(_FLUX_KLEIN_KEYS)) is True


def test_detect_flux_klein_wrapped():
    assert FluxKleinModel.detect(_FakeHandle(_FLUX_KLEIN_WRAPPED_KEYS)) is True


def test_detect_rejects_other_models():
    # Z-Image keys must not be claimed by Flux Klein (and vice versa).
    from tests.test_detect import _ANIMA_KEYS, _KREA2_KEYS, _ZIMAGE_KEYS

    assert FluxKleinModel.detect(_FakeHandle(_ANIMA_KEYS)) is False
    assert FluxKleinModel.detect(_FakeHandle(_KREA2_KEYS)) is False
    assert FluxKleinModel.detect(_FakeHandle(_ZIMAGE_KEYS)) is False

    from thenoise.models import AnimaModel, Krea2Model, ZImageModel

    assert AnimaModel.detect(_FakeHandle(_FLUX_KLEIN_KEYS)) is False
    assert Krea2Model.detect(_FakeHandle(_FLUX_KLEIN_KEYS)) is False
    assert ZImageModel.detect(_FakeHandle(_FLUX_KLEIN_KEYS)) is False


def test_resolve_flux_klein(tmp_path):
    import torch
    from safetensors.torch import save_file

    p = tmp_path / "klein.safetensors"
    save_file({k: torch.zeros(1) for k in _FLUX_KLEIN_WRAPPED_KEYS}, str(p))
    assert resolve(str(p)) is FluxKleinModel


def test_flux_klein_defaults():
    # Distilled defaults: 4 steps, guidance 1.0 (CFG off), Euler sampler.
    assert FluxKleinModel.DEFAULT_STEPS == 4
    assert FluxKleinModel.DEFAULT_GUIDANCE_SCALE == 1.0
    assert FluxKleinModel.SAMPLER == "euler"
    assert FluxKleinModel.LATENT_CHANNELS == 128


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


def test_flux2_forward_small_model():
    """End-to-end Flux2 forward with a tiny config (no weights, random init)."""
    torch.manual_seed(0)
    params = Flux2Params(
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
    model = Flux2(params)
    model.eval()

    seq = 4
    x = torch.randn(1, seq, 8)
    x_ids = torch.zeros(1, seq, 4, dtype=torch.long)
    t = torch.tensor([0.5])
    ctx = torch.randn(1, 8, 24)
    ctx_ids = torch.zeros(1, 8, 4, dtype=torch.long)

    with torch.no_grad():
        out = model(x=x, x_ids=x_ids, timesteps=t, ctx=ctx, ctx_ids=ctx_ids)
    assert out.shape == (1, seq, 8)
    # A flow-velocity output must be finite.
    assert torch.isfinite(out).all()


def test_flux2_vae_decode_shape():
    vae = AutoencoderKLFlux2()
    latents = torch.randn(1, 128, 4, 4)
    pixels = vae.decode_to_pixels(latents)
    # 16x spatial compression in packed space -> 4 -> 64 px.
    assert pixels.shape == (1, 3, 64, 64)
    assert pixels.min() >= -1.0 and pixels.max() <= 1.0


def test_qwen3_8b_config_matches_klein9b_context():
    # Klein 9B context = 3 * Qwen3-8B hidden (4096).
    assert QWEN3_8B_CONFIG["hidden_size"] == 4096


def test_flux2_upscale_format_registered():
    from thenoise.upscale import _UPSCALER_FORMATS, load_latent_upscaler

    assert "flux2" in _UPSCALER_FORMATS
    assert _UPSCALER_FORMATS["flux2"][1] == "upscaler_flux2.safetensors"
    assert _UPSCALER_FORMATS["flux2"][2] == 32  # raw VAE latent channels

    # The model declares the flux2 format for its upscale path (abstract method,
    # needs an instance; check it exists on the class).
    assert hasattr(FluxKleinModel, "_upscale_format")
    assert "flux2" in _UPSCALER_FORMATS


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
