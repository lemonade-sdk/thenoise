"""The VAE round-trip latent upscaler (no weights, fake VAE).

Nothing here loads a VAE: ``VAEPixelUpscaler`` adds no weights of its own, so a
fake one that speaks the engine's VAE interface covers the whole strategy — the
order of the round trip, the resolution it resizes to, the range it encodes back,
the dtypes and the channel width it refuses to drop. The wiring into a model is
covered in ``test_qwen_image21.py``; the pipeline drives every strategy through the
same ``LatentUpscaler`` mock (see ``conftest``).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from thenoise.upscale import LatentUpscaler, VAEPixelUpscaler
from thenoise.upscale import vae_pixel as vae_pixel_mod


class FakeVAE:
    """A weight-free stand-in for the engine's VAE interface.

    The decode emits a horizontal ramp (or a hard checkerboard, which forces the
    bicubic to overshoot) so a test can tell *which* pixels the encoder got back,
    and the encoder records them instead of compressing them. Geometry — latent
    width, pixel width, compression, dtype, device — is what the upscaler reads,
    so it is all settable.
    """

    def __init__(
        self,
        *,
        z_dim: int = 4,
        pixel_channels: int = 4,
        spatial_compression: int = 8,
        dtype: torch.dtype = torch.float32,
        pattern: str = "ramp",
        video_layout: bool = False,
    ):
        self.z_dim = z_dim
        self.pixel_channels = pixel_channels
        self.spatial_compression = spatial_compression
        self.dtype = dtype
        self.device = torch.device("cpu")
        self.pattern = pattern
        self.video_layout = video_layout

        self.calls: list[str] = []
        self.decode_input: torch.Tensor | None = None
        self.encoded_pixels: torch.Tensor | None = None

    def decode_to_pixels(self, z: torch.Tensor) -> torch.Tensor:
        self.calls.append("decode")
        self.decode_input = z
        b, _, h, w = z.shape
        p = self.spatial_compression
        h_pix, w_pix = h * p, w * p
        if self.pattern == "checker":
            ys = torch.arange(h_pix).view(-1, 1)
            xs = torch.arange(w_pix).view(1, -1)
            grid = torch.where(
                (xs + ys) % 2 == 0,
                torch.tensor(1.0),
                torch.tensor(-1.0),
            )
        else:
            grid = torch.linspace(-1.0, 1.0, w_pix).expand(h_pix, w_pix)
        pixels = grid.view(1, 1, h_pix, w_pix).expand(
            b, self.pixel_channels, h_pix, w_pix
        ).contiguous().to(self.dtype)
        if self.video_layout:  # [B, C, 1, H, W] (the family's video layout)
            pixels = pixels.unsqueeze(2)
        return pixels

    def encode_pixels_to_latents(self, pixels: torch.Tensor) -> torch.Tensor:
        self.calls.append("encode")
        self.encoded_pixels = pixels
        b, _, h, w = pixels.shape
        p = self.spatial_compression
        return torch.full(
            (b, self.z_dim, h // p, w // p), 0.75, dtype=pixels.dtype, device=pixels.device
        )


def _upscaler(vae=None, scale: int = 2, **vae_kwargs) -> VAEPixelUpscaler:
    return VAEPixelUpscaler(vae or FakeVAE(**vae_kwargs), scale=scale)


def _latent(vae: FakeVAE, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros(1, vae.z_dim, 4, 6, dtype=dtype)


# --------------------------------------------------------------- the round trip


def test_the_round_trip_is_decode_then_resize_then_encode():
    """The latent is decoded, its pixels scaled, and the result re-encoded."""
    vae = FakeVAE()
    z_up = _upscaler(vae)(torch.zeros(1, vae.z_dim, 4, 6))

    assert vae.calls == ["decode", "encode"]
    assert z_up.shape == (1, vae.z_dim, 8, 12)  # canonical in, canonical 2x out


def test_the_pixels_are_scaled_by_the_upscalers_scale():
    """The encoder is handed the pixels at ``scale`` times the decoded size."""
    vae = FakeVAE(spatial_compression=16)
    _upscaler(vae, scale=2)(torch.zeros(1, vae.z_dim, 4, 6))

    # 4x6 latent -> 64x96 pixels -> 2x -> 128x192 pixels.
    assert vae.encoded_pixels.shape == (1, vae.pixel_channels, 128, 192)


def test_the_scale_is_the_one_the_adapter_asked_for():
    """Nothing in the architecture fixes 2x here, so the factor is honoured."""
    vae = FakeVAE(spatial_compression=16)
    z_up = _upscaler(vae, scale=4)(torch.zeros(1, vae.z_dim, 4, 6))

    assert vae.encoded_pixels.shape[-2:] == (256, 384)
    assert z_up.shape == (1, vae.z_dim, 16, 24)


def test_the_bicubic_overshoot_is_clamped_before_encoding():
    """Bicubic rings past ``[-1, 1]``; the encoder must not see that.

    A checkerboard is the worst case (it overshoots to ~1.5x). The pixels arriving
    at the VAE encoder are back inside the range ``decode_to_pixels`` promised —
    most visibly so on an alpha, where ringing would mean negative opacity.
    """
    vae = FakeVAE(pattern="checker")
    _upscaler(vae)(torch.zeros(1, vae.z_dim, 4, 4))

    pixels = vae.encoded_pixels
    assert pixels.max() <= 1.0 and pixels.min() >= -1.0


def test_every_pixel_channel_round_trips_including_the_alpha():
    """The resize is channel-agnostic: an RGBA VAE keeps its alpha through it.

    Dropping to RGB here would be the pixel-domain upscaler's compromise, not this
    one's — the VAE on the other end of the encode wants all its channels back.
    """
    vae = FakeVAE(pixel_channels=4)
    _upscaler(vae)(torch.zeros(1, vae.z_dim, 4, 4))

    assert vae.encoded_pixels.shape[1] == 4


def test_the_rgb_vae_path_is_unchanged():
    """A 3-channel VAE is not padded to four on the way back in."""
    vae = FakeVAE(pixel_channels=3)
    _upscaler(vae)(torch.zeros(1, vae.z_dim, 4, 4))

    assert vae.encoded_pixels.shape[1] == 3


def test_the_vaes_own_dtype_and_device_are_used_for_its_calls():
    """Both VAE calls get exactly what that VAE runs on, whatever came in."""
    vae = FakeVAE(dtype=torch.bfloat16)
    z_up = _upscaler(vae)(_latent(vae, dtype=torch.float32))

    assert vae.decode_input.dtype == torch.bfloat16
    assert vae.encoded_pixels.dtype == torch.bfloat16
    assert z_up.dtype == torch.bfloat16


def test_a_video_layout_decode_is_squeezed_to_the_image_layout():
    """The family's 5D decode output must not be interpolated as a volume."""
    vae = FakeVAE(video_layout=True)
    _upscaler(vae)(torch.zeros(1, vae.z_dim, 4, 4))

    assert vae.encoded_pixels.ndim == 4
    assert vae.encoded_pixels.shape[-2:] == (64, 64)  # 4x4 latent -> 32px -> 2x


# ------------------------------------------------------------------ the contract


def test_it_is_a_latent_upscaler():
    """The pipeline's only handle on the strategy is the interface."""
    assert isinstance(_upscaler(), LatentUpscaler)


def test_scale_is_required():
    """The adapter states the factor it advertises through ``UPSCALE_SCALE``."""
    with pytest.raises(TypeError):
        VAEPixelUpscaler(FakeVAE())
    with pytest.raises(ValueError, match="scale must be >= 1"):
        VAEPixelUpscaler(FakeVAE(), scale=0)


def test_a_vae_that_cannot_encode_is_rejected_at_construction():
    """The Flux VAE is decoder-only; fail where the adapter wired it up.

    Without this the mismatch surfaces on the first upscale request, far from the
    line that caused it.
    """
    decode_only = SimpleNamespace(decode_to_pixels=lambda z: z)

    with pytest.raises(ValueError, match="encode_pixels_to_latents"):
        VAEPixelUpscaler(decode_only, scale=2)
