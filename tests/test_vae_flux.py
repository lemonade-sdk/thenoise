"""The Flux VAE (used by Z-Image, and the latent format of the Flux upscaler).

Decode-only by design (this engine never encodes pixels), so the tests cover the
8x decode geometry, the documented latent normalization and the checkpoint
loader's key handling. A reduced-channel instance keeps all of it sub-second;
the shipped constants are asserted on the class.
"""
from __future__ import annotations

import pytest
import torch

from conftest import write_safetensors
from thenoise.vae import AutoencoderKLFlux, load_flux_vae


def _tiny_vae() -> AutoencoderKLFlux:
    """The shipped architecture with 8-channel blocks (weights are random)."""
    return AutoencoderKLFlux(
        block_out_channels=(8, 8, 8, 8), norm_num_groups=4, layers_per_block=0
    )


@pytest.fixture(scope="module")
def tiny_vae():
    return _tiny_vae().eval().requires_grad_(False)


def test_documented_latent_normalization_constants():
    """The Flux latent is affine-normalized; the values must match the checkpoint."""
    assert AutoencoderKLFlux.z_dim == 16
    assert AutoencoderKLFlux.scaling_factor == pytest.approx(0.3611)
    assert AutoencoderKLFlux.shift_factor == pytest.approx(0.1159)


def test_decode_upsamples_the_latent_8x_and_clamps(tiny_vae):
    with torch.no_grad():
        pixels = tiny_vae.decode_to_pixels(torch.randn(1, 16, 4, 4))
    assert pixels.shape == (1, 3, 32, 32)  # 8x spatial compression
    assert pixels.dtype == torch.float32
    assert pixels.min() >= -1.0 and pixels.max() <= 1.0


def test_decode_accepts_odd_latent_sizes(tiny_vae):
    with torch.no_grad():
        pixels = tiny_vae.decode_to_pixels(torch.randn(1, 16, 3, 5))
    assert pixels.shape == (1, 3, 24, 40)


def test_decode_applies_the_affine_latent_transform(tiny_vae, monkeypatch):
    """``decode_to_pixels`` undoes ``(raw - shift) * scale`` before the decoder.

    This is the exact inverse of the Flux latent-format adaptor used by the
    Sesqui upscale path, so the two must agree.
    """
    seen = {}

    def spy_decode(z):
        seen["z"] = z.clone()
        return torch.zeros(1, 3, 8, 8)

    monkeypatch.setattr(tiny_vae, "decode", spy_decode)
    latents = torch.full((1, 16, 1, 1), 0.5)
    tiny_vae.decode_to_pixels(latents)

    expected = 0.5 / AutoencoderKLFlux.scaling_factor + AutoencoderKLFlux.shift_factor
    assert torch.allclose(seen["z"].float(), torch.full_like(seen["z"], expected, dtype=torch.float32))


def test_dtype_and_device_follow_the_weights(tiny_vae):
    assert tiny_vae.dtype == tiny_vae.conv_in.weight.dtype
    assert tiny_vae.device == tiny_vae.conv_in.weight.device


def _write_decoder_checkpoint(tmp_path, vae):
    """Write a checkpoint in the shipped ``ae.safetensors`` key layout."""
    from safetensors.torch import save_file

    tensors = {f"decoder.{k}": v for k, v in vae.state_dict().items()}
    # A real ae.safetensors also carries the encoder side, which must be ignored.
    tensors["encoder.conv_in.weight"] = torch.zeros(3)
    tensors["quant_conv.weight"] = torch.zeros(3)
    save_file(tensors, str(tmp_path / "ae.safetensors"))
    return str(tmp_path / "ae.safetensors")


def test_load_flux_vae_reads_only_the_decoder_side(tmp_path, monkeypatch):
    source = _tiny_vae()
    path = _write_decoder_checkpoint(tmp_path, source)

    monkeypatch.setattr("thenoise.vae.flux.AutoencoderKLFlux", _tiny_vae)
    vae = load_flux_vae(path, device="cpu", dtype=torch.bfloat16)

    assert isinstance(vae, AutoencoderKLFlux)
    assert vae.dtype == torch.bfloat16
    assert vae.training is False
    reference = source.state_dict()
    assert set(vae.state_dict()) == set(reference)
    for key, value in vae.state_dict().items():
        assert torch.allclose(value.float(), reference[key].float(), atol=1e-2), key


def test_load_flux_vae_rejects_a_file_without_decoder_keys(tmp_path, monkeypatch):
    path = write_safetensors(tmp_path / "wrong.safetensors", {"unet.weight": torch.zeros(1)})
    monkeypatch.setattr("thenoise.vae.flux.AutoencoderKLFlux", _tiny_vae)
    with pytest.raises(ValueError, match="No 'decoder\\.\\*' keys found"):
        load_flux_vae(path, device="cpu")
