"""CPU checks for TheNoise's standalone causal Wan 2.2 video VAE."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from thenoise.video.native_vae import (
    WanNativeVideoVAE, _CausalConv3D, _patchify, _unpatchify,
    load_wan_native_video_vae,
)


def config():
    """Small actual Wan TI2V topology (48 latent channels, 4x temporal)."""
    return dict(
        _class_name="AutoencoderKLWan", base_dim=4, decoder_base_dim=4,
        z_dim=48, dim_mult=[1, 2, 4, 4], num_res_blocks=1,
        temperal_downsample=[False, True, True], dropout=0.0, is_residual=True,
        in_channels=12, out_channels=12, patch_size=2,
        scale_factor_temporal=4, scale_factor_spatial=16,
        attn_scales=[], latents_mean=[0.0] * 48, latents_std=[1.0] * 48,
    )


def model():
    torch.manual_seed(9)
    return WanNativeVideoVAE(SimpleNamespace(**config())).eval()


def miniature_checkpoint(tmp_path, *, damage=None):
    directory = tmp_path / "vae"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config()))
    sd = {k: v.detach().contiguous() for k, v in model().state_dict().items()}
    if damage == "missing":
        sd.pop(next(iter(sd)))
    if damage == "shape":
        k = next(iter(sd))
        sd[k] = sd[k].flatten()[:1].clone()
    if damage == "bf16":
        k = next(iter(sd))
        sd[k] = sd[k].to(torch.bfloat16)
    save_file(sd, directory / "diffusion_pytorch_model.safetensors")
    return tmp_path


def test_patchify_unpatchify_exact_spatial_permutation():
    x = torch.arange(1 * 3 * 5 * 16 * 32, dtype=torch.float32).view(1, 3, 5, 16, 32)
    assert torch.equal(_unpatchify(_patchify(x, 2), 2), x)
    with pytest.raises(ValueError, match="divisible"):
        _patchify(x[..., :15], 2)


def test_causal_conv_cache_is_equivalent_to_full_temporal_conv():
    torch.manual_seed(100)
    conv = _CausalConv3D(3, 4, 3, padding=1).eval()
    x = torch.randn(1, 3, 5, 4, 4)
    with torch.no_grad():
        full = conv(x)
        first = conv(x[:, :, :1])
        later = conv(x[:, :, 1:], cache_x=x[:, :, :1])
    assert torch.allclose(torch.cat((first, later), dim=2), full, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("frames", [1, 5, 9])
def test_causal_video_encode_decode_and_cache_isolation(frames):
    vae = model()
    image = torch.randn(1, 3, frames, 16, 16)
    with torch.no_grad():
        posterior = vae.encode(image).latent_dist
        expected_t = (frames - 1) // 4 + 1
        assert posterior.mode().shape == (1, 48, expected_t, 1, 1)
        decoded1 = vae.decode(posterior.mode()).sample
        decoded2 = vae.decode(posterior.mode(), return_dict=False)[0]
        assert decoded1.shape == image.shape
        assert torch.allclose(decoded1, decoded2, atol=0.0, rtol=0.0)
        assert decoded1.abs().max() <= 1.0
        assert all(v is None for v in vae._feat_map)
        assert all(v is None for v in vae._enc_feat_map)
    with pytest.raises(ValueError, match="4\\*n\\+1"):
        vae.encode(image.repeat(1, 1, 2, 1, 1))


def test_checkpoint_loader_assigns_real_fp32_weights_without_diffusers(tmp_path):
    miniature_checkpoint(tmp_path)
    vae = load_wan_native_video_vae(tmp_path, _test_size=True)
    assert vae.dtype == torch.float32
    assert not any(p.is_meta for p in vae.parameters())
    assert torch.equal(vae.encoder.conv_in.weight, model().encoder.conv_in.weight)
    with torch.no_grad():
        z = torch.randn(1, 48, 2, 1, 1)
        expected = model().decode(z).sample
        actual = vae.decode(z).sample
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("damage", ["missing", "shape", "bf16"])
def test_checkpoint_loader_rejects_invalid_tensors(tmp_path, damage):
    miniature_checkpoint(tmp_path, damage=damage)
    with pytest.raises(ValueError, match="not FP32" if damage == "bf16" else "mismatch"):
        load_wan_native_video_vae(tmp_path, _test_size=True)


def test_checkpoint_loader_cancellation_between_tensors(tmp_path):
    miniature_checkpoint(tmp_path)
    calls = [0]
    def cancel():
        calls[0] += 1
        if calls[0] == 3:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        load_wan_native_video_vae(tmp_path, _test_size=True, check_cancel=cancel)
    assert calls[0] == 3


def test_production_config_rejects_incorrect_topology_before_allocation(tmp_path):
    miniature_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="base_dim"):
        load_wan_native_video_vae(tmp_path)
    cfg = config()
    cfg.update(base_dim=160, decoder_base_dim=256, num_res_blocks=2)
    (tmp_path / "vae" / "config.json").write_text(json.dumps(cfg))
    # Production topology can be checked without allocating weights.
    with torch.device("meta"):
        real = WanNativeVideoVAE(SimpleNamespace(**cfg))
    assert all(param.is_meta for param in real.parameters())
    assert real.config.z_dim == 48
    assert any("time_conv.weight" in k for k in real.state_dict())



def test_encode_and_decode_trigger_accelerate_offload_hook():
    vae = model()
    class Hook:
        calls = 0
        def pre_forward(self, module):
            assert module is vae
            self.calls += 1

    hook = Hook()
    vae._hf_hook = hook
    with torch.no_grad():
        vae.encode(torch.zeros(1, 3, 1, 16, 16))
        vae.decode(torch.zeros(1, 48, 1, 1, 1))
    assert hook.calls == 2


def test_bf16_video_vae_must_not_be_run_even_if_weights_are_upcast_later():
    vae = model().to(dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="remain in FP32"):
        vae.decode(torch.zeros(1, 48, 1, 1, 1))


def test_gaussian_posterior_sample_accepts_cpu_generator():
    vae = model()
    posterior = vae.encode(torch.zeros(1, 3, 1, 16, 16)).latent_dist
    x = posterior.sample(generator=torch.Generator(device="cpu").manual_seed(7))
    y = posterior.sample(generator=torch.Generator(device="cpu").manual_seed(7))
    assert torch.equal(x, y)
