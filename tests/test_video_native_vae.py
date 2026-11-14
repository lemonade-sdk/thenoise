"""Native Wan 2.2 causal VAE CPU regressions."""
from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from thenoise.vae.wan_video import (
    _AttentionBlock, _CausalConv3D, _comfy_vae_config, _patchify, _unpatchify,
    WanNativeVideoVAE, load_wan_native_video_vae,
)
from test_video_comfy_layout import _original_vae_key


TINY = dict(base_dim=4, decoder_base_dim=4, num_res_blocks=1)


def _vae():
    with torch.random.fork_rng():
        torch.manual_seed(9)
        return WanNativeVideoVAE(_comfy_vae_config(TINY)).eval()


def _write(root, *, missing=False, wrong_shape=False):
    path = root / "split_files" / "vae" / "wan2.2_vae.safetensors"
    path.parent.mkdir(parents=True)
    source = _vae()
    raw = {_original_vae_key(key): tensor.detach().contiguous().half()
           for key, tensor in source.state_dict().items()}
    if missing:
        raw.pop(next(iter(raw)))
    if wrong_shape:
        raw[next(iter(raw))] = torch.zeros(2, 2, dtype=torch.float16)
    save_file(raw, str(path))
    return path, source



@pytest.mark.parametrize("channels", [16, 320])
def test_spatial_attention_uses_shared_wan22_kernel(monkeypatch, channels):
    from thenoise.vae.wan22 import Wan22AttentionBlock

    block = _AttentionBlock(channels).eval()
    assert isinstance(block, Wan22AttentionBlock)
    assert set(block.state_dict()) == {"norm.gamma", "to_qkv.weight", "to_qkv.bias",
                                       "proj.weight", "proj.bias"}
    x = torch.randn(1, channels, 2, 4, 4)
    with torch.no_grad():
        frames = x.permute(0, 2, 1, 3, 4).reshape(2, channels, 4, 4)
        normalized = block.norm(frames)
        q, k, v = block.to_qkv(normalized).chunk(3, dim=1)
        q, k, v = (value.reshape(2, channels, 16).transpose(1, 2) for value in (q, k, v))
        reference = torch.nn.functional.scaled_dot_product_attention(
            q[:, None], k[:, None], v[:, None]
        ).squeeze(1).transpose(1, 2).reshape(2, channels, 4, 4)
        reference = (block.proj(reference) + frames).reshape(1, 2, channels, 4, 4).permute(0, 2, 1, 3, 4)

        def forbidden(*args, **kwargs):
            raise AssertionError("Video VAE must not use scaled_dot_product_attention")

        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", forbidden)
        result = block(x)
    torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)

def test_patchify_exact():
    x = torch.arange(1 * 3 * 5 * 16 * 32).float().reshape(1, 3, 5, 16, 32)
    assert torch.equal(_unpatchify(_patchify(x, 2), 2), x)


def test_causal_conv_cache_equivalent_to_whole_video():
    conv = _CausalConv3D(3, 4, 3, padding=1).eval()
    x = torch.randn(1, 3, 5, 4, 4)
    full = conv(x)
    pieces = [conv(x[:, :, :1]), conv(x[:, :, 1:], cache_x=x[:, :, :1])]
    torch.testing.assert_close(torch.cat(pieces, dim=2), full, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("frames", [1, 5, 9])
def test_causal_video_shape_and_cache_isolation(frames):
    vae = _vae()
    video = torch.randn(1, 3, frames, 16, 16)
    with torch.no_grad():
        z = vae.encode(video).latent_dist.mode()
        assert z.shape == (1, 48, (frames - 1) // 4 + 1, 1, 1)
        a = vae.decode(z).sample
        b = vae.decode(z).sample
    assert a.shape == b.shape == video.shape
    assert torch.equal(a, b)
    assert all(entry is None for entry in vae._feat_map)


def test_decode_frames_are_the_whole_clip_in_order():
    """Streaming the decode out one latent frame at a time must be the same
    computation the one-call decode does, caches and all.
    """
    vae = _vae()
    z = torch.randn(1, 48, 3, 1, 1)
    with torch.no_grad():
        streamed = torch.cat(list(vae.decode_frames(z)), dim=2)
        whole = vae.decode(z).sample
    torch.testing.assert_close(streamed, whole, rtol=0, atol=0)
    assert all(entry is None for entry in vae._feat_map)


def test_a_precision_nobody_measured_this_codec_at_is_refused():
    """FP16 is where this family overflows; FP32 only costs, so it is allowed."""
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="runs in torch.bfloat16"):
            _vae().half().decode(torch.randn(1, 48, 1, 1, 1))


def test_comfy_fp16_checkpoint_loads_at_the_compute_dtype(tmp_path):
    path, source = _write(tmp_path)
    vae = load_wan_native_video_vae(str(path), "cpu", config=_comfy_vae_config(TINY))
    assert vae.dtype is vae.compute_dtype is torch.bfloat16
    assert not any(p.is_meta for p in vae.parameters())
    assert torch.equal(vae.encoder.conv_in.weight,
                       source.encoder.conv_in.weight.half().bfloat16())
    with torch.no_grad():
        result = vae.decode(torch.randn(1, 48, 2, 1, 1)).sample
    assert result.shape == (1, 3, 5, 16, 16)


@pytest.mark.parametrize("damage,raises", [("missing", "Missing key"), ("shape", "size mismatch")])
def test_comfy_checkpoint_rejects_invalid_weights(tmp_path, damage, raises):
    path, _ = _write(tmp_path, missing=damage == "missing", wrong_shape=damage == "shape")
    with pytest.raises(RuntimeError, match=raises):
        load_wan_native_video_vae(str(path), "cpu", config=_comfy_vae_config(TINY))


def test_official_comfy_vae_topology_meta():
    with torch.device("meta"):
        vae = WanNativeVideoVAE(_comfy_vae_config())
    assert vae.config.z_dim == 48
    assert vae.config.base_dim == 160
    assert all(p.is_meta for p in vae.parameters())
