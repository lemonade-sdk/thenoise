"""Small CPU regressions for native LTX 2.3/2.5 numerical components.

These tests exercise TheNoise contracts; no Diffusers installation or
migration-time reference model is required.
"""
from __future__ import annotations

from types import SimpleNamespace
import pytest
import torch

from thenoise.dit.ltx_video.connectors import LTX2TextConnectors
from thenoise.dit.ltx_video.transformer import LTX2VideoTransformer3DModel
from thenoise.dit.ltx_video.checkpoint import make_video_vae
from thenoise.utils.rope import apply_rope_split_half, apply_ltx_rope_split_half
from thenoise.vae.ltx_video import LTXVideoVAE

from conftest import tiny_ltx_video_vae





def test_ltx_gemma_sdpa_fallback_is_scoped_and_overridable(monkeypatch):
    from thenoise.text_encoders.ltx_video import _use_math_gemma_attention

    monkeypatch.delenv("THENOISE_LTX_GEMMA_SDPA", raising=False)
    assert not _use_math_gemma_attention(torch.device("cpu"))

    # Fake ROCm attributes to test the architecture check on CPU-only CI.
    monkeypatch.setattr(torch.version, "hip", "test-rocm", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(gcnArchName="gfx1151:sramecc-"))
    assert _use_math_gemma_attention(torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(gcnArchName="gfx1100"))
    assert _use_math_gemma_attention(torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda device: SimpleNamespace(gcnArchName="gfx1201"))
    assert not _use_math_gemma_attention(torch.device("cuda:0"))

    monkeypatch.setenv("THENOISE_LTX_GEMMA_SDPA", "math")
    assert _use_math_gemma_attention(torch.device("cpu"))
    monkeypatch.setenv("THENOISE_LTX_GEMMA_SDPA", "default")
    assert not _use_math_gemma_attention(torch.device("cuda:0"))
    monkeypatch.setenv("THENOISE_LTX_GEMMA_SDPA", "invalid")
    with pytest.raises(ValueError, match="THENOISE_LTX_GEMMA_SDPA"):
        _use_math_gemma_attention(torch.device("cuda:0"))


@pytest.mark.parametrize("ltx25", [False, True])
def test_av_transformer_forward_and_checkpoint_variants(ltx25):
    torch.manual_seed(14)
    config = dict(
        in_channels=8, out_channels=8,
        num_attention_heads=4, attention_head_dim=16, cross_attention_dim=64,
        audio_in_channels=8, audio_out_channels=8,
        audio_num_attention_heads=4, audio_attention_head_dim=16,
        audio_cross_attention_dim=64, num_layers=1,
        ff_bias=not ltx25, audio_ff_bias=True,
        gated_attn=True, audio_gated_attn=True,
        cross_attn_mod=True, audio_cross_attn_mod=True,
        use_keyframes_abs_pos_embedding=ltx25,
    )
    model = LTX2VideoTransformer3DModel(**config).eval()
    args = dict(
        hidden_states=torch.randn(1, 4, 8),
        audio_hidden_states=torch.randn(1, 3, 8),
        encoder_hidden_states=torch.randn(1, 4, 64),
        audio_encoder_hidden_states=torch.randn(1, 4, 64),
        encoder_attention_mask=torch.tensor([[1, 0, 1, 1]], dtype=torch.float32),
        audio_encoder_attention_mask=torch.tensor([[1, 0, 1, 1]], dtype=torch.float32),
        timestep=torch.tensor([[0., 750., 750., 750.]]),
        audio_timestep=torch.tensor([750.]), sigma=torch.tensor([750.]),
        audio_sigma=torch.tensor([750.]), num_frames=1, height=2, width=2,
        audio_num_frames=3, use_cross_timestep=True,
    )
    with torch.inference_mode():
        video, audio = model(**args)
        video_isolated, audio_isolated = model(**args, isolate_modalities=True)
    assert video.shape == video_isolated.shape == (1, 4, 8)
    assert audio.shape == audio_isolated.shape == (1, 3, 8)
    assert all(t.isfinite().all() for t in (video, audio, video_isolated, audio_isolated))
    assert not torch.equal(video, video_isolated) or not torch.equal(audio, audio_isolated)

    # The LTX 2.5 keyframe and video-FFN bias difference is checkpoint ABI.
    state = model.state_dict()
    assert ("keyframes_abs_pos_embedding" in state) == ltx25
    assert ("transformer_blocks.0.ff.net.0.proj.bias" in state) != ltx25
    assert "transformer_blocks.0.audio_to_video_attn.to_k.weight" in state


def test_ltx_text_connectors_ignore_padded_features():
    torch.manual_seed(7)
    model = LTX2TextConnectors(
        caption_channels=8, text_proj_in_factor=3,
        video_connector_num_attention_heads=2,
        video_connector_attention_head_dim=16, video_connector_num_layers=2,
        video_connector_num_learnable_registers=4, video_gated_attn=True,
        audio_connector_num_attention_heads=2,
        audio_connector_attention_head_dim=8, audio_connector_num_layers=2,
        audio_connector_num_learnable_registers=4, audio_gated_attn=True,
        video_hidden_dim=32, audio_hidden_dim=16,
    ).eval()
    hidden = torch.randn(1, 8, 8, 3)
    mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]])
    changed = hidden.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 200
    with torch.inference_mode():
        video, audio, output_mask = model(hidden, mask)
        video_changed, audio_changed, _ = model(changed, mask)
    assert video.shape == (1, 8, 32)
    assert audio.shape == (1, 8, 16)
    assert output_mask.shape == mask.shape
    assert bool(output_mask.all())
    torch.testing.assert_close(video, video_changed)
    torch.testing.assert_close(audio, audio_changed)


def test_ltx_batched_rope_preserves_shared_convention_and_dtype():
    torch.manual_seed(17)
    x = torch.randn(2, 3, 4, 8)
    phases = torch.randn(2, 3, 4, 4)
    cos, sin = phases.cos(), phases.sin()
    expected = torch.cat((x[..., :4] * cos - x[..., 4:] * sin,
                          x[..., 4:] * cos + x[..., :4] * sin), dim=-1)
    torch.testing.assert_close(apply_ltx_rope_split_half(x, cos, sin), expected)
    assert apply_ltx_rope_split_half(x.bfloat16(), cos, sin).dtype == torch.bfloat16
    legacy_cos = torch.cat((cos[0, 0], cos[0, 0]), -1)[:, None, None, :]
    legacy_sin = torch.cat((sin[0, 0], sin[0, 0]), -1)[:, None, None, :]
    assert apply_rope_split_half(x[:1, :2], legacy_cos, legacy_sin).shape == (1, 2, 4, 8)


def test_native_video_vae_meta_layout_and_frame_grid():
    prod = make_video_vae()
    state = prod.state_dict()
    assert all(t.is_meta for t in state.values())
    assert state["encoder.conv_in.conv.weight"].shape == (128, 48, 3, 3, 3)
    assert state["decoder.up_blocks.0.upsamplers.0.conv.conv.weight"].shape == (4096, 1024, 3, 3, 3)

    small = tiny_ltx_video_vae()
    with torch.inference_mode():
        for frames in (1, 9):
            pixels = torch.randn(1, 3, frames, 16, 16)
            latent = small.encode(pixels)
            assert latent.shape == (1, 8, (frames - 1) // 4 + 1, 2, 2)
            decoded = small.decode(latent)
            assert decoded.shape == pixels.shape
            assert torch.isfinite(decoded).all()
