"""Small no-download tests for the Wan 2.2 A14B expert path."""
from types import SimpleNamespace

import pytest
import torch

from thenoise.dit.wan_video.checkpoint import _T2V_14B_CONFIG, _WanWeightModel
from thenoise.dit.wan_video.sampling import get_sigmas
from thenoise.models.wan_video import Wan14BVideo, WanVideo
from thenoise.utils.wan_weights import wan_vae21_key
from thenoise.vae.wan_video import (
    _comfy_vae_config, _WAN21_LATENTS_MEAN, _WAN21_LATENTS_STD,
    WanNativeVideoVAE,
)
from thenoise.video.jobs import Cancellation
from thenoise.video.spec import VideoSpec


def test_14b_catalog_detection_does_not_regress_5b():
    class Shape:
        def get_shape(self):
            return (5120, 16, 1, 2, 2)

    class Checkpoint:
        def keys(self):
            return ("patch_embedding.weight", "blocks.0.self_attn.q.weight",
                    "text_embedding.0.weight")

        def get_slice(self, name):
            assert name == "patch_embedding.weight"
            return Shape()

    f = Checkpoint()
    assert Wan14BVideo.detect(f)
    assert not WanVideo.detect(f)


def test_14b_topology_meta_and_vae21_key_mapping():
    with torch.device("meta"):
        dit = _WanWeightModel(SimpleNamespace(**_T2V_14B_CONFIG))
    assert dit.patch_embedding.weight.shape == (5120, 16, 1, 2, 2)
    assert dit.proj_out.weight.shape == (64, 5120)
    assert len(dit.blocks) == 40
    assert dit.blocks[0].ffn.net[0].proj.weight.shape == (13824, 5120)

    vae_cfg = _comfy_vae_config(dict(base_dim=8, decoder_base_dim=8, z_dim=16,
                                    in_channels=3, out_channels=3, patch_size=1,
                                    scale_factor_spatial=8, is_residual=False,
                                    temperal_downsample=[True, True, False],
                                    latents_mean=[0.] * 16, latents_std=[1.] * 16))
    with torch.device("meta"):
        vae = WanNativeVideoVAE(vae_cfg)
    assert vae.encoder.conv_in.weight.shape[1] == 3
    assert vae.decoder.up_blocks[0].upsampler.resample[1].weight.shape[:2] == (16, 32)
    assert wan_vae21_key("encoder.downsamples.2.time_conv.weight") == (
        "encoder.down_blocks.0.downsampler.time_conv.weight")
    assert wan_vae21_key("decoder.upsamples.3.resample.1.weight") == (
        "decoder.up_blocks.0.upsampler.resample.1.weight")
    assert wan_vae21_key("encoder.downsamples.0.residual.0.gamma") == (
        "encoder.down_blocks.0.resnets.0.norm1.gamma")


def test_14b_uses_separate_experts_and_single_progress_clock():
    class Expert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.sigmas = []

        @property
        def dtype(self):
            return self.weight.dtype

        def forward(self, x, timestep, encoder_hidden_states, return_dict):
            assert timestep.shape == (1,)  # scalar WAN14 timestep, not 5B token mask
            self.sigmas.append(timestep.item() / 1000)
            return (torch.zeros_like(x),)

    model = Wan14BVideo(config=SimpleNamespace(
        dit_path="unused", vae_path="unused", text_encoder_path="unused",
        device="cpu", offload_device="cpu"))
    model.transformer = Expert()
    model.low_transformer = Expert()
    model.memory.register("dit_high", model.transformer)
    model.memory.register("dit_low", model.low_transformer)
    spec = model.resolve_video(VideoSpec(prompt="fox", width=64, height=64,
                                         frames=5, steps=4, sampler="euler",
                                         guidance_scale=1.0, seed=123))
    progress = []
    out = model._run_denoiser(torch.zeros(1, 16, 2, 8, 8), spec,
                              torch.zeros(1, 2, 16), None, None,
                              Cancellation(), lambda *e: progress.append(e), 64 * 64)
    assert out.shape == (1, 16, 2, 8, 8)
    assert len(model.transformer.sigmas) == 3
    assert len(model.low_transformer.sigmas) == 1
    assert progress == [("sampling", i, 4) for i in range(1, 5)]
    sigmas = get_sigmas(4, shift=model.FLOW_SHIFT)
    assert sigmas[2] > model.BOUNDARY > sigmas[3]


def test_wan21_latent_normalization_is_not_identity():
    assert len(_WAN21_LATENTS_MEAN) == len(_WAN21_LATENTS_STD) == 16
    assert _WAN21_LATENTS_MEAN[0] == -0.7571
    assert _WAN21_LATENTS_STD[0] == 2.8184
    video = Wan14BVideo(config=SimpleNamespace(
        dit_path="unused", vae_path="unused", text_encoder_path="unused",
        device="cpu", offload_device="cpu"))
    video.vae = SimpleNamespace(config=SimpleNamespace(
        z_dim=16, latents_mean=_WAN21_LATENTS_MEAN,
        latents_std=_WAN21_LATENTS_STD))
    z = torch.randn(1, 16, 2, 3, 4)
    normalized = video._normalize(z)
    assert not torch.allclose(z, normalized)
    torch.testing.assert_close(video._denormalize(normalized), z)


def test_14b_rejects_image_input():
    with pytest.raises(ValueError, match="I2V"):
        Wan14BVideo.resolve_video(VideoSpec(prompt="x", image=b"image"))
