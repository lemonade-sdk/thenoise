"""The two-expert Wan 2.2 A14B path, and the 2.1 codec it decodes with."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from conftest import wan_adapter, write_wan_dit
from thenoise.dit.wan_video.model import Wan22Params, WanNativeDiT, detect_params
from thenoise.dit.wan_video.sampling import get_sigmas
from thenoise.models import resolve
from thenoise.models.wan_video import VARIANTS, Wan22Model, variant_for
from thenoise.runtime import ModelPaths, Runtime, Settings
from thenoise.utils.wan_weights import wan_vae21_key
from thenoise.vae.wan_video import (
    _comfy_vae_config, _WAN21_LATENTS_MEAN, _WAN21_LATENTS_STD, _wan21_vae_config,
    WanNativeVideoVAE,
)
from thenoise.video.jobs import Cancellation
from thenoise.video.spec import VideoSpec

A14B = VARIANTS["t2v_14b"]


def test_one_adapter_claims_both_transformers(tmp_path):
    """Detection stops at "a Wan 2.2 DiT": the variant is a shape read afterwards."""
    for channels, name in ((48, "ti2v_5b"), (16, "t2v_14b")):
        path = write_wan_dit(tmp_path / f"dit{channels}.safetensors", channels=channels)
        assert resolve(path) is Wan22Model
        assert variant_for(detect_params(path)) is VARIANTS[name]


def test_the_model_name_is_one_machine_string():
    assert Wan22Model.name == "wan22"
    assert wan_adapter("ti2v_5b").CAPABILITIES["i2v"] is True
    assert wan_adapter("t2v_14b").CAPABILITIES["i2v"] is False


def test_a_two_expert_transformer_demands_its_second_file(tmp_path):
    high = write_wan_dit(tmp_path / "high.safetensors", channels=16)
    with pytest.raises(ValueError, match="--dit-low"):
        Wan22Model.validate_paths({"": high})
    with pytest.raises(FileNotFoundError, match="low-noise"):
        Wan22Model.validate_paths({"": high, "low": str(tmp_path / "gone.safetensors")})
    with pytest.raises(ValueError, match="different checkpoints"):
        Wan22Model.validate_paths({"": high, "low": high})
    Wan22Model.validate_paths(
        {"": high, "low": write_wan_dit(tmp_path / "low.safetensors", channels=16)})


def test_a_single_expert_transformer_refuses_a_second_file(tmp_path):
    with pytest.raises(ValueError, match="one expert"):
        Wan22Model.validate_paths({
            "": write_wan_dit(tmp_path / "a.safetensors"),
            "low": write_wan_dit(tmp_path / "b.safetensors"),
        })


def test_runtime_maps_the_cli_flags_into_roles(tmp_path, monkeypatch):
    """The runtime hands the adapter the paths by role and keeps no Wan knowledge."""
    import thenoise.models as catalog

    class StubVideo:
        name = "stub"
        VIDEO = True
        validated = []

        def __init__(self, *, config):
            self.config = config

        @classmethod
        def validate_paths(cls, dit_paths):
            cls.validated.append(dict(dit_paths))

    monkeypatch.setattr(catalog, "resolve", lambda path: StubVideo)
    rt = Runtime(Settings(device="cpu", offload_device="cpu"))
    high, low = str(tmp_path / "high.safetensors"), str(tmp_path / "low.safetensors")
    Path(high).write_bytes(b"high")
    Path(low).write_bytes(b"low")
    rt.load(ModelPaths(high, "vae", "te", dit_low_path=low))
    assert StubVideo.validated == [{"": high, "low": low}]
    assert rt.model.config.dit_paths == {"": high, "low": low}
    assert rt.model.config.dit_path == high


def test_14b_topology_meta_and_vae21_key_mapping():
    params = Wan22Params(in_channels=16, out_channels=16, dim=5120, num_layers=40,
                         ffn_dim=13824, freq_dim=256, text_dim=4096, patch_size=(1, 2, 2))
    with torch.device("meta"):
        dit = WanNativeDiT(params)
    assert params.heads == 40
    assert dit.patch_embedding.weight.shape == (5120, 16, 1, 2, 2)
    assert dit.proj_out.weight.shape == (64, 5120)
    assert len(dit.blocks) == 40
    assert dit.blocks[0].ffn_in.weight.shape == (13824, 5120)

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
            assert timestep.shape == (1,)  # scalar A14B timestep, not 5B token mask
            self.sigmas.append(timestep.item() / 1000)
            return (torch.zeros_like(x),)

    model = wan_adapter("t2v_14b")
    model.transformer = Expert()
    model.low_transformer = Expert()
    model.memory.register("dit", model.transformer)
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
    sigmas = get_sigmas(4, shift=A14B.experts.flow_shift)
    assert sigmas[2] > A14B.experts.boundary > sigmas[3]


def test_wan21_latent_normalization_is_not_identity():
    assert len(_WAN21_LATENTS_MEAN) == len(_WAN21_LATENTS_STD) == 16
    assert _WAN21_LATENTS_MEAN[0] == -0.7571
    assert _WAN21_LATENTS_STD[0] == 2.8184
    video = wan_adapter("t2v_14b")
    video.vae = SimpleNamespace(config=SimpleNamespace(
        z_dim=16, latents_mean=_WAN21_LATENTS_MEAN, latents_std=_WAN21_LATENTS_STD))
    z = torch.randn(1, 16, 2, 3, 4)
    normalized = video._normalize(z)
    assert not torch.allclose(z, normalized)
    torch.testing.assert_close(video._denormalize(normalized), z)


def test_each_video_codec_declares_its_own_compute_dtype():
    """The 48ch 2.2 codec runs at the 2.2 family's BF16 (see
    ``vae.wan22.COMPUTE_DTYPE``), the 16ch 2.1 codec at the FP16 it ships in; the
    loader moves the module to whatever its config declares.
    """
    assert _comfy_vae_config().compute_dtype is torch.bfloat16
    assert _comfy_vae_config(_wan21_vae_config()).compute_dtype is torch.float16


def test_wan21_fp16_vae_decodes_float32_latents_and_clears_cache():
    cfg = _comfy_vae_config(_wan21_vae_config() | dict(
        base_dim=4, decoder_base_dim=4, num_res_blocks=1,
    ))
    with torch.inference_mode():
        vae = WanNativeVideoVAE(cfg).half().eval()
        latents = torch.randn(1, 16, 2, 2, 2, dtype=torch.float32)
        first = vae.decode(latents).sample
        second = vae.decode(latents).sample
        assert first.dtype == torch.float16
        assert first.shape == second.shape == (1, 3, 5, 16, 16)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert all(item is None for item in vae._feat_map)


def test_the_two_expert_variant_has_no_i2v():
    assert A14B.i2v is False and VARIANTS["ti2v_5b"].i2v is True
    with pytest.raises(ValueError, match="I2V"):
        wan_adapter("t2v_14b").resolve_video(VideoSpec(prompt="x", image=b"image"))
