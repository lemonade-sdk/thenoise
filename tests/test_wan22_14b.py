"""The two-expert Wan 2.2 A14B path, and the 2.1 codec it decodes with."""
from pathlib import Path

import pytest
import torch

from conftest import wan_adapter, write_wan_dit, write_wan_vae
from thenoise.dit.wan_video.model import Wan22Params, WanNativeDiT, detect_params
from thenoise.dit.wan_video.sampling import get_sigmas
from thenoise.models import resolve
from thenoise.models.config import ModelConfig, VideoSamplingParams
from thenoise.models.wan_video import VARIANTS, Wan22Model, variant_for
from thenoise.samplers import Step
from thenoise.runtime import ModelPaths, Runtime, Settings
from thenoise.utils.wan_weights import wan_vae21_key
from thenoise.vae.qwen_image import (
    AutoencoderKLQwenImage, QWEN_IMAGE_LATENTS_MEAN, QWEN_IMAGE_LATENTS_STD,
    _WAN21_FAMILY_ARCH,
)
from thenoise.vae.wan_video import (
    _comfy_vae_config, _RELEASES, _vae_release, _wan21_vae_config,
    AutoencoderKLWanVideo,
)
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


def test_a_codec_from_the_other_release_is_refused_from_the_headers(tmp_path):
    """The clip's latent is the only thing that crosses between the two checkpoints, and
    the normalization table is keyed by its width, so a mismatched `--vae` is answered
    before either module is built rather than at the first denoising step.
    """
    config = ModelConfig(
        dit_paths={"": write_wan_dit(tmp_path / "dit.safetensors", channels=48)},
        vae_path=str(write_wan_vae(tmp_path / "vae.safetensors", channels=16)),
        text_encoder_path="", device="cpu", offload_device="cpu")
    with pytest.raises(ValueError, match="different Wan releases"):
        Wan22Model(config=config)


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
                                    is_residual=False,
                                    temperal_downsample=[True, True, False],
                                    latents_mean=[0.] * 16, latents_std=[1.] * 16))
    with torch.device("meta"):
        vae = AutoencoderKLWanVideo(vae_cfg)
    assert vae.encoder.conv_in.weight.shape[1] == 3
    assert vae.decoder.up_blocks[0].upsampler.resample[1].weight.shape[:2] == (16, 32)
    assert wan_vae21_key("encoder.downsamples.2.time_conv.weight") == (
        "encoder.down_blocks.0.downsampler.time_conv.weight")
    assert wan_vae21_key("decoder.upsamples.3.resample.1.weight") == (
        "decoder.up_blocks.0.upsampler.resample.1.weight")
    assert wan_vae21_key("encoder.downsamples.0.residual.0.gamma") == (
        "encoder.down_blocks.0.resnets.0.norm1.gamma")


def test_the_two_experts_split_the_schedule_at_the_sigma_boundary():
    model = wan_adapter("t2v_14b")
    high, low = object(), object()
    model.dit, model.low_dit = high, low
    steps = model.schedule(VideoSamplingParams(height=64, width=64, steps=4, seed=1,
                                               guidance_scale=4.0, sampler="euler"))
    spans = model.segments(None, steps, guidance_scale=4.0)

    assert [s.stage for s in spans] == ["dit", "dit_low"]
    assert [s.module for s in spans] == [high, low]
    assert [len(s.steps) for s in spans] == [3, 1]
    assert sum(len(s.steps) for s in spans) == len(steps)
    # Upstream ships guidance 4.0 on the high half and 3.0 on the low.
    assert spans[1].guidance_scale == pytest.approx(4.0 * A14B.experts.low_cfg_ratio)
    sigmas = get_sigmas(4, shift=A14B.experts.flow_shift)
    assert sigmas[2] > A14B.experts.boundary > sigmas[3]


def test_a_single_expert_variant_answers_one_segment():
    model = wan_adapter()
    model.dit = object()
    spans = model.segments(None, [Step(t=torch.zeros(()), delta=torch.zeros(()))], 5.0)
    assert [s.stage for s in spans] == ["dit"]


def test_the_16ch_video_codec_is_the_qwen_image_codec_in_3d():
    """One architecture, two folds: the geometry is `qwen_image`'s rather than restated
    here, and the latent statistics are the same numbers -- which is why the 14B experts
    and Qwen-Image agree on what a canonical latent is.
    """
    video_cfg = _comfy_vae_config(_wan21_vae_config())
    assert (video_cfg.base_dim, video_cfg.z_dim, list(video_cfg.dim_mult),
            video_cfg.num_res_blocks) == (
        _WAN21_FAMILY_ARCH["base_dim"], _WAN21_FAMILY_ARCH["z_dim"],
        list(_WAN21_FAMILY_ARCH["dim_mult"]), _WAN21_FAMILY_ARCH["num_res_blocks"])
    assert video_cfg.latents_mean == QWEN_IMAGE_LATENTS_MEAN
    assert video_cfg.latents_std == QWEN_IMAGE_LATENTS_STD

    with torch.device("meta"):
        video = AutoencoderKLWanVideo(video_cfg)
    image = AutoencoderKLQwenImage(**_WAN21_FAMILY_ARCH)
    assert video.spatial_compression == image.spatial_compression == 8
    assert video.pixel_channels == image.pixel_channels == 3
    # The family signature: Wan 2.1 halves channels on the way up, 2.2 keeps them.
    upsample = video.decoder.up_blocks[0].upsampler.resample[1]
    assert upsample.weight.shape[0] == upsample.weight.shape[1] // 2


def test_the_16ch_latent_normalisation_is_not_identity():
    assert len(QWEN_IMAGE_LATENTS_MEAN) == len(QWEN_IMAGE_LATENTS_STD) == 16
    with torch.device("meta"):
        vae = AutoencoderKLWanVideo(_comfy_vae_config(_wan21_vae_config() | dict(
            base_dim=4, decoder_base_dim=4, num_res_blocks=1)))
    z = torch.randn(1, 16, 2, 3, 4)
    normalized = vae._normalize(z)
    assert not torch.allclose(z, normalized)
    torch.testing.assert_close(vae._denormalize(normalized), z)


def test_a_video_codec_of_another_latent_width_is_refused_by_name():
    assert sorted(_RELEASES) == [16, 48]
    assert _vae_release(16)[1] is not _vae_release(48)[1]
    with pytest.raises(ValueError, match=r"32-channel video VAE is no Wan release.*\[16, 48\]"):
        _vae_release(32)

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
        vae = AutoencoderKLWanVideo(cfg).half().eval()
        latents = torch.randn(1, 16, 2, 2, 2, dtype=torch.float32)
        first = vae.decode(latents)
        second = vae.decode(latents)
        assert first.dtype == torch.float16
        assert first.shape == second.shape == (1, 3, 5, 16, 16)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert all(item is None for item in vae._feat_map)


def test_the_two_expert_variant_has_no_i2v():
    assert A14B.i2v is False and VARIANTS["ti2v_5b"].i2v is True
    with pytest.raises(ValueError, match="I2V"):
        wan_adapter("t2v_14b").resolve_video(VideoSpec(prompt="x", image=b"image"))
