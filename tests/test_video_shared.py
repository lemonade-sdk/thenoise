"""Video architecture seams: shared weights, model paths and adapter lifecycle."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from thenoise.models.video_base import VideoModel
from thenoise.models.wan_video import WanVideo
from thenoise.vae.wan22 import WAN22_LATENTS_MEAN, WAN22_LATENTS_STD
from thenoise.vae.wan_video import _comfy_vae_config
from thenoise.video.spec import VideoSpec


def test_video_model_contract_and_shared_latent_statistics():
    assert issubclass(WanVideo, VideoModel)
    assert WanVideo.resolve_video.__func__ is not VideoModel.resolve_video.__func__
    config = _comfy_vae_config()
    assert config.latents_mean == WAN22_LATENTS_MEAN
    assert config.latents_std == WAN22_LATENTS_STD


def test_video_base_is_usable_without_wan(tmp_path, monkeypatch):
    """A future adapter reuses resolution, arbitrary components and teardown."""
    class AnotherVideo(VideoModel):
        name = "other-video"
        CAPABILITIES = {**VideoModel.CAPABILITIES, "t2v": True}
        DEFAULT_PREFS = {
            "negative_prompt": "", "width": 640, "height": 384,
            "frames": 17, "fps": 25, "steps": 8, "guidance_scale": 1.0,
            "output_format": "mp4", "sampler": "euler",
        }

        @staticmethod
        def detect(_checkpoint):
            return False

        def load_weights(self):
            raise AssertionError("not a server startup test")

        def generate(self, spec, token, progress, output):
            raise AssertionError("not an inference test")

        def _clear_weights(self):
            self.extra_component = None

    config = SimpleNamespace(
        dit_path=str(tmp_path / "dit"), vae_path=str(tmp_path / "vae"),
        text_encoder_path=str(tmp_path / "text"), device="cpu", offload_device="cpu",
    )
    model = AnotherVideo(config=config)
    model.extra_component = object()
    resolved = model.resolve_video(VideoSpec(prompt="another model", width=800))
    assert (resolved.width, resolved.height, resolved.steps, resolved.frame_count) == (
        800, 384, 8, 17,
    )

    model.memory.register("audio_vae", torch.nn.Linear(1, 1))
    assert model.memory.weight_bytes("audio_vae") > 0
    old_manager = model.memory
    released = []
    monkeypatch.setattr("thenoise.models.video_base.release_device_memory", released.append)
    model.unload()
    assert model.extra_component is None
    assert model.memory is not old_manager
    assert model.memory.weight_bytes("audio_vae") == 0
    assert released == [model.device]


