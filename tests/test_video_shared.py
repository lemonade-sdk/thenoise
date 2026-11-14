"""Video architecture seams: shared weights, model paths and adapter lifecycle."""
from __future__ import annotations

import pytest
import torch

from conftest import StubVideo, wan_adapter
from thenoise.models.config import ModelConfig
from thenoise.models.video_base import Segment, VideoModel
from thenoise.models.wan_video import VARIANTS, Wan22Model
from thenoise.vae.wan22 import WAN22_LATENTS_MEAN, WAN22_LATENTS_STD
from thenoise.vae.wan_video import _comfy_vae_config
from thenoise.video.controller import VideoPipelineController
from thenoise.video.jobs import Cancellation
from thenoise.video.spec import VideoSpec


def test_video_model_contract_and_shared_latent_statistics():
    assert issubclass(Wan22Model, VideoModel)
    assert Wan22Model.resolve_video is not VideoModel.resolve_video
    config = _comfy_vae_config()
    assert config.latents_mean == WAN22_LATENTS_MEAN
    assert config.latents_std == WAN22_LATENTS_STD


def test_video_base_is_usable_without_wan(tmp_path):
    """A future adapter reuses resolution, arbitrary components and teardown."""
    class AnotherVideo(StubVideo):
        name = "other-video"
        CAPABILITIES = {**VideoModel.CAPABILITIES, "t2v": True}
        DEFAULT_PREFS = {**VideoModel.DEFAULT_PREFS,
                         "width": 640, "height": 384, "frames": 17, "fps": 25,
                         "steps": 8, "guidance_scale": 1.0,
                         "output_format": "mp4", "sampler": "euler"}

        def load_weights(self):
            raise AssertionError("not a server startup test")

        def _clear_weights(self):
            self.extra_component = None

    # No checkpoint exists here, and none is read: the markers a header could carry
    # are absent, which is the same answer an unreadable path gives.
    model = AnotherVideo(config=ModelConfig(
        dit_paths={"": str(tmp_path / "dit")}, vae_path=str(tmp_path / "vae"),
        text_encoder_path=str(tmp_path / "text"), device="cpu", offload_device="cpu"))
    model.extra_component = object()
    resolved = model.resolve_video(VideoSpec(prompt="another model", width=800))
    assert (resolved.width, resolved.height, resolved.steps, resolved.frame_count) == (
        800, 384, 8, 17,
    )

    model.memory.register("audio_vae", torch.nn.Linear(1, 1))
    assert model.memory.weight_bytes("audio_vae") > 0
    old_manager = model.memory
    model.unload()
    # What `unload` owns is the instance's own state; giving the device back is the
    # runtime's, after it dropped the instance (see test_runtime).
    assert model.extra_component is None
    assert model.memory is not old_manager
    assert model.memory.weight_bytes("audio_vae") == 0




def test_every_released_variant_answers_every_request_preference():
    """One wholly unset request per variant touches every fallback its table owns, so a
    key a release forgets fails here rather than at the first request served.
    """
    for variant in VARIANTS.values():
        model = wan_adapter(variant.name)
        resolved = model.resolve_video(VideoSpec(prompt="x"))
        for key, value in variant.prefs.items():
            if key != "frames":  # rounded onto the model's temporal grid
                assert getattr(resolved, key) == value, f"{variant.name}: {key}"
        # What the class answers for, no release restates.
        assert resolved.output_format == VideoModel.DEFAULT_PREFS["output_format"]


def test_a_video_adapter_declines_the_image_only_features():
    """One capability vocabulary, so `/health` needs no per-medium table: a clip says it
    has no latent upscaler rather than inheriting the image base's `True`.
    """
    model = wan_adapter()
    assert model.capability("video") and model.capability("t2v")
    assert not model.capability("upscale") and not model.capability("edit")
    with pytest.raises(NotImplementedError, match="no latent upscaler"):
        model.get_upscaler()


def _stub_config(tmp_path):
    return ModelConfig(dit_paths={"": str(tmp_path / "dit")}, vae_path="vae",
                       text_encoder_path="te", device="cpu", offload_device="cpu")


def test_a_model_with_two_segments_gets_one_progress_clock(tmp_path, monkeypatch):
    """The controller drives spans, not models: the job sees one sampling clock and one
    stage per span, whatever the model split its schedule into.
    """
    class TwoExpert(StubVideo):
        def segments(self, params, schedule, guidance_scale):
            half = len(schedule) // 2
            return [Segment("dit", self.dit, schedule[:half], guidance_scale),
                    Segment("dit_low", self.dit, schedule[half:], guidance_scale / 2)]

    model = TwoExpert(config=_stub_config(tmp_path))
    staged = []
    real_stage = model.memory.stage
    monkeypatch.setattr(model.memory, "stage",
                        lambda *names, **kw: staged.append(names[0]) or real_stage(*names, **kw))
    monkeypatch.setattr("thenoise.video.controller.encode_video",
                        lambda frames, spec, token, dest, progress: dest.write_bytes(b"v"))

    events = []
    spec = model.resolve_video(VideoSpec(prompt="x", steps=4))
    VideoPipelineController(model).render(spec, Cancellation(),
                                          lambda *e: events.append(e), tmp_path / "clip.webm")

    assert [e for e in events if e[0] == "sampling"] == [("sampling", i, 4) for i in range(1, 5)]
    assert staged == ["text_encoder", "dit", "dit_low", "vae"]
