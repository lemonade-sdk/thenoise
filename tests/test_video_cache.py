"""The video pipeline's encode-stage cache: what a second request still pays for."""
from __future__ import annotations

import io

import pytest
from PIL import Image

from conftest import StubVideo
from thenoise.jobs import Cancellation
from thenoise.models.config import ModelConfig
from thenoise.video.controller import VideoPipelineController
from thenoise.video.spec import VideoSpec

STEPS = StubVideo.DEFAULT_PREFS["steps"]


@pytest.fixture
def video(tmp_path) -> StubVideo:
    return StubVideo(config=ModelConfig(
        dit_paths={"": str(tmp_path / "dit")}, vae_path=str(tmp_path / "vae"),
        text_encoder_path=str(tmp_path / "te"), device="cpu", offload_device="cpu"))


@pytest.fixture
def controller(video, monkeypatch) -> VideoPipelineController:
    monkeypatch.setattr("thenoise.video.controller.encode_video",
                        lambda frames, spec, token, dest, progress: dest.write_bytes(b"v"))
    return VideoPipelineController(video)


@pytest.fixture
def render(video, controller, tmp_path):
    def go(**kwargs):
        controller.render(video.resolve_video(VideoSpec(**kwargs)), Cancellation(),
                          lambda *_: None, tmp_path / "clip.webm")
    return go


def _picture(color=(200, 30, 30), size=(64, 64), fmt="PNG") -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format=fmt)
    return out.getvalue()


def test_a_repeated_prompt_is_encoded_once_but_denoised_twice(render, video):
    render(prompt="a fox")
    render(prompt="a fox")
    assert video.calls["encode_prompt"] == 1
    assert video.calls["denoise_step"] == 2 * STEPS


def test_a_different_prompt_is_encoded_again(render, video):
    render(prompt="a fox")
    render(prompt="a hound")
    assert video.calls["encode_prompt"] == 2


def test_the_negative_prompt_makes_guidance_part_of_the_key(render, video):
    render(prompt="a fox", guidance_scale=1.0)
    render(prompt="a fox", guidance_scale=5.0)
    assert video.calls["encode_prompt"] == 2


def test_a_second_run_of_one_image_and_prompt_encodes_neither_stage(render, video):
    image = _picture()
    render(prompt="a fox", image=image)
    render(prompt="a fox", image=image)
    assert (video.calls["encode_reference"], video.calls["encode_prompt"]) == (1, 1)


def test_iterating_on_the_prompt_of_one_image_costs_the_text_encoder_only(render, video):
    image = _picture()
    render(prompt="a fox", image=image)
    render(prompt="a fox wearing a hat", image=image)
    assert (video.calls["encode_reference"], video.calls["encode_prompt"]) == (1, 2)


def test_a_new_reference_costs_the_prompt_as_well(render, video):
    """Conservative on purpose: the frames are handed to the encoder, so a conditioning
    encoded over another image goes with it.
    """
    render(prompt="a fox", image=_picture())
    render(prompt="a fox", image=_picture(color=(30, 30, 200)))
    assert (video.calls["encode_reference"], video.calls["encode_prompt"]) == (2, 2)


def test_a_text_to_video_run_does_not_inherit_a_conditioning_that_saw_no_image(
    render, video
):
    render(prompt="a fox", image=_picture())
    render(prompt="a fox")
    assert video.calls["encode_prompt"] == 2


def test_the_same_picture_in_another_container_is_the_same_reference(render, video):
    render(prompt="a fox", image=_picture())
    render(prompt="a fox", image=_picture(fmt="BMP"))
    assert video.calls["encode_reference"] == 1


def test_a_longer_clip_reencodes_the_reference(render, video):
    image = _picture()
    render(prompt="a fox", image=image, frames=5)
    render(prompt="a fox", image=image, frames=9)
    assert video.calls["encode_reference"] == 2


def test_a_larger_canvas_reencodes_the_reference(render, video):
    image = _picture()
    render(prompt="a fox", image=image, width=64)
    render(prompt="a fox", image=image, width=128)
    assert video.calls["encode_reference"] == 2


def test_the_cache_lives_and_dies_with_the_controller(video, tmp_path, monkeypatch):
    monkeypatch.setattr("thenoise.video.controller.encode_video",
                        lambda frames, spec, token, dest, progress: dest.write_bytes(b"v"))
    spec = video.resolve_video(VideoSpec(prompt="a fox"))
    for _ in range(2):
        VideoPipelineController(video).render(
            spec, Cancellation(), lambda *_: None, tmp_path / "clip.webm")
    assert video.calls["encode_prompt"] == 2
