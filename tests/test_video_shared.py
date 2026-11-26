"""Video architecture seams: shared weights, model paths and adapter lifecycle."""
from __future__ import annotations

import pytest
import torch

from conftest import StubVideo, header_of, wan_adapter
from test_media_read import RATE, _clip, _wav
from thenoise.models.config import ModelConfig
from thenoise.models.video_base import Segment, VideoModel
from thenoise.models.wan_video import VARIANTS, Wan22Model
from thenoise.vae.wan22 import WAN22_LATENTS_MEAN, WAN22_LATENTS_STD
from thenoise.vae.wan_video import _comfy_vae_config
from thenoise.video.controller import VideoPipelineController
from thenoise.video_cli import one_reference
from thenoise.jobs import Cancellation
from thenoise.utils.attention import (ATTENTION_METHODS,
                                  DEFAULT_ATTENTION)
from thenoise.video.spec import Guide, ResolvedVideo, VideoSpec, mode_of


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
        CAPABILITIES = {**VideoModel.CAPABILITIES,
                        "video": {**VideoModel.CAPABILITIES["video"], "t2v": True}}
        DEFAULT_PREFS = {**VideoModel.DEFAULT_PREFS,
                         "width": 640, "height": 384, "frames": 17, "fps": 25,
                         "steps": 8, "guidance_scale": 1.0,
                         "output_format": "mp4", "sampler": "euler",
                         "flow_shift": 5.0}

        def _clear_weights(self):
            self.extra_component = None

    model = AnotherVideo(config=ModelConfig(
        dit_paths={"": str(tmp_path / "dit")}, vae_path=str(tmp_path / "vae"),
        text_encoder_path=str(tmp_path / "text"), device="cpu", offload_device="cpu",
        headers={"": header_of(["weight"])}))
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


def test_an_adapter_declines_the_features_it_does_not_implement():
    """The vocabulary is answered per model: `upscale` False is Wan's own declaration,
    not something the video medium answers for it, because `_create_upscaler` is
    abstract and a model has to say what it built.
    """
    model = wan_adapter()
    assert model.capability("video")["t2v"]
    assert not model.capability("upscale")
    # The other medium is answered null, not with a group of refusals: a clip serves
    # no stills at all, whatever a request asks of it.
    assert model.capability("image") is None
    with pytest.raises(NotImplementedError, match="no latent upscaler"):
        model.get_upscaler()


def _stub_config(tmp_path):
    return ModelConfig(dit_paths={"": str(tmp_path / "dit")}, vae_path="vae",
                       text_encoder_path="te", device="cpu", offload_device="cpu",
                       headers={"": header_of(["weight"])})


def test_a_model_with_two_segments_gets_one_progress_clock(tmp_path, monkeypatch):
    """The controller drives spans, not models: the job sees one sampling clock and one
    stage per span, whatever the model split its schedule into.
    """
    class TwoExpert(StubVideo):
        def segments(self, params, schedule, guidance_scale):
            half = len(schedule) // 2
            return [Segment("dit", self.dit, schedule[:half], guidance_scale, params),
                    Segment("dit_low", self.dit, schedule[half:],
                            guidance_scale / 2, params)]

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


def test_shared_controller_cleans_video_state_on_failure(tmp_path, monkeypatch):
    """Request-owned AV references must be cleared even when a stage fails."""
    model = StubVideo(config=_stub_config(tmp_path))
    cleared = []
    model.clear_video_state = lambda: cleared.append(True)
    controller = VideoPipelineController(model)

    def fail(*_args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(controller, "_render", fail)
    spec = model.resolve_video(VideoSpec(prompt="fox"))
    with pytest.raises(RuntimeError, match="injected failure"):
        controller.render(spec, Cancellation(), lambda *_: None,
                          tmp_path / "failed.webm")
    assert cleared == [True]


def test_a_codec_that_streams_frames_sizes_its_decode_stage_by_one(tmp_path):
    """The default takes :meth:`decode_frames` at its word: a frame at a time.

    A codec that must run over the whole clip before the first frame can leave declares
    the clip instead, since a stage sized by a frame alone then survives only by being
    retried for memory.
    """
    model = StubVideo(config=_stub_config(tmp_path))
    resolved = model.resolve_video(VideoSpec(prompt="frames", width=128, height=96))
    assert model.decoded_pixels(resolved) == 128 * 96


def test_a_frame_count_lands_on_the_grid_a_codec_can_pack():
    """One causal-clip grid, whatever the codec's temporal compression is.

    A codec packs `compression*n+1` frames -- a first frame, then whole groups -- so the
    shortest clip it can hold is one group, and the nearest grid point to a single frame
    is a whole group rather than nothing.
    """
    from thenoise.video.spec import frame_grid

    assert frame_grid(9, compression=8, upward=True) == 9
    assert [frame_grid(n, compression=8, upward=True) for n in (1, 2, 8)] == [9, 9, 9]
    assert [frame_grid(n, compression=8, upward=True) for n in (10, 17)] == [17, 17]
    # A duration may round down, and a half goes up: 12 is nearer 9, 13 nearer 17.
    assert [frame_grid(n, compression=8, upward=False) for n in (12, 13, 20)] == [9, 17, 17]
    assert frame_grid(1.2 * 24, compression=4, upward=False) == 29  # the other codec, same rule

    for wanted in (0, -3, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive and finite"):
            frame_grid(wanted, compression=8, upward=True)


def test_reference_conditions_decide_the_mode():
    """The mode is read off what arrived, so no second switch can disagree with it."""
    assert mode_of(None, None) == "t2v"
    assert mode_of(b"png", None) == "i2v"
    assert mode_of(b"png", b"png") == "flf2v"
    for keyframes in ((None, None), (b"png", None), (b"png", b"png")):
        assert mode_of(*keyframes, references=(b"png",)) == "ref2v"


def test_a_request_carries_its_references_to_the_model():
    resolved = ResolvedVideo(
        prompt="a fox, like this", negative_prompt="", width=64, height=64, frame_count=5,
        fps=24, steps=4, guidance_scale=1.0, flow_shift=3.0, seed=0, output_format="mp4",
        image=None, end_image=None, sampler="euler", lora_specs={},
        references=(Guide(media=b"ref", frame=-1),))
    assert resolved.mode == "ref2v"


def test_references_are_refused_where_the_architecture_has_no_place_for_them():
    """The refusal is the vocabulary answering, not a check stacked beside it."""
    model = wan_adapter("ti2v_5b")
    with pytest.raises(ValueError, match="does not support ref2v"):
        VideoPipelineController(model).resolve(
            VideoSpec(prompt="a fox, like this", references=[Guide(media=b"png")]))


def test_references_reach_the_conditioner_and_the_packing(tmp_path, monkeypatch):
    """A reference goes the whole way round a clip run: shown to the language model as a
    picture, pinned to the frame the request named."""
    class RefVideo(StubVideo):
        CAPABILITIES = {**StubVideo.CAPABILITIES,
                        "video": {**StubVideo.CAPABILITIES["video"], "ref2v": True}}

        def prepare_latent(self, latents, cond, params, ref=None, ref_method="index",
                           conditions=()):
            self.conditions_seen = [one.frame for one in conditions]
            return super().prepare_latent(latents, cond, params, ref=ref,
                                          ref_method=ref_method)

    model = RefVideo(config=_stub_config(tmp_path))
    monkeypatch.setattr("thenoise.video.controller.encode_video",
                        lambda frames, spec, token, dest, progress: dest.write_bytes(b"v"))
    spec = model.resolve_video(VideoSpec(prompt="a fox, like this",
                                         references=[Guide(media=_png_bytes(), frame=-1)]))
    VideoPipelineController(model).render(spec, Cancellation(), lambda *_: None,
                                          tmp_path / "clip.webm")

    # The reference reaches the conditioner as media of its own, which is what lets a
    # model present a clip and its sound rather than one frame of each.
    guides = model.encode_prompt_args.guides
    assert len(guides) == 1 and len(guides[0].frames) == 1
    assert model.conditions_seen == [-1]


@pytest.mark.parametrize(("media", "what_is_missing"), [
    (_clip, "frames into one latent"),
    (_wav, "audio stream"),
])
def test_a_reference_the_trunk_cannot_hold_is_refused_by_what_it_lacks(
        tmp_path, media, what_is_missing):
    """A file the reader understands is not yet a file the trunk can place, and the
    refusal names the half that is missing rather than the file it was handed."""
    class RefVideo(StubVideo):
        CAPABILITIES = {**StubVideo.CAPABILITIES,
                        "video": {**StubVideo.CAPABILITIES["video"], "ref2v": True}}

    model = RefVideo(config=_stub_config(tmp_path))
    spec = model.resolve_video(VideoSpec(prompt="a fox, like this",
                                         references=[Guide(media=media())]))
    controller = VideoPipelineController(model)
    with pytest.raises(ValueError, match=what_is_missing):
        controller._encode_references([], controller._guides(spec), spec)


def test_a_reference_clip_is_handed_to_the_codec_in_one_piece(tmp_path):
    """Its frames go over as one tensor: what a causal codec takes together is one latent,
    and a model that cannot take them says so rather than keeping the first."""
    seen = []

    class ClipVideo(StubVideo):
        CAPABILITIES = {**StubVideo.CAPABILITIES,
                        "video": {**StubVideo.CAPABILITIES["video"], "ref2v": True}}

        def encode_references(self, pixels, frames=1):
            seen.extend(one.shape for one in pixels)
            return [torch.zeros(1, self.Z_DIM, 1, 1, 1)]

    model = ClipVideo(config=_stub_config(tmp_path))
    spec = model.resolve_video(VideoSpec(prompt="like this clip",
                                         references=[Guide(media=_clip(sound=False))]))
    controller = VideoPipelineController(model)
    _, [condition] = controller._encode_references([], controller._guides(spec), spec)
    assert len(seen) == 1 and seen[0][1] == 12
    assert condition.audio is None


def test_a_sound_reference_is_a_condition_with_a_sound_and_no_picture(tmp_path):
    """The reader's rate travels with the samples: the codec's grid is in samples."""
    seen = []

    class SoundVideo(StubVideo):
        CAPABILITIES = {**StubVideo.CAPABILITIES,
                        "video": {**StubVideo.CAPABILITIES["video"], "ref2v": True}}

        def encode_sound(self, pcm, sample_rate=None):
            seen.append(sample_rate)
            return torch.zeros(1, 2, 2, 4)

    model = SoundVideo(config=_stub_config(tmp_path))
    spec = model.resolve_video(VideoSpec(prompt="like this sound",
                                         references=[Guide(media=_wav())]))
    controller = VideoPipelineController(model)
    _, [condition] = controller._encode_references([], controller._guides(spec), spec)
    assert seen == [RATE]
    assert condition.video is None and condition.audio is not None


def test_two_clips_that_open_the_same_way_are_not_the_same_reference(tmp_path):
    """The frame a language model is shown cannot stand in for a clip: the latent the
    trunk reads is made of every frame of it."""
    model = StubVideo(config=_stub_config(tmp_path))
    controller = VideoPipelineController(model)

    def key(media):
        spec = model.resolve_video(VideoSpec(prompt="x", references=[Guide(media=media)]))
        guides = controller._guides(spec)
        assert [guide.frames[0].tobytes() for guide in guides] == \
               [guides[0].frames[0].tobytes()]
        return controller._cache_key_reference([guide.frames[0] for guide in guides], spec)

    assert key(_clip(frames=6)) != key(_clip(frames=12))


def _png_bytes() -> bytes:
    import io

    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, "PNG")
    return buffer.getvalue()


def test_a_reference_names_its_frame_in_the_argument():
    assert one_reference("dog.png") == ("dog.png", None)
    assert one_reference("dog.png@12") == ("dog.png", 12)
    assert one_reference("dog.png@-1") == ("dog.png", -1)
    assert one_reference("shots@night.png") == ("shots@night.png", None)


def test_a_box_without_the_kernel_answers_no_lossy_attention(tmp_path):
    """The words are the architecture's; the offer is the box's, and /health reads this."""
    class Lossy(StubVideo):
        ATTENTION_METHODS = (*StubVideo.ATTENTION_METHODS, "optimized")

        def attention_routing_blocked(self):
            return "no compiled sol_attn kernel for cuda:0"

    model = Lossy(config=_stub_config(tmp_path))
    assert model.ATTENTION_METHODS != model.attention_methods()
    assert model.attention_methods() == ATTENTION_METHODS


def test_a_refused_optimized_says_what_the_box_lacks(tmp_path):
    class Lossy(StubVideo):
        ATTENTION_METHODS = (*StubVideo.ATTENTION_METHODS, "optimized")

        def attention_routing_blocked(self):
            return "no compiled sol_attn kernel for cuda:0"

    with pytest.raises(ValueError, match="no compiled sol_attn kernel"):
        Lossy(config=_stub_config(tmp_path)).resolve_video(
            VideoSpec(prompt="fox", attention="optimized"))


def test_attention_a_model_does_not_answer_is_refused_by_name(tmp_path):
    model = StubVideo(config=_stub_config(tmp_path))
    assert model.ATTENTION_METHODS == ATTENTION_METHODS
    with pytest.raises(ValueError, match="does not answer 'optimized' attention"):
        model.resolve_video(VideoSpec(prompt="fox", attention="optimized"))
    assert model.resolve_video(VideoSpec(prompt="fox")).attention == DEFAULT_ATTENTION

