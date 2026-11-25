"""The H3 video adapter at a reduced trunk width: what a request becomes, the pack the
solver steps over both streams, the audio carry at either end of a run, and a clip and its
sound coming out of the codecs.

The trunk and both codecs here are the released module trees at a fraction of their size,
so a step is the real packing, the real rope and the real shift maps — only small.
"""
from __future__ import annotations

import pytest
import torch
from PIL import Image
from types import SimpleNamespace

from conftest import header_of
from test_dit_h3_av import _av_params, av_state
from test_vae_h3_audio import TINY as AUDIO_GEOMETRY, _fill as fill_audio
from test_vae_h3_video import _fill as fill_video
from thenoise.dit.h3 import H3AVDiT
from thenoise.models.config import VideoSamplingParams
from thenoise.models.h3_image import H3ImageModel
from thenoise.models.h3_video import (
    TRAINING_AREA,
    H3VideoConditioning,
    H3VideoModel,
)
from thenoise.samplers import create_sampler
from thenoise.vae import AutoencoderKLH3Audio, AutoencoderKLH3Video
from thenoise.vae.h3 import TemporalPlan
from thenoise.video.spec import VideoSpec

#: Prompt tokens, and the canvas the tiny codecs can still round-trip.
TEXT = 5
SIDE = 32
HIDDEN = _av_params().hidden_size


def _dit() -> H3AVDiT:
    """The trunk at H3's shape ratios and a twentieth of its width: the codec's 24-channel
    video lattice and the audio codec's 32-channel stream."""
    torch.manual_seed(0)
    dit = H3AVDiT(_av_params(latents_dim=24, audio_dim=32)).eval().requires_grad_(False)
    dit.adaln_t_table.normal_()
    dit.rope.inv_freq.copy_(torch.tensor([1.0, 0.25]))
    return dit


@pytest.fixture(scope="module")
def model():
    m = object.__new__(H3VideoModel)
    m.device = "cpu"
    m.dtype = torch.float32
    m.dit = _dit()
    m.vae = fill_video(AutoencoderKLH3Video(ch=32, dim=64, num_layers=1)
                       ).eval().requires_grad_(False)
    m.audio_vae = fill_audio(AutoencoderKLH3Audio(**AUDIO_GEOMETRY)
                             ).eval().requires_grad_(False)
    m._reset_run()
    return m


def _params(frames=5, steps=2, height=SIDE, width=SIDE, fps=24, flow_shift=12.0,
            seed=3) -> VideoSamplingParams:
    return VideoSamplingParams(height=height, width=width, steps=steps, seed=seed,
                               guidance_scale=1.0, sampler="euler", frames=frames,
                               flow_shift=flow_shift, fps=fps)


def _conditioning() -> H3VideoConditioning:
    torch.manual_seed(7)
    cond = torch.randn(1, TEXT, HIDDEN)
    mask = torch.ones(1, TEXT, dtype=torch.long)
    return H3VideoConditioning(cond=cond, cond_mask=mask, tags=torch.zeros(1, TEXT, dtype=torch.long))


def _prepare(model, params, cond=None, refs=()):
    latents = model.init_latents(params)
    prepared = model.prepare_latent(latents, cond or _conditioning(), params,
                                    ref=list(refs), ref_method="index_timestep_zero")
    model.begin_sampling(model.schedule(params))
    return prepared


# --- what a request becomes ---------------------------------------------------


def test_a_request_becomes_two_latent_shapes(model):
    streams = model._streams(_params(frames=22, height=768, width=1344))
    assert streams.video == (24, 7, 48, 84)
    assert streams.audio == (32, 2, 37)


def test_the_sound_is_as_long_as_the_picture(model):
    """The picture is 124 frames at 24 fps; the sound has to be the same length of time."""
    for frames in (5, 22, 124):
        streams = model._streams(_params(frames=frames))
        seconds = model.audio_vae.decode_output_length(streams.audio[2]) / 32000
        assert abs(seconds - frames / 24) < 1 / 40


@pytest.mark.parametrize(("spec", "expected"), [
    ({"frames": 24}, 39),
    ({"frames": 22}, 22),
    ({"frames": 14}, 22),
    ({"frames": 130}, 141),
    ({"duration": 1.0}, 22),
    # The two numbers the released aligner is documented on: never down for a
    # named frame count, nearest for a duration.
    ({"duration": 130 / 24}, 124),
    ({}, 124),
])
def test_a_request_lands_on_a_length_the_codec_runs(spec, expected):
    resolved = H3VideoModel.resolve_video(_canvas(), VideoSpec(prompt="a clip", **spec))
    assert resolved.frame_count == expected
    # Every length the resolver hands out is one the codec decodes back exactly.
    assert TemporalPlan.of_frames(resolved.frame_count).frames == resolved.frame_count


def test_the_canvas_is_rounded_onto_the_grid():
    assert H3VideoModel.resolve_size(_canvas(), 1000, 600) == (992, 576)
    with pytest.raises(ValueError, match="at least"):
        H3VideoModel.resolve_size(_canvas(), 16, 768)


def test_a_canvas_bigger_than_the_trained_area_runs_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        assert H3VideoModel.resolve_size(_canvas(), 1920, 1088) == (1920, 1088)
    assert "off distribution" in caplog.text
    assert TRAINING_AREA == 768 * 1344


def _canvas():
    """Just enough adapter to call the two request hooks, which need no weights."""
    return object.__new__(H3VideoModel)


# --- the pack ----------------------------------------------------------------


def test_the_pack_carries_both_streams_as_one_row(model):
    params = _params()
    streams = model._streams(params)
    latents = model.init_latents(params)
    assert latents.shape == (1, streams.video_rows + streams.audio_rows)
    video, audio = streams.split(latents)
    assert video.shape == (1, *streams.video)
    assert audio.shape == (1, *streams.audio)
    assert torch.equal(latents, streams.join(video, audio))


def test_the_same_seed_gives_the_same_clip_and_sound(model):
    params = _params()
    assert torch.equal(model.init_latents(params), model.init_latents(params))
    assert not torch.equal(model.init_latents(params),
                           model.init_latents(_params(seed=params.seed + 1)))


def test_the_layout_packs_keyframes_ahead_of_the_clip(model):
    params = _params()
    streams = model._streams(params)
    cells = (streams.video[2] // 2) * (streams.video[3] // 2)
    refs = [torch.zeros(1, 24, 1, *streams.video[2:]), torch.ones(1, 24, 1, *streams.video[2:])]
    _prepare(model, params, refs=refs)
    assert [kind for _, _, kind in model._layout.segments] == [
        "text", "cond", "cond", "audio", "video"]
    assert [stop - start for start, stop, kind in model._layout.segments] == [
        TEXT, cells, cells, 2 * streams.audio[2], streams.video[1] * cells]


def test_a_keyframe_is_never_written_into_the_clip(model):
    params = _params()
    streams = model._streams(params)
    latents = model.init_latents(params)
    out = model.prepare_latent(latents, _conditioning(), params,
                               ref=[torch.ones(1, 24, 1, *streams.video[2:])],
                               ref_method="index_timestep_zero")
    assert torch.equal(out, latents), "a keyframe is a row of its own, not an overwrite"


def test_the_second_branch_gets_its_own_packing(model):
    params = _params()
    cond = _conditioning()
    cond.null = torch.randn(1, TEXT + 2, HIDDEN)
    cond.null_mask = torch.ones(1, TEXT + 2, dtype=torch.long)
    _prepare(model, params, cond=cond)
    assert model._null_layout.text_len == TEXT + 2
    # The branch's rope table is built for its own sequence, not the first one's.
    cos, _ = model.dit.pe_embedder["uncond"]
    assert cos.shape[-2] == model._null_layout.seq_len > model._layout.seq_len


def test_a_third_keyframe_is_refused(model):
    images = [Image.new("RGB", (64, 64))] * 3
    with pytest.raises(ValueError, match="at most"):
        model.fit_references(images, (SIDE, SIDE))


def test_a_reference_method_that_overwrites_is_refused(model):
    params = _params()
    streams = model._streams(params)
    with pytest.raises(ValueError, match="denoises it"):
        model.prepare_latent(model.init_latents(params), _conditioning(), params,
                             ref=[torch.zeros(1, 24, 1, *streams.video[2:])],
                             ref_method="index")


# --- the run -----------------------------------------------------------------


def test_the_schedule_walks_the_video_stream_from_noise_to_clean(model):
    steps = model.schedule(_params(steps=4))
    assert len(steps) == 4
    assert steps[0].t == pytest.approx(1.0)
    assert sum(step.delta for step in steps) == pytest.approx(steps[0].t)
    assert all(step.delta > 0 for step in steps)


def test_one_run_moves_both_streams_and_stays_finite(model):
    params = _params()
    cond = _conditioning()
    x = _prepare(model, params, cond=cond)
    out = create_sampler("euler", model).sample(x, model.schedule(params), cond, 1.0,
                                                params.seed)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert not torch.equal(out, x)


def test_the_audio_is_read_as_itself_at_pure_noise(model, monkeypatch):
    """The carry is 1 at sigma 1, so nothing has to be scaled off the starting audio."""
    params = _params()
    streams = model._streams(params)
    x = _prepare(model, params)
    _, carried = streams.split(x)
    seen = {}
    forward = model.dit.forward

    def spy(*args, **kwargs):
        seen["audio"] = args[1]
        return forward(*args, **kwargs)

    monkeypatch.setattr(model.dit, "forward", spy)
    model.denoise_step(x, torch.tensor(1.0), _conditioning(), 1.0, 0)
    assert torch.equal(seen["audio"], carried)


@pytest.mark.parametrize(("flow_shift", "scale"), [(12.0, 4.0), (6.0, 2.0)])
def test_the_finished_sound_is_un_carried_by_the_shift_ratio(model, monkeypatch,
                                                             flow_shift, scale):
    """The pack ends carrying ``S = video shift / audio shift`` times the latent the audio
    codec was trained on, whatever path the run took to get there."""
    params = _params(flow_shift=flow_shift)
    streams = model._streams(params)
    latents = streams.join(torch.zeros(1, *streams.video),
                           torch.full((1, *streams.audio), scale))
    seen = {}

    def vocoder(latent):
        seen["latent"] = latent
        return torch.zeros(1, 2, streams.audio[2] * model.audio_vae.samples_per_latent)

    monkeypatch.setattr(model.audio_vae, "decode_to_pcm", vocoder)
    sound = model.decode_audio(latents, params)
    assert float(seen["latent"].mean()) == pytest.approx(1.0)
    assert sound.sample_rate == 32000
    assert sound.samples.shape == (2, streams.audio[2] * 800)


def test_the_pack_is_decoded_a_frame_at_a_time(model):
    params = _params(frames=5)
    x = _prepare(model, params)
    frames = list(model.decode_frames(x, params))
    assert len(frames) == params.frames
    assert frames[0].shape == (3, SIDE, SIDE)
    assert all(frame.abs().max() <= 1.0 for frame in frames)


def test_a_whole_run_hands_back_a_clip_and_its_sound(model):
    params = _params()
    cond = _conditioning()
    x = _prepare(model, params, cond=cond)
    x = model.finalize_latent(
        create_sampler("euler", model).sample(x, model.schedule(params), cond, 1.0,
                                              params.seed), params)
    assert len(list(model.decode_frames(x, params))) == params.frames
    sound = model.decode_audio(x, params)
    assert sound.samples.shape[1] == (model._streams(params).audio[2] * 800)
    assert torch.isfinite(sound.samples).all()
    assert model.audio_components() == ("audio_vae",)


def test_the_decode_stage_is_sized_by_a_clip_of_pixels(model):
    """The codec walks a whole clip before it hands a frame back, so the headroom that
    suffices for a codec answering one frame at a time is a fraction of this one's."""
    plan = TemporalPlan.of(1)
    per_clip = (plan.chunk_tokens + plan.overlap_tokens) * plan.ratio_t
    assert model.decoded_pixels(SimpleNamespace(width=1344, height=768)) == (1344
                                                                            * 768 * per_clip)
    # Every frame of the clip being walked is live, whatever the canvas.
    assert model.decoded_pixels(SimpleNamespace(width=64, height=64)) == 64 * 64 * per_clip


# --- claiming the checkpoint --------------------------------------------------


def test_the_audio_pair_claims_the_video_adapter():
    av = header_of(av_state(), path="minimax_h3_fl2va.safetensors")
    assert H3VideoModel.detect(av)
    assert not H3ImageModel.detect(av)
    image = header_of({k: v for k, v in av_state().items()
                       if not k.startswith("audio")
                       and k != "final_layer.audio_out.weight"})
    assert H3ImageModel.detect(image)
    assert not H3VideoModel.detect(image)


def test_a_silent_request_is_refused_before_anything_loads():
    with pytest.raises(ValueError, match="--audio-vae"):
        H3VideoModel.validate_paths({"": header_of(av_state())})
