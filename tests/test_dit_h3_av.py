"""The audiovisual H3 checkpoint: who claims it, and what the file states.

Detection only. The stream itself arrives with the trunk's second half; what has to hold
first is that a video file is measured by the video loader and refused by the image one,
since the two share every name but the audio pair.
"""
from __future__ import annotations

import pytest
import torch

from conftest import header_of, write_safetensors
from thenoise.dit.h3.checkpoint import (
    H3_AV_SIGNATURE,
    H3_IMAGE_SIGNATURE,
    detect_params,
)
from thenoise.dit.h3.trunk import H3Params

HIDDEN, HEADS, HEAD_DIM, FFN, TEXT = 48, 2, 16, 32, 16
#: A video latent row is 24 channels over a 2x2 patch; an audio row is the stream's own
#: channels, unpached.
VIDEO_DIM, AUDIO_DIM = 24 * 4, 32
ROPE_FREQS, CURVE_ROWS, CURVE_DIM = 2, 5, 3


def av_state(*, layers=2, refiners=1, audio_dim=AUDIO_DIM, video_out_rows=VIDEO_DIM,
             audio_out_rows=None) -> dict:
    """The release's key names at a fraction of the width, as a video checkpoint spells them.

    ``audio_out_rows`` exists to be wrong: the stream states its width twice and a file
    that disagrees between the two describes a model nothing can wire.
    """
    state = {
        "video_patch_proj.weight": torch.zeros(HIDDEN, VIDEO_DIM),
        "audio_patch_proj.weight": torch.zeros(HIDDEN, audio_dim),
        "condition_proj.weight": torch.zeros(HIDDEN, TEXT),
        "adaln_t_table": torch.zeros(CURVE_ROWS, CURVE_DIM),
        "rope.inv_freq": torch.zeros(ROPE_FREQS),
        "final_layer.video_out.weight": torch.zeros(video_out_rows, HIDDEN),
        "final_layer.audio_out.weight": torch.zeros(
            AUDIO_DIM if audio_out_rows is None else audio_out_rows, HIDDEN),
        "final_layer.adaln_proj.linear.weight": torch.zeros(2 * HIDDEN, CURVE_DIM),
    }
    for i in range(layers):
        state.update({
            f"blocks.{i}.attn.qkv_proj.weight": torch.zeros(3 * HEADS * HEAD_DIM, HIDDEN),
            f"blocks.{i}.attn.q_norm.weight": torch.zeros(HEAD_DIM),
            f"blocks.{i}.attn.k_norm.weight": torch.zeros(HEAD_DIM),
            f"blocks.{i}.mlp.fc1.weight": torch.zeros(2 * FFN, HIDDEN),
            f"blocks.{i}.mlp.fc2.weight": torch.zeros(HIDDEN, FFN),
            f"blocks.{i}.adaln_proj.linear.weight": torch.zeros(6 * HIDDEN, CURVE_DIM),
        })
    for i in range(refiners):
        state[f"token_refiner.blocks.{i}.attn.qkv_proj.weight"] = torch.zeros(
            3 * HEADS * HEAD_DIM, HIDDEN)
    return state


def _image_names(state: dict) -> list:
    return [k for k in state if k not in AUDIO_NAMES]


#: The audio stream's two names, in the file's own spelling.
AUDIO_NAMES = ("audio_patch_proj.weight", "final_layer.audio_out.weight")


def _detect(tmp_path, state, signature=H3_AV_SIGNATURE):
    return detect_params(write_safetensors(tmp_path / "h3-av.safetensors", state), signature)


# ---------------------------------------------------------------- detection


def test_the_audio_pair_is_what_separates_the_two_models():
    """Every other name is shared, so the image adapter has to refuse the video file.

    Half-loading it would build a trunk that never reads the second stream: a plausible
    silent video, not an error.
    """
    video = header_of(av_state(), path="minimax_h3_fl2va.safetensors")
    image = header_of(_image_names(av_state()))

    assert video.identifies(H3_AV_SIGNATURE)
    assert not video.identifies(H3_IMAGE_SIGNATURE)
    assert image.identifies(H3_IMAGE_SIGNATURE)
    assert not image.identifies(H3_AV_SIGNATURE)


def test_the_image_loader_names_the_stream_it_refuses(tmp_path):
    with pytest.raises(ValueError, match="carries.*audio_patch_proj"):
        _detect(tmp_path, av_state(), H3_IMAGE_SIGNATURE)


def test_an_unpruned_checkpoint_is_sent_to_the_pruned_file(tmp_path):
    """Without the curve there is nothing to project the modulation onto.

    "missing 'adaln_t_table'" is true and useless: it does not say which file to get.
    """
    unpruned = av_state()
    unpruned.pop("adaln_t_table")
    unpruned["time_embedder.proj_in.weight"] = torch.zeros(256, HIDDEN)
    with pytest.raises(ValueError, match="unpruned"):
        _detect(tmp_path, unpruned)
    # The name test alone would have called this an unknown model, not this model's
    # unsupported variant.
    assert header_of(unpruned).missing(H3_AV_SIGNATURE) == ["adaln_t_table"]


# ------------------------------------------------------------------ geometry


def test_detect_reads_the_architecture_a_video_checkpoint_states(tmp_path):
    detected = _detect(tmp_path, av_state(layers=3, refiners=2))

    assert (detected.hidden_size, detected.num_layers, detected.token_refiner_layers) == (
        HIDDEN, 3, 2)
    assert (detected.num_heads, detected.head_dim) == (HEADS, HEAD_DIM)
    assert (detected.ffn_hidden_size, detected.text_dim) == (FFN, TEXT)
    assert (detected.latents_dim, detected.rope_inv_freq) == (24, ROPE_FREQS)
    assert (detected.adaln_grid, detected.adaln_dim) == (CURVE_ROWS, CURVE_DIM)


def test_the_audio_stream_is_read_from_both_of_its_names(tmp_path):
    """Its width is the projection's input and the head's output, and it is not a patch.

    The pruned video release ships no metadata at all, so the geometry is shapes or
    nothing: 32 channels, one output head over the plain two-point grid.
    """
    detected = _detect(tmp_path, av_state())

    assert detected.audio_dim == AUDIO_DIM
    assert detected.pdd_steps == 1
    assert detected.pdd_sigmas == (1.0, 0.0)
    assert detected.video_dim == VIDEO_DIM


def test_an_image_checkpoint_reads_as_a_model_without_audio(tmp_path):
    assert _detect(tmp_path, {k: v for k, v in av_state().items()
                              if k not in AUDIO_NAMES},
                   H3_IMAGE_SIGNATURE).audio_dim is None


def test_a_stream_whose_two_widths_disagree_is_refused(tmp_path):
    """One width feeds the trunk's audio rows, the other emits them: they are one number."""
    with pytest.raises(ValueError, match="audio stream's two widths"):
        _detect(tmp_path, av_state(audio_out_rows=AUDIO_DIM * 2))


def test_a_head_bank_stored_quantized_cannot_fill_the_fp32_island(tmp_path):
    """The pruned ``w6a8`` release does this to ``final_layer.video_out``.

    The packed layouts live in ``QuantizedLinear``; this head is an ``Fp32Linear``, so a
    6-bit weight has nowhere to go. Loading it anyway is how an output head gets rounded
    twice without anything saying so.
    """
    state = av_state()
    state["final_layer.video_out.weight"] = torch.zeros(VIDEO_DIM, HIDDEN, dtype=torch.int8)
    with pytest.raises(ValueError, match=r"final_layer.video_out.weight.*stored I8"):
        _detect(tmp_path, state)


def test_an_island_stored_at_half_precision_is_not_the_same_problem(tmp_path):
    """Half to full is exact, so a file that ships the curve as FP16 is not refused."""
    state = av_state()
    state["adaln_t_table"] = torch.zeros(CURVE_ROWS, CURVE_DIM, dtype=torch.float16)
    assert _detect(tmp_path, state).adaln_grid == CURVE_ROWS


def test_a_head_bank_wider_than_one_row_still_needs_its_distilled_grid(tmp_path):
    """The video file's single head is read the same way as the image model's six.

    A multi-row bank without a grid to place its rows in would be averaged against
    sigmas nobody distilled, so it is refused rather than guessed at.
    """
    with pytest.raises(ValueError, match="head bank is 2 rows"):
        _detect(tmp_path, av_state(video_out_rows=VIDEO_DIM * 2))


# --------------------------------------------------------------------- the second stream
#
# Packing and forwards at a reduced width. The expected coordinates below were written
# out of the reference implementation's packed layout once, then derived from the
# constants here.

from thenoise.dit.h3.av import H3AVDiT
from thenoise.dit.h3.layout import (
    COND_TIMESTEP,
    FRAME_PER_TOKEN,
    FRAME_RESCALE,
    Layout,
    frame_grid,
    time_spans,
)
from thenoise.dit.h3.trunk import pack_audio, unpack_audio

AUDIO_STREAM = 6
PROMPT = 5
LATENTS = (2, 4, 6)  # padded already: latent_t, h, w
AUDIO_T = 3


def _av_params(**over) -> H3Params:
    base = dict(
        hidden_size=HIDDEN,
        num_layers=2,
        token_refiner_layers=1,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        ffn_hidden_size=FFN,
        latents_dim=4,
        text_dim=TEXT,
        rope_inv_freq=ROPE_FREQS,
        adaln_grid=CURVE_ROWS,
        adaln_dim=CURVE_DIM,
        pdd_steps=1,
        pdd_sigmas=(1.0, 0.0),
        audio_dim=AUDIO_STREAM,
    )
    return H3Params(**{**base, **over})


def _av_dit(params: H3Params | None = None) -> H3AVDiT:
    torch.manual_seed(0)
    dit = H3AVDiT(params or _av_params()).eval().requires_grad_(False)
    dit.adaln_t_table.normal_()
    dit.rope.inv_freq.copy_(torch.tensor([1.0, 0.25]))
    return dit


def _av_layout(keyframes=(), **over) -> Layout:
    opts = dict(text_len=PROMPT, latent_t=LATENTS[0], latent_h=LATENTS[1],
                latent_w=LATENTS[2], audio_t=AUDIO_T, keyframes=keyframes)
    return Layout.av(**{**opts, **over})


def _run(dit, layout, latent=None, audio=None, **over):
    torch.manual_seed(11)
    latent = (torch.randn(1, dit.params.latents_dim, *layout.latents)
              if latent is None else latent)
    audio = torch.randn(1, AUDIO_STREAM, 2, AUDIO_T) if audio is None else audio
    context = torch.randn(1, layout.text_len, dit.params.hidden_size)
    dit.prepare_rope(layout, context.dtype)
    levels = dict(sigma=0.8, audio_sigma=0.5)
    return dit(latent, audio, context=context, layout=layout, **{**levels, **over})


def test_av_packing_is_the_released_order():
    lay = Layout.av(text_len=5, latent_t=4, latent_h=6, latent_w=8, audio_t=3,
                    keyframes=[0, 12])
    assert lay.segments == [(0, 5, "text"), (5, 17, "cond"), (17, 29, "cond"),
                            (29, 35, "audio"), (35, 83, "video")]
    assert lay.seq_len == 83
    assert lay.video_span == (35, 83)
    assert lay.audio_span == (29, 35)
    assert lay.video_rows == 12 * 2 + 4 * 12  # both keyframes and the target
    assert lay.audio_rows == 2 * 3


def test_a_keyframe_is_pinned_to_the_pixel_frame_it_represents():
    lay = Layout.av(text_len=5, latent_t=2, latent_h=4, latent_w=4, audio_t=2,
                    keyframes=[0, 5])
    t = lay.position_ids[:, 0]
    assert t[5:9].unique().tolist() == [5.0]
    assert t[9:13].unique().tolist() == [5.0 + FRAME_RESCALE * 5]


def test_the_video_time_axis_advances_by_each_frames_own_span():
    lay = Layout.av(text_len=3, latent_t=6, latent_h=2, latent_w=2, audio_t=2)
    t = lay.position_ids[lay.video_span[0]:].view(6, 1, 3)[:, 0, 0]
    expected = [3.0]
    for span in time_spans(6)[:-1]:
        expected.append(expected[-1] + span)
    assert torch.equal(t, torch.tensor(expected, dtype=t.dtype))
    assert time_spans(6) == [FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] for k in range(6)]


def test_audio_rows_are_pinned_to_the_targets_width_extremes():
    """The stream has no spatial extent of its own; its two channels sit at either end of
    the picture's width axis so rope still separates them.
    """
    _, w_axis = frame_grid(6, 8)
    lay = Layout.av(text_len=5, latent_t=2, latent_h=6, latent_w=8, audio_t=3)
    rows = lay.position_ids[lay.audio_span[0]:lay.audio_span[1]]
    assert rows[:, 1].tolist() == [0.0] * 6
    assert rows[:3, 2].tolist() == [float(w_axis[0])] * 3
    assert rows[3:, 2].tolist() == [float(w_axis[-1])] * 3
    assert rows[:, 0].tolist() == [5.0, 6.0, 7.0, 5.0, 6.0, 7.0]


def test_the_two_streams_are_modulated_at_their_own_levels():
    lay = _av_layout(keyframes=[0])
    timesteps = lay.timesteps(0.2, 0.7)
    assert timesteps.values == (0.2, 0.7, COND_TIMESTEP)
    assert timesteps.rows == {"text": 0, "cond": 2, "audio": 1, "video": 0}
    # Without a keyframe the pinned row is not in the table at all.
    assert lay.timesteps(0.2, 0.7).rows["cond"] == 2
    assert _av_layout().timesteps(0.2, 0.7).values == (0.2, 0.7)


def test_a_packing_with_audio_demands_both_timesteps():
    with pytest.raises(ValueError, match="audio stream"):
        _av_layout().timesteps(0.5)


def test_audio_rows_are_channel_major():
    latent = torch.tensor([[[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]]])
    assert pack_audio(latent).view(-1).tolist() == [0, 1, 2, 10, 11, 12]
    assert torch.equal(unpack_audio(pack_audio(latent)), latent)


def test_a_forward_returns_both_streams_at_their_latent_shapes():
    dit, lay = _av_dit(), _av_layout()
    video, audio = _run(dit, lay)
    assert video.shape == (1, dit.params.latents_dim, *LATENTS)
    assert audio.shape == (1, AUDIO_STREAM, 2, AUDIO_T)
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()


def test_the_streams_reach_each_other():
    """One joint model, not two: the audio is attended to while the picture is denoised."""
    dit, lay = _av_dit(), _av_layout()
    video, _ = _run(dit, lay)
    other_audio = torch.randn(1, AUDIO_STREAM, 2, AUDIO_T)
    video2, _ = _run(dit, lay, audio=other_audio)
    assert not torch.allclose(video, video2)


def test_muting_the_audio_head_leaves_the_picture_alone():
    dit, lay = _av_dit(), _av_layout()
    video, _ = _run(dit, lay)
    dit.final_layer.audio_out.weight.data.zero_()
    dit.final_layer.audio_out.bias.data.zero_()
    video_muted, audio_muted = _run(dit, lay)
    assert torch.equal(video, video_muted)
    assert torch.equal(audio_muted, torch.zeros_like(audio_muted))


def test_a_keyframe_conditions_the_run():
    dit = _av_dit()
    keyframe = torch.randn(1, dit.params.latents_dim, 1, LATENTS[1], LATENTS[2])
    plain, _ = _run(dit, _av_layout())
    with_key, _ = _run(dit, _av_layout(keyframes=[0]), keyframes=[keyframe])
    assert not torch.allclose(plain, with_key)

    rows = dit._keyframe_rows(keyframe, seed=7)
    bare = dit.patchify(keyframe)
    noise = torch.randn(bare.shape, generator=torch.Generator().manual_seed(7),
                        dtype=torch.float32)
    assert torch.equal(rows, COND_TIMESTEP * bare + (1.0 - COND_TIMESTEP) * noise)
    # The stream restarts for every keyframe, so a second one takes the same draw. The
    # reconstruction above is exact; recovering the draw from two mixes is not.
    other = torch.randn_like(keyframe)
    assert torch.allclose(dit._keyframe_rows(other, seed=7) - COND_TIMESTEP * dit.patchify(other),
                          rows - COND_TIMESTEP * bare, atol=1e-5)


def test_the_forward_refuses_a_pack_the_layout_does_not_describe():
    dit = _av_dit()
    lay = _av_layout()
    with pytest.raises(ValueError, match="layout was built for"):
        dit(torch.randn(1, 4, 2, 6, 6), torch.randn(1, AUDIO_STREAM, 2, AUDIO_T),
            0.8, 0.5, torch.randn(1, PROMPT, dit.params.hidden_size), lay)
    with pytest.raises(ValueError, match="audio rows"):
        dit(torch.randn(1, 4, 2, 4, 6), torch.randn(1, AUDIO_STREAM, 2, AUDIO_T + 1),
            0.8, 0.5, torch.randn(1, PROMPT, dit.params.hidden_size), lay)


def test_a_model_without_the_stream_is_not_an_audiovisual_one():
    with pytest.raises(ValueError, match="no audio stream"):
        H3AVDiT(_av_params(audio_dim=None))


# ---------------------------------------------------------------- the two schedules
#
# One solver steps the video's sigma while the audio stream reads its own, and the audio
# slice of the pack travels in between. The algebra here is what makes that exact rather
# than fudged; SHIFT is the released video shift.

from thenoise.dit.h3.sampling import (
    AUDIO_SHIFT,
    audio_sigma,
    audio_velocity,
    carry,
    carry_ratio,
    get_sigmas,
    reshift,
    shift_ratio,
)

SHIFT = 12.0


def test_the_ladder_runs_the_shifted_schedule_from_noise_to_clean():
    sigmas = get_sigmas(20, shift=SHIFT)
    assert len(sigmas) == 21
    assert sigmas[0] == 1.0 and sigmas[-1] == 0.0
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))


def test_the_two_schedules_share_their_endpoints():
    """Noise is noise and clean is clean under either shift; only between differs."""
    for sigma in get_sigmas(12, shift=SHIFT):
        own = audio_sigma(sigma, shift=SHIFT)
        assert own <= sigma
        assert reshift(own, from_shift=AUDIO_SHIFT, to_shift=SHIFT) == pytest.approx(sigma)


def test_the_carry_is_the_ratio_of_the_two_sigmas():
    # The closed form is also the only answer at the clean end, where both sigmas are 0.
    for sigma in get_sigmas(20, shift=SHIFT)[1:-1]:
        assert carry_ratio(sigma, shift=SHIFT) == pytest.approx(
            sigma / audio_sigma(sigma, shift=SHIFT), rel=1e-12)


def test_the_carry_is_linear_in_the_video_sigma_and_unscaled_at_the_start():
    """The start needs no scaling because the two schedules meet at noise; the clean end
    is where the pack's audio slice has grown to ``S`` times the stream's own latent.
    """
    assert carry_ratio(1.0, shift=SHIFT) == 1.0
    assert carry_ratio(0.0, shift=SHIFT) == shift_ratio(SHIFT) == 4.0
    assert carry_ratio(0.5, shift=SHIFT) == pytest.approx(2.5)
    assert carry(carry_ratio(0.3, shift=SHIFT) * torch.ones(2), 0.3, shift=SHIFT) \
        == pytest.approx(torch.ones(2))


def test_the_carrys_two_derivatives_are_constant_in_the_base_grid():
    """The identities that let a velocity be converted without a Jacobian."""
    h = 1e-6
    for sigma in (0.05, 0.3, 0.7, 0.99):
        r = carry_ratio(sigma, shift=SHIFT)
        d_r = (carry_ratio(sigma + h, shift=SHIFT) - carry_ratio(sigma - h, shift=SHIFT)) / (2 * h)
        assert d_r == pytest.approx(1.0 - shift_ratio(SHIFT), abs=1e-5)
        d_a = (audio_sigma(sigma + h, shift=SHIFT) - audio_sigma(sigma - h, shift=SHIFT)) / (2 * h)
        assert r * d_a == pytest.approx(1.0 + (shift_ratio(SHIFT) - 1.0)
                                        * audio_sigma(sigma, shift=SHIFT), abs=1e-5)


def test_the_velocity_conversion_is_the_derivative_of_the_carry():
    """Step a stream along its own schedule, carry it along the video's, and compare the
    finite difference with what :func:`audio_velocity` answers.
    """
    x0 = torch.tensor([[-1.7, 0.4]], dtype=torch.float64)
    noise = torch.tensor([[2.2, -0.9]], dtype=torch.float64)
    h = 1e-5

    def carried(sigma_video):
        sigma_audio = audio_sigma(sigma_video, shift=SHIFT)
        own = (1.0 - sigma_audio) * x0 + sigma_audio * noise
        return carry_ratio(sigma_video, shift=SHIFT) * own

    for sigma in (0.2, 0.55, 0.9):
        finite = (carried(sigma + h) - carried(sigma - h)) / (2 * h)
        sigma_audio = audio_sigma(sigma, shift=SHIFT)
        own = (1.0 - sigma_audio) * x0 + sigma_audio * noise
        predicted = audio_velocity(noise - x0, own, sigma=sigma, shift=SHIFT)
        assert torch.allclose(predicted, finite, atol=1e-6)
