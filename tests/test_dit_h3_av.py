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
