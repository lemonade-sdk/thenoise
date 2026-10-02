"""Model-side helpers that are pure functions of their inputs.

No weights and no device: these are the pieces whose silent failure shows up as a
wrong schedule or a mis-shaped token stream.
"""
from __future__ import annotations

import math
import pytest
import torch

from conftest import write_safetensors
from thenoise.dit.anima.utils import _count_anima_blocks
from thenoise.utils.text_encoder import QWEN_VL_DROP_IDX, compute_drop_idx
from thenoise.dit.krea2.sampling import (
    encode_prompts,
    gather_valid_text,
    prepare,
    timesteps,
)


# ------------------------------------------------------------------ krea2 grid


def test_krea2_timesteps_run_one_to_zero_and_shift_only_with_interpolated_mu():
    """The distilled checkpoint passes an explicit mu, which pins the grid: the
    resolution shift must not leak into it.
    """
    shifted = [timesteps(seq, 8, 256, 6400, y1=0.5, y2=1.15) for seq in (256, 4096)]
    assert len(shifted[0]) == len(shifted[1]) == 9
    assert shifted[0] != shifted[1]
    assert shifted[0][0] == 1.0 and shifted[0][-1] == 0.0

    pinned = [timesteps(seq, 8, 256, 6400, mu=1.15) for seq in (256, 1024, 4096)]
    assert pinned[0] == pinned[1] == pinned[2]
    assert all(a > b for a, b in zip(pinned[0], pinned[0][1:]))  # monotonic


# -------------------------------------------------------- krea2 patchify/pos


def test_prepare_patchifies_the_latent_and_builds_positions():
    img = torch.arange(1 * 4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4)
    txtmask = torch.tensor([[True, True, False]])

    tokens, pos, mask = prepare(img, txtlen=3, patch=2, txtmask=txtmask)

    # 4x4 latent with patch 2 -> 2x2 = 4 image tokens of 4*2*2 channels.
    assert tokens.shape == (1, 4, 16)
    # Image tokens lead, then the text tokens; the mask carries both.
    assert pos.shape == (1, 4 + 3, 3)
    assert mask.shape == (1, 4 + 3)
    assert mask[0, :4].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert mask[0, 4:].tolist() == [1.0, 1.0, 0.0]
    # Image positions are (t, h, w) with t=0 and a row-major h/w grid...
    assert pos[0, :4, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert pos[0, :4, 1].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert pos[0, :4, 2].tolist() == [0.0, 1.0, 0.0, 1.0]
    # ...and text tokens sit at the origin.
    assert torch.equal(pos[0, 4:], torch.zeros(3, 3))


def test_gather_valid_text_compacts_the_valid_tokens():
    # (B, seq, L, D) with a [valid, pad, valid] mask, as the Qwen3-VL conditioner
    # produces (prompt, padding, template suffix).
    txt = torch.arange(2 * 4 * 1 * 3, dtype=torch.float32).reshape(2, 4, 1, 3)
    mask = torch.tensor([[True, False, False, True], [True, True, True, True]])

    out, out_mask = gather_valid_text(txt, mask)

    # Right-padded to the batch's maximum valid count.
    assert out.shape == (2, 4, 1, 3)
    assert out_mask.tolist() == [[True, True, False, False], [True, True, True, True]]
    assert torch.equal(out[0, 0], txt[0, 0])
    assert torch.equal(out[0, 1], txt[0, 3])  # the trailing valid token moved up
    assert torch.equal(out[1], txt[1])


# ----------------------------------------------------------- krea2 encode_prompts


class _Recorder:
    """Stand-in encoder that records the prompt batches it is handed."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompts):
        self.calls.append(list(prompts))
        batch = len(prompts)
        # (B, seq, L, D) with a fully-valid mask (gather is then a no-op).
        txt = torch.arange(batch * 4 * 1 * 2, dtype=torch.float32).reshape(batch, 4, 1, 2)
        mask = torch.ones(batch, 4, dtype=torch.bool)
        return txt, mask


def test_krea2_encode_prompts_encodes_the_unconditional_branch_only_when_cfg():
    enc = _Recorder()
    txt, txtmask, untxt, untxtmask = encode_prompts(enc, ["a", "b"], cfg=True)
    # No negatives given -> a batch of empty prompts, shaped like the conditional one.
    assert enc.calls == [["a", "b"], ["", ""]]
    assert untxt.shape == txt.shape and untxtmask.shape == txtmask.shape

    enc = _Recorder()
    encode_prompts(enc, ["a"], negative_prompts=["bad"], cfg=True)
    assert enc.calls == [["a"], ["bad"]]

    enc = _Recorder()
    assert encode_prompts(enc, ["a"], cfg=False)[2] is None
    assert enc.calls == [["a"]]


# ------------------------------------------------------------------ anima utils


def test_count_anima_blocks_reads_the_header(tmp_path):
    keys = [f"blocks.{i}.mlp.weight" for i in range(28)] + ["x_embedder.weight"]
    path = write_safetensors(tmp_path / "anima.safetensors", {k: torch.zeros(1) for k in keys})
    assert _count_anima_blocks(path) == 28


@pytest.mark.parametrize("prefix", ["", "net.", "model.diffusion_model."])
def test_count_anima_blocks_ignores_the_wrapper_prefix(tmp_path, prefix):
    """Raw and repackaged checkpoints must count identically."""
    keys = [f"{prefix}blocks.{i}.mlp.weight" for i in range(3)]
    path = write_safetensors(tmp_path / "anima.safetensors", {k: torch.zeros(1) for k in keys})
    assert _count_anima_blocks(path) == 3


def test_count_anima_blocks_requires_block_keys(tmp_path):
    path = write_safetensors(tmp_path / "other.safetensors", {"attn.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match=r"could not find any 'blocks\.\*' keys"):
        _count_anima_blocks(path)


# ------------------------------------------------------------------ timestep embedding


def test_timestep_embedding_matches_reference():
    """The shared function reproduces the reference cos/sin grid scaled by 1000."""
    from thenoise.utils.timestep import timestep_embedding

    t = torch.linspace(1, 0, 4)
    emb = timestep_embedding(t, 16)
    assert emb.shape == (4, 16)

    half = 8
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    args = t.float()[:, None] * 1000 * freqs[None]
    ref = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    assert torch.allclose(emb, ref)


def test_timestep_embedding_shape_dtype_and_odd_padding():
    from thenoise.utils.timestep import timestep_embedding

    t = torch.linspace(1, 0, 6).reshape(2, 3)  # leading dims preserved, as Anima passes
    assert timestep_embedding(t, 16).shape == (2, 3, 16)
    assert timestep_embedding(torch.tensor([0.5]), 7).shape == (1, 7)
    assert timestep_embedding(torch.tensor([0.5]), 7)[0, -1] == 0.0  # odd dim is zero-padded
    assert timestep_embedding(torch.tensor([0.5], dtype=torch.bfloat16), 16).dtype == torch.bfloat16


# -------------------------------------------------------------- anima video rope


def test_split_half_rope_3d_patches_the_grid_and_builds_cos_sin():
    """The Anima builder patches the raw latent grid and emits ``(cos, sin)``."""
    from thenoise.utils.rope import split_half_rope_3d

    builder = split_half_rope_3d(head_dim=64, patch_spatial=2, patch_temporal=1)
    raw = (1, 16, 2, 8, 12)  # B, C, T, H, W
    cos, sin = builder(raw, "cpu")
    # Patched grid: T'=2, H'=4, W'=6 -> 48 tokens.
    assert cos.shape == (2 * 4 * 6, 1, 1, 64)
    assert sin.shape == cos.shape
    assert torch.isfinite(cos).all()


def test_apply_rope_split_half_is_orthogonal():
    """Split-half RoPE preserves the norm of each head vector."""
    from thenoise.utils.rope import apply_rope_split_half, split_half_rope_3d

    torch.manual_seed(0)
    builder = split_half_rope_3d(head_dim=16, patch_spatial=1, patch_temporal=1)
    cos, sin = builder((1, 16, 1, 8, 8), "cpu")  # 8x8 grid -> 64 tokens
    q = torch.randn(2, 3, 64, 16)  # [B, H, L, D]
    out = apply_rope_split_half(q, cos, sin)
    assert out.shape == q.shape
    # RoPE is a rotation: per-token-per-head norms are preserved.
    assert torch.allclose(q.norm(dim=-1), out.norm(dim=-1), atol=1e-6)


# ------------------------------------------------------------------ QK-norm key map


@pytest.mark.parametrize(
    "key,legacy,expected",
    [
        # Anima/Z-Image checkpoints store q_norm/k_norm on the attention module;
        # the shared ``QKNorm`` stores query_norm/key_norm.
        ("blocks.0.self_attn.q_norm.weight", ("q_norm", "k_norm"),
         "blocks.0.self_attn.qk_norm.query_norm.weight"),
        ("layers.0.attention.k_norm.weight", ("q_norm", "k_norm"),
         "layers.0.attention.qk_norm.key_norm.weight"),
        # Krea 2 spells them qnorm/knorm, on ``scale`` (renamed to ``weight`` by
        # the loader's value map).
        ("blocks.0.attn.qnorm.scale", ("qnorm", "knorm"),
         "blocks.0.attn.qk_norm.query_norm.scale"),
        ("blocks.0.attn.knorm.scale", ("qnorm", "knorm"),
         "blocks.0.attn.qk_norm.key_norm.scale"),
    ],
)
def test_qk_norm_key_map_maps_the_legacy_spellings(key, legacy, expected):
    from thenoise.utils.qk_norm import qk_norm_key_map

    assert qk_norm_key_map(key, *legacy) == expected




# ------------------------------------------------------------------ sequence padding


def test_pad_len_to_multiple_rounds_up():
    from thenoise.utils.sequence import pad_len_to_multiple

    assert pad_len_to_multiple(0, 32) == 0
    assert pad_len_to_multiple(1, 32) == 32
    assert pad_len_to_multiple(32, 32) == 32
    assert pad_len_to_multiple(33, 32) == 64
    assert pad_len_to_multiple(255, 256) == 256


def test_pad_to_batch_right_pads_to_the_max_and_keeps_positions_in_lockstep():
    from thenoise.utils.sequence import pad_to_batch

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    b = torch.tensor([[5.0, 6.0]])              # (1, 2)
    pos_a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    pos_b = torch.tensor([[0.0, 0.0, 0.0]])

    out, positions, seqlens = pad_to_batch([a, b])
    assert out.shape == (2, 2, 2)
    assert torch.equal(out[0], a)
    assert torch.equal(out[1, 0], b[0])
    assert torch.equal(out[1, 1], torch.zeros(2))
    assert seqlens == [2, 1]
    assert positions is None

    out, pos, _ = pad_to_batch([a, b], [pos_a, pos_b])
    assert pos.shape == (2, 2, 3)
    assert torch.equal(pos[0], pos_a)
    assert torch.equal(pos[1, 0], pos_b[0])
    assert torch.equal(pos[1, 1], torch.zeros(3))


def test_make_key_padding_mask_is_none_on_uniform_lengths():
    from thenoise.utils.sequence import make_key_padding_mask

    assert make_key_padding_mask([3, 3], "cpu") is None
    mask = make_key_padding_mask([3, 3], "cpu", always=True)
    assert mask.shape == (2, 3)
    assert mask.tolist() == [[True, True, True], [True, True, True]]


def test_make_key_padding_mask_marks_valid_prefix():
    from thenoise.utils.sequence import make_key_padding_mask

    mask = make_key_padding_mask([3, 1], "cpu")
    assert mask.shape == (2, 3)
    assert mask.tolist() == [[True, True, True], [True, False, False]]


# --------------------------------------------------------- alignment pad-fill / mask


def test_pad_to_length_fills_with_zeros_by_default():
    from thenoise.utils.sequence import pad_to_length

    seq = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = pad_to_length([seq], [4])[0]
    assert out.shape == (4, 2)
    assert torch.equal(out[:2], seq)
    assert torch.equal(out[2:], torch.zeros(2, 2))


def test_pad_to_length_tiles_the_pad_token():
    """The learned-pad path: every new slot is the SAME embedding."""
    from thenoise.utils.sequence import pad_to_length

    seq = torch.ones(3, 2)
    pad_token = torch.full((1, 2), 9.0)
    out = pad_to_length([seq], [6], pad_token=pad_token)[0]
    assert torch.equal(out[:3], seq)
    assert torch.equal(out[3:], torch.full((3, 2), 9.0))


def test_pad_to_length_is_a_noop_when_already_long_enough():
    """An aligned sequence is passed through as the SAME tensor (no copy)."""
    from thenoise.utils.sequence import pad_to_length

    seq = torch.ones(3, 2)
    assert pad_to_length([seq], [3])[0] is seq


def test_pad_to_length_refuses_to_trim():
    from thenoise.utils.sequence import pad_to_length

    with pytest.raises(ValueError, match="cannot pad"):
        pad_to_length([torch.ones(5, 2)], [4])


def test_alignment_padding_mask_holes_land_on_the_pads():
    """Two segments (image then caption): each keeps its own valid prefix."""
    from thenoise.utils.sequence import alignment_padding_mask

    mask = alignment_padding_mask([[(4, 8), (3, 5)]], "cpu")
    assert mask.shape == (1, 13)
    assert mask[0].tolist() == (
        [True] * 4 + [False] * 4 + [True] * 3 + [False] * 2
    )


def test_alignment_padding_mask_masks_the_batch_padding_too():
    """An item shorter than the batch max loses its tail as well as its pads."""
    from thenoise.utils.sequence import alignment_padding_mask

    mask = alignment_padding_mask([[(4, 8)], [(1, 2)]], "cpu")
    assert mask.shape == (2, 8)
    assert mask[0].tolist() == [True] * 4 + [False] * 4
    assert mask[1].tolist() == [True] + [False] * 7


def test_alignment_padding_mask_uniform_and_unpadded_is_the_fast_path():
    """Nothing masked -> ``None``, so the caller keeps the mask-free SDPA path."""
    from thenoise.utils.sequence import alignment_padding_mask

    assert alignment_padding_mask([[(8, 8)]], "cpu") is None
    assert alignment_padding_mask([[(4, 4)], [(4, 4)]], "cpu") is None
    # Two segments both fully attended: still nothing to mask.
    assert alignment_padding_mask([[(4, 4), (4, 4)]], "cpu") is None
    # ``always`` forces a real tensor for callers that consume it unconditionally.
    assert alignment_padding_mask([[(8, 8)]], "cpu", always=True).shape == (1, 8)


# ------------------------------------------------------------------ position ids


def test_grid_positions_is_row_major_with_per_axis_start():
    from thenoise.utils.positions import grid_positions

    # 2x3 grid -> 6 tokens, columns (h, w), row-major.
    pos = grid_positions([2, 3], dtype=torch.float32)
    assert pos.shape == (6, 2)
    assert pos[:, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert pos[:, 1].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]

    assert grid_positions([2, 3], start=[5, 0], dtype=torch.float32)[:, 0].tolist() == [
        5.0, 5.0, 5.0, 6.0, 6.0, 6.0
    ]


def test_grid_positions_centered_matches_qwen_scale_rope():
    """Qwen-Image h/w use ``r - ceil(size / 2)`` (the ``scale_rope`` convention)."""
    from thenoise.utils.positions import grid_positions

    pos = grid_positions([1, 4, 6], centered=[False, True, True], dtype=torch.float32)
    # h: 4 -> [-2, -1, 0, 1]; w: 6 -> [-3, -2, -1, 0, 1, 2]; t stays 0.
    h = pos[:, 1]
    assert sorted(h.unique().tolist()) == [-2.0, -1.0, 0.0, 1.0]
    w = pos[:, 2]
    assert sorted(w.unique().tolist()) == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
    assert (pos[:, 0] == 0).all()


def test_grid_positions_preserves_int_dtype():
    from thenoise.utils.positions import grid_positions

    assert grid_positions([2, 3], dtype=torch.int32).dtype == torch.int32


def test_broadcast_positions_repeats_a_single_index():
    from thenoise.utils.positions import broadcast_positions

    pos = broadcast_positions(4, 3, offset=7)
    assert pos.shape == (4, 3)
    assert torch.equal(pos[:, 0], pos[:, 1])
    assert torch.equal(pos[:, 1], pos[:, 2])
    assert pos[0].tolist() == [7.0, 7.0, 7.0]
    assert pos[3].tolist() == [10.0, 10.0, 10.0]


# ------------------------------------------------------------- chat-template drop


def test_drop_idx_skips_the_system_turn_and_the_user_header():
    """The user content starts 3 tokens past the SECOND chat-start marker."""
    ids = torch.tensor([151644, 1, 2, 151644, 9, 9, 9, 9])
    assert compute_drop_idx(ids) == 6  # 3 markers + ``user`` + ``\n`` -> 2 tokens kept


def test_drop_idx_counts_markers_rather_than_assuming_a_prefix_length():
    """The prefix length depends on how the system prompt tokenizes, so it is counted:
    Qwen-Image's t2i template lands on 34 (``QWEN_VL_DROP_IDX``), the edit template's
    longer system prompt on 64, and only the marker count is common to both.
    """
    t2i = torch.tensor([151644] + [5] * 30 + [151644] + [6] * 10)
    edit = torch.tensor([151644] + [5] * 60 + [151644] + [6] * 10)
    assert compute_drop_idx(t2i) == QWEN_VL_DROP_IDX
    assert compute_drop_idx(edit) == 64
    assert compute_drop_idx(torch.tensor([151644, 1, 2, 3])) == 0  # no user turn at all
