"""Model-side helpers that are pure functions of their inputs.

Krea 2's resolution-aware timestep grid, latent patchify/position builder and
text-token compaction; Anima's checkpoint-header block counter and key map. No
weights and no device: these are the pieces whose silent failure would show up
as a wrong schedule or a mis-shaped token stream.
"""
from __future__ import annotations

import math
import pytest
import torch

from conftest import write_safetensors
from thenoise.dit.anima.utils import _count_anima_blocks
from thenoise.dit.krea2.sampling import (
    encode_prompts,
    gather_valid_text,
    prepare,
    timesteps,
)


# ------------------------------------------------------------------ krea2 grid


def test_krea2_timesteps_shift_with_the_token_count_when_mu_is_interpolated():
    small = timesteps(256, 8, 256, 6400, y1=0.5, y2=1.15)
    large = timesteps(4096, 8, 256, 6400, y1=0.5, y2=1.15)
    assert len(small) == len(large) == 9
    assert small != large
    assert small[0] == 1.0 and small[-1] == 0.0


def test_krea2_timesteps_are_pinned_when_mu_is_given():
    """The distilled checkpoint passes an explicit mu -> no resolution shift."""
    pinned = [timesteps(seq, 8, 256, 6400, mu=1.15) for seq in (256, 1024, 4096)]
    assert pinned[0] == pinned[1] == pinned[2]


def test_krea2_timesteps_are_monotonic():
    ts = timesteps(1024, 8, 256, 6400, mu=1.15)
    assert all(a > b for a, b in zip(ts, ts[1:]))


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


def test_krea2_encode_prompts_defaults_negatives_to_blank():
    """With cfg and no negatives, the encoder is run on a batch of empty prompts."""
    enc = _Recorder()
    txt, txtmask, untxt, untxtmask = encode_prompts(enc, ["a", "b"], cfg=True)
    assert enc.calls == [["a", "b"], ["", ""]]
    assert untxt is not None and untxtmask is not None
    assert untxt.shape == txt.shape
    assert untxtmask.shape == txtmask.shape


def test_krea2_encode_prompts_uses_given_negatives():
    enc = _Recorder()
    txt, txtmask, untxt, untxtmask = encode_prompts(
        enc, ["a"], negative_prompts=["bad"], cfg=True
    )
    assert enc.calls == [["a"], ["bad"]]
    assert untxt is not None


def test_krea2_encode_prompts_skips_unconditional_without_cfg():
    enc = _Recorder()
    txt, txtmask, untxt, untxtmask = encode_prompts(enc, ["a"], cfg=False)
    assert enc.calls == [["a"]]
    assert untxt is None and untxtmask is None


# ------------------------------------------------------------------ anima utils


def test_count_anima_blocks_reads_the_header(tmp_path):
    keys = [f"blocks.{i}.mlp.weight" for i in range(28)] + ["x_embedder.weight"]
    path = write_safetensors(tmp_path / "anima.safetensors", {k: torch.zeros(1) for k in keys})
    assert _count_anima_blocks(path) == 28


def test_count_anima_blocks_ignores_the_wrapper_prefix(tmp_path):
    """Raw (``net.``) and repackaged checkpoints must count identically."""
    for prefix in ("", "net.", "model.diffusion_model."):
        keys = [f"{prefix}blocks.{i}.mlp.weight" for i in range(3)]
        path = write_safetensors(tmp_path / f"anima-{prefix or 'bare'}.safetensors",
                                 {k: torch.zeros(1) for k in keys})
        assert _count_anima_blocks(path) == 3, prefix


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


def test_timestep_embedding_preserves_leading_dims():
    from thenoise.utils.timestep import timestep_embedding

    t = torch.linspace(1, 0, 6).reshape(2, 3)  # (B, T), as Anima passes
    emb = timestep_embedding(t, 16)
    assert emb.shape == (2, 3, 16)


def test_timestep_embedding_pads_odd_dim():
    from thenoise.utils.timestep import timestep_embedding

    emb = timestep_embedding(torch.tensor([0.5]), 7)
    assert emb.shape == (1, 7)
    assert emb[0, -1] == 0.0


def test_timestep_embedding_casts_to_input_dtype():
    from thenoise.utils.timestep import timestep_embedding

    t = torch.tensor([0.5], dtype=torch.bfloat16)
    emb = timestep_embedding(t, 16)
    assert emb.dtype == torch.bfloat16


def test_timestep_embedding_time_factor_one_is_the_unscaled_grid():
    """time_factor=1 is the fallback (no 1000x scale); used if Anima is reverted."""
    from thenoise.utils.timestep import timestep_embedding

    t = torch.tensor([1.0, 0.5])
    emb = timestep_embedding(t, 16, time_factor=1.0)
    half = 8
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    args = t.float()[:, None] * freqs[None]
    ref = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    assert torch.allclose(emb, ref)


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


def test_qk_norm_key_map_maps_anima_zimage_legacy_keys():
    """The shared ``QKNorm`` stores ``query_norm``/``key_norm``; Anima/Z-Image
    checkpoints store ``q_norm``/``k_norm`` on the attention module."""
    from thenoise.utils.qk_norm import qk_norm_key_map

    assert (
        qk_norm_key_map("blocks.0.self_attn.q_norm.weight")
        == "blocks.0.self_attn.qk_norm.query_norm.weight"
    )
    assert (
        qk_norm_key_map("blocks.0.self_attn.k_norm.weight")
        == "blocks.0.self_attn.qk_norm.key_norm.weight"
    )
    assert (
        qk_norm_key_map("layers.0.attention.q_norm.weight")
        == "layers.0.attention.qk_norm.query_norm.weight"
    )


def test_qk_norm_key_map_maps_krea2_legacy_keys():
    """Krea 2 checkpoints store ``qnorm``/``knorm`` (the ``scale``->``weight``
    rename is handled by the loader's value map)."""
    from thenoise.utils.qk_norm import qk_norm_key_map

    assert (
        qk_norm_key_map("blocks.0.attn.qnorm.scale", "qnorm", "knorm")
        == "blocks.0.attn.qk_norm.query_norm.scale"
    )
    assert (
        qk_norm_key_map("blocks.0.attn.knorm.scale", "qnorm", "knorm")
        == "blocks.0.attn.qk_norm.key_norm.scale"
    )




# ------------------------------------------------------------------ sequence padding


def test_pad_len_to_multiple_rounds_up():
    from thenoise.utils.sequence import pad_len_to_multiple

    assert pad_len_to_multiple(0, 32) == 0
    assert pad_len_to_multiple(1, 32) == 32
    assert pad_len_to_multiple(32, 32) == 32
    assert pad_len_to_multiple(33, 32) == 64
    assert pad_len_to_multiple(255, 256) == 256


def test_pad_to_batch_right_pads_to_the_max():
    from thenoise.utils.sequence import pad_to_batch

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    b = torch.tensor([[5.0, 6.0]])              # (1, 2)
    out, positions, seqlens = pad_to_batch([a, b])
    assert out.shape == (2, 2, 2)
    assert torch.equal(out[0], a)
    assert torch.equal(out[1, 0], b[0])
    assert torch.equal(out[1, 1], torch.zeros(2))
    assert seqlens == [2, 1]
    assert positions is None


def test_pad_to_batch_pads_positions_in_lockstep():
    from thenoise.utils.sequence import pad_to_batch

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0]])
    pos_a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    pos_b = torch.tensor([[0.0, 0.0, 0.0]])
    out, pos, seqlens = pad_to_batch([a, b], [pos_a, pos_b])
    assert out.shape == (2, 2, 2)
    assert pos.shape == (2, 2, 3)
    assert torch.equal(pos[0], pos_a)
    assert torch.equal(pos[1, 0], pos_b[0])
    assert torch.equal(pos[1, 1], torch.zeros(3))
    assert seqlens == [2, 1]


def test_pad_to_batch_replaces_pad_positions_with_pad_token():
    """Z-Image path: pad positions (True in replace_mask) are swapped for a token."""
    from thenoise.utils.sequence import pad_to_batch

    feat = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    replace_mask = torch.tensor([False, False, True])  # True = pad
    pad_token = torch.tensor([[99.0, 99.0]])
    out, _, seqlens = pad_to_batch([feat], [feat], pad_token=pad_token, replace_mask=[replace_mask])
    assert torch.equal(out[0, 0], feat[0])
    assert torch.equal(out[0, 1], feat[1])
    assert torch.equal(out[0, 2], pad_token[0])
    assert seqlens == [3]


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


def test_make_key_padding_mask_matches_zimage_reference():
    """Z-Image's original ``_prepare_sequence`` built the same valid mask."""
    from thenoise.utils.sequence import make_key_padding_mask

    item_seqlens = [48, 64]  # padded-to-32 lengths
    mask = make_key_padding_mask(item_seqlens, "cpu")
    assert mask[0].sum().item() == 48
    assert mask[1].sum().item() == 64


# ------------------------------------------------------------------ position ids


def test_grid_positions_is_row_major():
    from thenoise.utils.positions import grid_positions

    # 2x3 grid -> 6 tokens, columns (h, w).
    pos = grid_positions([2, 3], dtype=torch.float32)
    assert pos.shape == (6, 2)
    # Row-major: (0,0), (0,1), (0,2), (1,0), ...
    assert pos[:, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert pos[:, 1].tolist() == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]


def test_grid_positions_applies_per_axis_start():
    from thenoise.utils.positions import grid_positions

    pos = grid_positions([2, 3], start=[5, 0], dtype=torch.float32)
    assert pos[:, 0].tolist() == [5.0, 5.0, 5.0, 6.0, 6.0, 6.0]


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

    pos = grid_positions([2, 3], dtype=torch.int32)
    assert pos.dtype == torch.int32
    assert pos.shape == (6, 2)


def test_grid_from_axes_matches_flux2_cartesian_order():
    """Flux.2's ``cartesian_prod(t, h, w, l)`` lexicographic order."""
    import torch
    from thenoise.utils.positions import grid_from_axes

    t = torch.arange(1)
    h = torch.arange(2)
    w = torch.arange(3)
    l = torch.arange(1)
    pos = grid_from_axes([t, h, w, l])
    ref = torch.cartesian_prod(t, h, w, l)
    assert torch.equal(pos, ref)


def test_broadcast_positions_repeats_a_single_index():
    from thenoise.utils.positions import broadcast_positions

    pos = broadcast_positions(4, 3, offset=7)
    assert pos.shape == (4, 3)
    assert torch.equal(pos[:, 0], pos[:, 1])
    assert torch.equal(pos[:, 1], pos[:, 2])
    assert pos[0].tolist() == [7.0, 7.0, 7.0]
    assert pos[3].tolist() == [10.0, 10.0, 10.0]
