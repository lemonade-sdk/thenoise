"""Model-side helpers that are pure functions of their inputs.

Krea 2's resolution-aware timestep grid, latent patchify/position builder and
text-token compaction; Anima's checkpoint-header block counter and key map. No
weights and no device: these are the pieces whose silent failure would show up
as a wrong schedule or a mis-shaped token stream.
"""
from __future__ import annotations

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


