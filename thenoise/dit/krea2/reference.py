"""Krea 2 reference-latent ("edit") conditioning helpers.

Ported from ``comfyui-krea2edit``'s ``krea2_edit_forward`` math (training-matched
``fit`` geometry + 3-axis RoPE frame indexing), without the ComfyUI plumbing.

The Krea 2 DiT is single-stream and image-first in this repo, so the reference
tokens are prepended *before* the (noisy) target tokens and the output is sliced
back to the target only. Reference tokens are distinguished from the target purely
by the RoPE frame index (source=1..N, target=0), with the h/w axes aligned to the
target grid (stride-1 at a centered offset, matching the ``fit`` geometry v1.2 was
trained with).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def reference_positions(
    bs: int,
    frame: int,
    gh: int,
    gw: int,
    th: int,
    tw: int,
    device=None,
) -> Tensor:
    """``[1, gh*gw, 3]`` RoPE grid for one reference at ``frame``.

    ``gh``/``gw`` are the reference token-grid dims (latent H//patch, W//patch) and
    ``th``/``tw`` the target's. Uses the ``fit`` geometry's stride-1 *centered*
    (fractional) offset. Requires ``gh <= th`` and ``gw <= tw`` (guaranteed by
    ``fit_reference``).
    """
    off_h = max(0.0, (th - gh) / 2)
    off_w = max(0.0, (tw - gw) / 2)
    ids = torch.zeros(gh, gw, 3, device=device, dtype=torch.float32)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(gh, device=device, dtype=torch.float32)[:, None] + off_h
    ids[..., 2] = torch.arange(gw, device=device, dtype=torch.float32)[None, :] + off_w
    return ids.reshape(1, gh * gw, 3).repeat(bs, 1, 1)


def fit_reference(latent: Tensor, th: int, tw: int, patch: int) -> Tensor:
    """Fit a reference latent to the target token grid ``(th, tw)``, training-matched.

    Center-crops to the target aspect ratio then resizes (the ``_fit_src`` /
    ``fit`` geometry). A no-op when the reference is already at the target grid
    (the common case: the pipeline cover-crops the source to the target first).
    """
    tgt_h, tgt_w = th * patch, tw * patch
    H, W = latent.shape[-2:]
    if (H, W) == (tgt_h, tgt_w):
        return latent
    s = max(tgt_h / H, tgt_w / W)
    ch = min(H, int(round(tgt_h / s)))
    cw = min(W, int(round(tgt_w / s)))
    y0, x0 = (H - ch) // 2, (W - cw) // 2
    latent = latent[..., y0:y0 + ch, x0:x0 + cw]
    return F.interpolate(latent.float(), size=(tgt_h, tgt_w), mode="bilinear")


def pack_reference(
    latent: Tensor,
    th: int,
    tw: int,
    patch: int,
    ref_index: int,
    device=None,
) -> tuple[Tensor, Tensor]:
    """Canonical reference latent -> ``(ref_tokens, ref_pos)`` at the target grid.

    ``ref_tokens`` is ``[B, gh*gw, C*patch^2]`` (the patchified reference) and
    ``ref_pos`` its ``[B, gh*gw, 3]`` RoPE grid (frame = ``ref_index``, centered
    offset). ``th``/``tw`` are the target token-grid dims.
    """
    fitted = fit_reference(latent, th, tw, patch)
    ref_tokens = rearrange(fitted, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    gh, gw = fitted.shape[-2] // patch, fitted.shape[-1] // patch
    ref_pos = reference_positions(1, ref_index, gh, gw, th, tw, device)
    return ref_tokens, ref_pos


def concat_reference(ref_tokens: list[Tensor], target_tokens: Tensor) -> Tensor:
    """Prepend the reference tokens to the (noisy) target tokens."""
    if not ref_tokens:
        return target_tokens
    return torch.cat([*ref_tokens, target_tokens], dim=1)


def slice_reference_output(out: Tensor, ref_len: int, target_len: int) -> Tensor:
    """Drop the leading reference tokens; keep only the target tokens."""
    return out[:, ref_len : ref_len + target_len]


__all__ = [
    "reference_positions",
    "fit_reference",
    "pack_reference",
    "concat_reference",
    "slice_reference_output",
]
