"""Generic reference-latent helpers for instruction-based editing.

Shared by Flux2 Klein and Qwen Image Edit (and any future reference-latent
model). Both use the Flux-family mechanism: the input image is encoded to a
latent, packed into DiT tokens + position ids, concatenated to the generated
image tokens, and sliced back off the output.

  * ``concat_reference``       — append reference tokens+ids to image tokens+ids.
  * ``slice_reference_output`` — drop the trailing reference tokens from the DiT output.
"""
from __future__ import annotations

import torch

__all__ = ["concat_reference", "slice_reference_output"]


def concat_reference(
    img: torch.Tensor,
    img_ids: torch.Tensor,
    ref_tokens: torch.Tensor | None,
    ref_ids: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append reference tokens+ids to the image tokens+ids.

    The reference stream is concatenated *after* the image tokens (``torch.cat([img, ref])``),
    and ``slice_reference_output`` drops the trailing refs. Positions stay distinct via the
    t-axis, so ordering does not affect attention.

    ``ref_tokens``/``ref_ids`` of ``None`` (plain generation) return
    ``img``/``img_ids`` unchanged.
    """
    if ref_tokens is None or ref_ids is None:
        return img, img_ids
    return torch.cat([img, ref_tokens], dim=1), torch.cat([img_ids, ref_ids], dim=1)


def slice_reference_output(out: torch.Tensor, num_img_tokens: int) -> torch.Tensor:
    """Drop the trailing reference tokens from a DiT output.

    Reference tokens are concatenated *after* the image tokens.
    """
    return out[:, :num_img_tokens]
