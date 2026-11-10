"""The packed token sequence one H3-Image generation runs over.

The DiT is single-stream: the prompt, any reference pictures and the target lattice
share one row sequence, and every row carries a ``(t, h, w)`` rope coordinate. Three
properties of the release fix how those coordinates are laid out:

  * **the axes are on different scales.** Spatial coordinates are area-normalised
    (:func:`_axis`): a picture's rows span the same stretch of coordinate space
    whatever their aspect ratio, because the model was trained at any size rather
    than at a fixed grid. The time axis is measured in the same units, so a
    reference sits exactly one unit ahead of the target's origin and the target's
    single frame at ``1.0`` — which is what separates a reference from the picture
    being denoised, since both are fed as clean-looking rows (see the timestep tags
    in :mod:`thenoise.dit.h3_image.models`).
  * **a coordinate names a patch, not a pixel.** One row is a 2x2 block of latent
    cells, so the grid is ``latent_h // 2`` by ``latent_w // 2`` and the latent has
    to be rounded up to a whole patch before it is packed.
  * **text rows are positions.** They take ``t = 0 … L-1`` on the time axis and zero
    on both spatial ones, which is why the rope has to be partial: a prompt longer
    than the target would otherwise dominate the spatial axes.

The packed order is ``[text | target | references]``. Attention over the packed
sequence is unmasked and every row carries its own coordinates, so the order is free
-- and it is chosen to leave the reference rows as a suffix: they are the rows whose
inputs do not change between steps, which is what a run's K/V cache can freeze (see
:mod:`thenoise.dit.kvcache`).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch

#: One reference picture's latent shape, the only kind the image model accepts.
RefShape = Tuple[int, int]

#: Latent cells per DiT token: one frame, a 2x2 spatial block. One row of the packed
#: sequence is one patch, so a latent is rounded up to a whole tile before packing.
PATCH_SIZE = (1, 2, 2)

#: The three segment kinds of a packed image sequence, and the adaLN modality tag
#: each one selects: the visual tag 0 for every row that came out of the codec, the
#: text tag 1 for the prompt. The tag picks a row of the modulation table, so it is
#: how a reference stops being treated as noise to be denoised.
SEG_TAG = {"text": 1, "ref": 0, "target": 0}

#: Rows of the modulation table each timestep owns: the tag space is the video model's,
#: whose audio stream takes tag 2. This model writes 0 and 1, and spends the third row.
MODALITIES = 3


def _axis(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """Coordinates of one axis of ``dim`` latent cells, area-normalised.

    ``sqrt_area`` is the lattice's side length, so ``dim / sqrt_area`` is how much
    narrower this axis is than a square of the same area; the row centres then run
    from ``(1 - ratio) / 2`` to ``(1 + ratio) / 2`` of that side, in units of 1/32
    of it.
    """
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def frame_grid(latent_h: int, latent_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The ``(h, w)`` coordinates of one frame's patch rows, and the width axis.

    Row-major over the ``latent_h // 2`` by ``latent_w // 2`` patch grid, matching
    the order :func:`thenoise.dit.h3_image.models.patchify` produces.
    """
    area = math.sqrt(latent_h * latent_w)
    h_axis = _axis(latent_h, PATCH_SIZE[1], area)
    w_axis = _axis(latent_w, PATCH_SIZE[1], area)
    hh, ww = torch.meshgrid(h_axis, w_axis, indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w_axis


class Layout:
    """Segment spans, rope coordinates and row indices for one shape.

    ``latent_h`` / ``latent_w`` are the *patch-padded* latent dimensions (see
    :func:`pad_to_patch`), and ``refs`` the latent shape of each reference, in
    packing order.
    """

    def __init__(self, text_len: int, latent_h: int, latent_w: int,
                 refs: Sequence[RefShape] = ()) -> None:
        frame, _ = frame_grid(latent_h, latent_w)

        segments: List[Tuple[int, int, str]] = [(0, text_len, "text")]
        # [n, 3] float64 per segment: (t, h, w)
        positions = [self._text_positions(text_len)]
        offset = text_len

        # A reference occupies one unit of the time axis, so the target's frame sits
        # behind all of them in the coordinates -- even though it is packed ahead of
        # them, which is what leaves the references a cacheable suffix.
        target_rows = frame.shape[0]
        grid = torch.empty(target_rows, 3, dtype=torch.float64)
        grid[:, 0] = text_len + len(refs)
        grid[:, 1:] = frame
        segments.append((offset, offset + target_rows, "target"))
        positions.append(grid)
        offset += target_rows

        for i, (ref_h, ref_w) in enumerate(refs):
            ref_frame, _ = frame_grid(ref_h, ref_w)
            grid = torch.empty(ref_frame.shape[0], 3, dtype=torch.float64)
            grid[:, 0] = float(text_len + i)
            grid[:, 1:] = ref_frame
            segments.append((offset, offset + ref_frame.shape[0], "ref"))
            positions.append(grid)
            offset += ref_frame.shape[0]

        self.text_len = text_len
        self.latent_h = latent_h
        self.latent_w = latent_w
        self.refs = tuple(refs)
        self.seq_len = offset
        self.segments = segments
        self.position_ids = torch.cat(positions)
        #: The target's own span, which is the only one the output head reads.
        self.target_span = segments[1][0], segments[1][1]
        #: Row count of the codec rows the embedding consumes: target then references.
        self.codec_rows = offset - text_len

    @staticmethod
    def _text_positions(text_len: int) -> torch.Tensor:
        grid = torch.zeros(text_len, 3, dtype=torch.float64)
        grid[:, 0] = torch.arange(text_len, dtype=torch.float64)
        return grid

    def modulation_rows(self, tags: Optional[torch.Tensor] = None
                        ) -> List[Tuple[int, int, int]]:
        """``(start, stop, modulation_row)`` covering the sequence, contiguous.

        A run's modulation embedding has two timestep rows: the step's own ``t``, which
        the prompt and the target follow, and the
        :data:`~thenoise.dit.h3_image.models.COND_TIMESTEP` the reference rows are
        pinned at. Those are not the same row except at the very end of a run, and
        reading the wrong one conditions a reference as picture-to-be-cleaned. The table
        is indexed by ``timestep_row * MODALITIES + tag``, so a segment's row is its
        base plus the tag. The prompt is the one segment that mixes tags: the vision
        placeholders inside it carry the visual tag, so it is split into one segment per
        run of equal tags. ``tags`` is the text encoder's per-token tag vector, or
        ``None`` for a prompt with no images in it.
        """
        timestep_rows = {"text": 0, "target": 0, "ref": 1}
        segments = []
        for start, stop, kind in self.segments:
            base = timestep_rows[kind] * MODALITIES
            if kind == "text" and tags is not None:
                segments.extend(_tag_runs(start, stop, base, tags))
            else:
                segments.append((start, stop, base + SEG_TAG[kind]))
        return segments


def _tag_runs(start: int, stop: int, row_base: int, tags: torch.Tensor
              ) -> List[Tuple[int, int, int]]:
    """Split ``[start, stop)`` into runs of equal tag, as modulation segments."""
    values = tags.view(-1).tolist()
    runs = []
    run_start = 0
    for i in range(1, stop - start + 1):
        if i == stop - start or values[i] != values[run_start]:
            runs.append((start + run_start, start + i, row_base + int(values[run_start])))
            run_start = i
    return runs


def pad_to_patch(latent: torch.Tensor, patch_size: Tuple[int, int, int]
                 ) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Repeat-pad a ``[B, C, H, W]`` latent onto a whole ``patch_size`` tile.

    Returns the padded latent and the original ``(H, W)``, which the output is cut
    back to: a patch's leftover cells are predicted and dropped, never fed as noise.
    """
    _, _, h_full, w_full = latent.shape
    _, ph, pw = patch_size
    pad_h, pad_w = (-h_full) % ph, (-w_full) % pw
    if not (pad_h or pad_w):
        return latent, (h_full, w_full)
    return torch.nn.functional.pad(latent, (0, pad_w, 0, pad_h), mode="replicate"), (h_full, w_full)


__all__ = [
    "PATCH_SIZE",
    "MODALITIES",
    "SEG_TAG",
    "Layout",
    "RefShape",
    "frame_grid",
    "pad_to_patch",
]
