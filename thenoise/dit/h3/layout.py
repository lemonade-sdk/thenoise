"""The packed token sequence one H3 run reads over.

The prompt, the conditioned pictures and the target lattices share one row sequence, every
row carrying a ``(t, h, w)`` rope coordinate. Spatial coordinates are area-normalised —
the model was trained at any size, not at a fixed grid — in units of 1/32 of the lattice's
side, and one row is a 2x2 block of latent cells rather than a cell.

Attention over the sequence is unmasked, so the packing order is free and only two things
decide what a row does: where it sits on the time axis, and which modulation row it reads.
That makes the order a policy, one per variant — :meth:`Layout.image` puts the references
last so a run's K/V cache can freeze them as a suffix (:mod:`thenoise.dit.kvcache`), while
:meth:`Layout.av` keeps the released order, conditions first.
"""
from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

#: One reference picture's latent shape, the only kind the image model accepts.
RefShape = Tuple[int, int]

class Keyframe(NamedTuple):
    """A condition pinned to a pixel frame of the clip, on the target's own grid.

    ``latent_t`` latent frames, read at that pixel frame's time coordinate: one for a
    keyframe, several for a guide clip.
    """

    frame: int
    latent_t: int = 1


class Reference(NamedTuple):
    """A condition the clip is asked to resemble, which is not a frame of the clip.

    A reference is packed ahead of the target timeline and keeps its own spatial grid --
    the released pipeline sizes it to the generation's area or to a 2048px short edge, not
    to the canvas -- so it occupies rope time units of its own and moves the whole target
    ahead of them.
    """

    kind: str
    latent_t: int = 0
    latent_h: int = 0
    latent_w: int = 0


def ref_span(ref: Reference) -> float:
    """The rope time units one reference occupies ahead of the target streams."""
    return 1.0 if ref.kind == "image" else sum(time_spans(ref.latent_t))

#: Latent cells per DiT token: one frame, a 2x2 spatial block.
PATCH_SIZE = (1, 2, 2)

#: The codec's decode plan: five latent frames cover seventeen pixel frames. The video VAE
#: derives the same schedule from its own geometry and a test ties the two, since sharing one
#: constant would mean the DiT importing the module that carries the codec.
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

#: Rope time units per pixel frame. The time axis runs in audio latent frames — forty a
#: second, one third of a pixel frame at the 24 fps the model was trained on.
FRAME_RESCALE = 5.0 / 3.0

#: A conditioned row's timestep label: its noise is mixed to match it, and its rows are
#: read at it instead of the step's, so it is never treated as something to denoise.
COND_TIMESTEP = 0.999

#: The audio stream's own label. Nothing conditions on audio yet, so no row takes it.
AUDIO_COND_TIMESTEP = 1.0

#: The adaLN modality tag of each segment kind: 0 for anything that came out of a codec
#: lattice, 1 for the prompt, 2 for audio. The tag picks a row of the modulation table.
SEG_TAG = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0,
           "cond_audio": 2, "ref_audio": 2}

#: Which embedding feeds a segment: the video lattice, the audio stream, or the prompt.
SEG_EMBED = {"text": None, "video": "video", "cond": "video", "ref_img": "video",
             "audio": "audio", "cond_audio": "audio", "ref_audio": "audio"}

#: Rows of the modulation table one timestep owns. The tag space is the audiovisual
#: model's; the image model writes 0 and 1.
MODALITIES = 3


def _axis(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """Coordinates of one axis of ``dim`` latent cells, area-normalised.

    ``dim / sqrt_area`` is how much narrower this axis is than a square of the same
    area; the row centres run across that fraction of the lattice's side, in units of
    1/32 of it.
    """
    ratio = dim / sqrt_area
    n = dim // patch
    return (torch.arange(n, dtype=torch.float64) * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0


def frame_grid(latent_h: int, latent_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The ``(h, w)`` coordinates of one frame's patch rows, and the width axis.

    Row-major, matching the order :meth:`thenoise.dit.h3.trunk.H3DiT.patchify` gives.
    """
    area = math.sqrt(latent_h * latent_w)
    h_axis = _axis(latent_h, PATCH_SIZE[1], area)
    w_axis = _axis(latent_w, PATCH_SIZE[1], area)
    hh, ww = torch.meshgrid(h_axis, w_axis, indexing="ij")
    return torch.stack([hh.reshape(-1), ww.reshape(-1)], dim=-1), w_axis


def pixel_frames(latent_t: int) -> int:
    """How many pixel frames ``latent_t`` latent frames decode to."""
    return sum(FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)] for k in range(latent_t))


def time_spans(latent_t: int) -> List[float]:
    """How far each latent frame advances the time axis, in rope units."""
    return [FRAME_RESCALE * FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)]
            for k in range(latent_t)]


def _frame_rows(frame: torch.Tensor, t: float) -> torch.Tensor:
    """One frame's ``(h, w)`` rows, all read at the same time coordinate."""
    grid = torch.empty(frame.shape[0], 3, dtype=torch.float64)
    grid[:, 0] = t
    grid[:, 1:] = frame
    return grid


def _video_grid(latent_t: int, frame: torch.Tensor, origin: float) -> torch.Tensor:
    """Every frame's rows, the time axis advancing by each frame's own span."""
    spans = torch.tensor(time_spans(latent_t), dtype=torch.float64)
    t = origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])
    grid = torch.empty(latent_t, frame.shape[0], 3, dtype=torch.float64)
    grid[:, :, 0] = t[:, None]
    grid[:, :, 1:] = frame[None]
    return grid.reshape(-1, 3)


def _audio_grid(latent_t: int, origin: float, w_low: float, w_high: float) -> torch.Tensor:
    """The audio stream's rows: channel-major, so the whole of one channel then the other.

    ``t`` advances once per audio latent frame — one rope unit is one of them — and the
    two stereo channels are pinned to the extremes of the target's width axis rather than
    to positions of their own, because the stream has no spatial extent.
    """
    grid = torch.zeros(latent_t * 2, 3, dtype=torch.float64)
    grid[:, 0] = (origin + torch.arange(latent_t, dtype=torch.float64)).repeat(2)
    grid[:latent_t, 2] = w_low
    grid[latent_t:, 2] = w_high
    return grid


class Timesteps(NamedTuple):
    """The run's distinct timesteps, and the modulation row each segment kind reads.

    The table is indexed ``row * MODALITIES + tag``, so a run's rows are the sorted set of
    the timesteps its segments actually carry: prompt and targets follow the step, a
    conditioned row keeps its pinned label even at the last step, where the two coincide
    and the set collapses to one row.
    """

    values: Tuple[float, ...]
    rows: Dict[str, int]

    @classmethod
    def of(cls, kinds: "frozenset[str]", video: float,
           audio: Optional[float] = None) -> "Timesteps":
        """The table for the segment kinds a run has, at the step's two timesteps."""
        pinned = max(video, COND_TIMESTEP)
        per_kind = {"text": video, "video": video, "cond": pinned, "ref_img": pinned}
        if audio is not None:
            pinned_audio = max(audio, AUDIO_COND_TIMESTEP)
            per_kind.update({"audio": audio, "cond_audio": pinned_audio,
                             "ref_audio": pinned_audio})
        missing = kinds - per_kind.keys()
        if missing:
            raise ValueError(
                f"no timestep for the segment kinds {sorted(missing)}: a packing with an "
                f"audio stream has to be given the audio's timestep too")
        values = sorted({per_kind[kind] for kind in kinds})
        return cls(tuple(values), {kind: values.index(per_kind[kind]) for kind in kinds})


class _Packing:
    """Segments in reading order, with the rope coordinates each one contributes."""

    def __init__(self, text_len: int) -> None:
        grid = torch.zeros(text_len, 3, dtype=torch.float64)
        grid[:, 0] = torch.arange(text_len, dtype=torch.float64)
        self.segments: List[Tuple[int, int, str]] = [(0, text_len, "text")]
        self.positions: List[torch.Tensor] = [grid]
        self.offset = text_len

    def add(self, kind: str, rows: int, coordinates: torch.Tensor) -> None:
        self.segments.append((self.offset, self.offset + rows, kind))
        self.positions.append(coordinates)
        self.offset += rows


class Layout:
    """Segment spans, rope coordinates and modulation rows for one shape.

    ``latents`` is the *patch-padded* ``(t, h, w)`` a forward has to present; building the
    layout for one shape and feeding it another is the one mistake a packed run can make
    silently, so a forward checks it. Each variant builds one through its own classmethod:
    what differs is which segments exist and how they are ordered, not how a row is placed.
    """

    def __init__(self, text_len: int, latents: Tuple[int, int, int],
                 packing: _Packing) -> None:
        self.text_len = text_len
        self.latents = latents
        self.segments = packing.segments
        self.position_ids = torch.cat(packing.positions)
        self.seq_len = packing.offset
        self.kinds = frozenset(kind for _, _, kind in packing.segments)
        #: The streams' own spans: the only two the output heads read.
        self.video_span = _span_of(packing.segments, "video")
        self.audio_span = _span_of(packing.segments, "audio")
        #: Rows each embedding consumes, in packing order.
        self.video_rows = sum(stop - start for start, stop, kind in packing.segments
                              if SEG_EMBED[kind] == "video")
        self.audio_rows = sum(stop - start for start, stop, kind in packing.segments
                              if SEG_EMBED[kind] == "audio")

    @classmethod
    def image(cls, text_len: int, latent_h: int, latent_w: int,
              refs: Sequence[RefShape] = ()) -> "Layout":
        """``[text | target | references]`` — one frame, with the references as a suffix.

        A reference takes one unit of the time axis, so the target's frame sits behind all
        of them in the coordinates while being packed ahead of them in the rows.
        """
        frame, _ = frame_grid(latent_h, latent_w)
        packing = _Packing(text_len)
        packing.add("video", frame.shape[0], _frame_rows(frame, text_len + len(refs)))
        for i, (ref_h, ref_w) in enumerate(refs):
            ref_frame, _ = frame_grid(ref_h, ref_w)
            packing.add("ref_img", ref_frame.shape[0],
                        _frame_rows(ref_frame, float(text_len + i)))
        return cls(text_len, (1, latent_h, latent_w), packing)

    @classmethod
    def av(cls, text_len: int, latent_t: int, latent_h: int, latent_w: int,
           audio_t: int, keyframes: Sequence[Keyframe] = (),
           refs: Sequence[Reference] = ()) -> "Layout":
        """``[text | conditions | references | audio | video]`` — the released packing.

        A keyframe given as a bare integer is one frame of the target's grid; a
        :class:`Keyframe` adds the frames of a guide clip and the audio of an audio guide.
        A reference given as a :class:`Reference` keeps its own grid.

        A keyframe is anchored *inside* the clip's timeline, at a pixel frame of it; a
        reference is packed *ahead* of that timeline, so the clip and everything anchored
        in it start after the references' spans. Both conditions and references are read at
        the conditioned timestep and never denoised.

        Conditions come first in the rows, then references, which is the order the trunk
        takes their latents in.
        """
        frame, w_axis = frame_grid(latent_h, latent_w)
        w_low, w_high = float(w_axis[0]), float(w_axis[-1])
        # A bare pixel-frame index is the common case: one frame, no audio.
        keyframes = [k if isinstance(k, Keyframe) else Keyframe(k) for k in keyframes]
        # The target timeline's origin, and every anchor within it, sits behind the
        # references: they are not frames of this clip and take time units of their own.
        origin = float(text_len) + sum(ref_span(ref) for ref in refs)
        packing = _Packing(text_len)

        for pinned in keyframes:
            anchor = origin + FRAME_RESCALE * pinned.frame
            packing.add("cond", pinned.latent_t * frame.shape[0],
                        _video_grid(pinned.latent_t, frame, anchor))

        cursor = float(text_len)
        for ref in refs:
            ref_frame = frame_grid(ref.latent_h, ref.latent_w)[0]
            if ref.kind == "image":
                packing.add("ref_img", ref_frame.shape[0], _frame_rows(ref_frame, cursor))
            else:
                packing.add("ref_img", ref.latent_t * ref_frame.shape[0],
                            _video_grid(ref.latent_t, ref_frame, cursor))
            cursor += ref_span(ref)

        if audio_t:
            packing.add("audio", audio_t * 2, _audio_grid(audio_t, origin, w_low, w_high))
        packing.add("video", latent_t * frame.shape[0], _video_grid(latent_t, frame, origin))
        return cls(text_len, (latent_t, latent_h, latent_w), packing)

    def timesteps(self, video: float, audio: Optional[float] = None) -> Timesteps:
        """This packing's modulation table at the step's timesteps."""
        return Timesteps.of(self.kinds, video, audio)

    def modulation_rows(self, timesteps: Timesteps,
                        tags: Optional[torch.Tensor] = None) -> List[Tuple[int, int, int]]:
        """``(start, stop, modulation_row)`` covering the sequence, contiguous.

        A segment's row is its timestep's, offset by its modality tag: reading the wrong
        one conditions a picture as something to denoise, or denoises a target as if it
        were held. ``tags`` is the text encoder's per-token tag vector — the prompt is the
        one segment that mixes tags, because the vision placeholders inside it carry the
        visual tag — or ``None`` for a text-only prompt.
        """
        segments = []
        for start, stop, kind in self.segments:
            base = timesteps.rows[kind] * MODALITIES
            if kind == "text" and tags is not None:
                segments.extend(_tag_runs(start, stop, base, tags))
            else:
                segments.append((start, stop, base + SEG_TAG[kind]))
        return segments


def _span_of(segments: Sequence[Tuple[int, int, str]], kind: str
             ) -> Optional[Tuple[int, int]]:
    for start, stop, segment_kind in segments:
        if segment_kind == kind:
            return start, stop
    return None


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


def pad_to_patch(latent: torch.Tensor, patch_size: Tuple[int, ...]
                 ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Repeat-pad a latent onto a whole ``patch_size`` tile, over every axis it has.

    The tile is matched to the latent from the right, so a still is padded by the spatial
    part of a clip's tile. Returns the padded latent and the original trailing dimensions
    the output is cut back to.
    """
    trailing = latent.shape[2:]
    patches = patch_size[-len(trailing):]
    pads = [(-dim) % patch for dim, patch in zip(trailing, patches)]
    if not any(pads):
        return latent, tuple(trailing)
    padding = [p for pad in reversed(pads) for p in (0, pad)]
    return torch.nn.functional.pad(latent, padding, mode="replicate"), tuple(trailing)


__all__ = [
    "AUDIO_COND_TIMESTEP",
    "COND_TIMESTEP",
    "FRAME_PER_TOKEN",
    "FRAME_RESCALE",
    "Layout",
    "MODALITIES",
    "PATCH_SIZE",
    "SEG_EMBED",
    "SEG_TAG",
    "Keyframe",
    "RefShape",
    "Timesteps",
    "frame_grid",
    "pad_to_patch",
    "time_spans",
]
