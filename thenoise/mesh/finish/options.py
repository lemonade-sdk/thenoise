from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from thenoise.utils.options import (FilterSpec, chain_in_stage, declared_chain,
                                   defaults_of, flat_values, option,
                                   option_names, validate)

FINISH = "finish"


@dataclass
class MeshFinishOptions:
    """What a request asks of a shape once its surface has been cut."""

    floaters: float = option(
        0.0, label="Drop floaters", ge=0.0, le=1.0, step=0.01,
        help="discard pieces reaching less than this fraction across the whole shape; the "
             "widest piece reaches all of it, so 0.15 keeps anything spanning a sixth as far",
        filter="floaters", stage=FINISH)

    triangles: float = option(
        1.0, label="Triangles", ge=0.001, le=1.0, step=0.01, ui_min=0.01,
        help="keep this fraction of the triangles the surface was cut with; the shape "
             "keeps its detail where it has any and loses it where it is flat",
        filter="simplify", stage=FINISH)

    def __post_init__(self) -> None:
        validate(self)


FINISH_NAMES = option_names(MeshFinishOptions)

DEFAULTS: Dict[str, Any] = defaults_of(MeshFinishOptions)

CHAIN: Tuple[FilterSpec, ...] = declared_chain(MeshFinishOptions)


def filter_specs(stage: str = FINISH) -> Tuple[FilterSpec, ...]:
    """The filters of one stage, in chain order."""
    return chain_in_stage(CHAIN, stage)


def flat_options(holder: Any) -> Dict[str, Any]:
    """The finish knobs read off anything that carries them flat."""
    return flat_values(MeshFinishOptions, holder, noun="finish")


def options_from(holder: Any) -> MeshFinishOptions:
    """The finish knobs of anything carrying them flat, as the chain wants them."""
    return MeshFinishOptions(**flat_options(holder))


__all__ = [
    "CHAIN",
    "DEFAULTS",
    "FINISH",
    "FINISH_NAMES",
    "MeshFinishOptions",
    "filter_specs",
    "flat_options",
    "options_from",
]
