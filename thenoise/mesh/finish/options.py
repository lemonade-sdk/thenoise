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
        0.0, label="Clean up junk", ge=0.0, le=1.0, step=0.01,
        help="throw away pieces of the shape smaller than this fraction of its widest part; "
             "0 keeps everything",
        filter="floaters", stage=FINISH)

    triangles: float = option(
        1.0, label="Level of detail", ge=0.001, le=1.0, step=0.01, ui_min=0.01,
        help="keep this fraction of the surface detail; less gives a smaller file and a "
             "coarser shape",
        filter="simplify", stage=FINISH)

    smooth: float = option(
        0.0, label="Smooth the surface", ge=0.0, le=1.0, step=0.01,
        help="ease out the roughness a cut leaves, corners and all, on whatever the detail "
             "level has left by then",
        filter="smooth", stage=FINISH)

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
