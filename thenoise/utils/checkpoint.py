"""Checkpoint-level markers: what a DiT file says about how it wants to be run.

Some checkpoints carry header entries that are not weights but *flags* — the empty
``__index_timestep_zero__`` buffer, written by trainers that conditioned an edit
model's reference tokens at timestep zero, is one. Reading markers here, once, at
load time keeps that knowledge out of the model adapters: a marker maps to a
*generation preference* and its value, and the adapter just asks for the preference.

Markers are a *hint* layer, never a gate: checkpoints circulate that were trained
with timestep-zero reference conditioning but carry no marker, so an explicit
request is always honoured.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Tuple

from thenoise.utils.safetensors import checkpoint_keys

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Marker:
    """A checkpoint entry that implies ``pref = value`` when present."""

    pref: str
    value: Any
    key: str


CHECKPOINT_MARKERS: Tuple[Marker, ...] = (
    Marker("ref_method", "index_timestep_zero", "__index_timestep_zero__"),
)

CHECKPOINT_MARKER_KEYS: FrozenSet[str] = frozenset(m.key for m in CHECKPOINT_MARKERS)


def detect_checkpoint_prefs(dit_path: str) -> Dict[str, Any]:
    """Preferences implied by the markers in ``dit_path`` (``{}`` when none apply).

    An unreadable path yields ``{}`` rather than raising: the runtime must not fail
    to load weights over a header it could not parse.
    """
    try:
        keys = checkpoint_keys(dit_path)
    except FileNotFoundError:
        logger.debug("No checkpoint markers: %s not readable", dit_path)
        return {}
    except Exception as exc:  # not a safetensors file / truncated header
        logger.warning("Could not read checkpoint markers from %s (%s)", dit_path, exc)
        return {}

    prefs: Dict[str, Any] = {}
    for marker in CHECKPOINT_MARKERS:
        if marker.key not in keys:
            continue
        if marker.pref in prefs:
            logger.debug("Ignoring marker %s: %s already set", marker.key, marker.pref)
            continue
        prefs[marker.pref] = marker.value
        logger.info("Checkpoint marker %s -> %s=%s", marker.key, marker.pref, marker.value)
    return prefs


__all__ = ["Marker", "CHECKPOINT_MARKERS", "CHECKPOINT_MARKER_KEYS", "detect_checkpoint_prefs"]
