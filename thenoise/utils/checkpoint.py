"""Checkpoint-level markers: what a DiT file says about how it wants to be run.

Some checkpoints carry header entries that are not weights but *flags* — the empty
``__index_timestep_zero__`` buffer, written by trainers that conditioned an edit
model's reference tokens at timestep zero, is one. Reading markers here, once, at
load time keeps that knowledge out of the model adapters: a marker maps to a
*generation preference* and its value, and the adapter just asks for the preference
(see ``DiffusionModel.pref``, which layers request > checkpoint > model default).

Two properties make this safe to extend and safe to ignore:

* a model that has no use for a preference never asks for it, so a marker it does
  not care about changes nothing;
* a checkpoint that carries no marker contributes no entry, so the automatic layer
  simply has nothing to say and the model default applies.

Markers are a *hint* layer, never a gate: checkpoints circulate that were trained
with timestep-zero reference conditioning but carry no marker (and LoRAs can change
what a checkpoint expects), so an explicit request is always honoured.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from thenoise.utils.safetensors import checkpoint_keys

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Marker:
    """A checkpoint entry that implies ``pref = value`` when present."""

    pref: str
    value: Any
    key: str


# The registry. One line per marker; earlier entries win if several markers feed
# the same preference.
CHECKPOINT_MARKERS: Tuple[Marker, ...] = (
    Marker("ref_method", "index_timestep_zero", "__index_timestep_zero__"),
)


def detect_checkpoint_prefs(dit_path: str) -> Dict[str, Any]:
    """Preferences implied by the markers in ``dit_path`` (``{}`` when none apply).

    An unreadable path yields ``{}`` rather than raising: markers are an optional
    hint, and the runtime must not fail to load weights over a header it could not
    parse.
    """
    try:
        keys = checkpoint_keys(dit_path)
    except FileNotFoundError:
        # Nothing to read (a stub path, or a path that moved on): no hints.
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


__all__ = ["Marker", "CHECKPOINT_MARKERS", "detect_checkpoint_prefs"]
