"""Kernel choices a dependency pins to the wrong architecture, corrected here.

Each patch is a private seam in a dependency, an architecture measured with it, and a
silent fall back to the shipped choice if the seam moves. An architecture joins one of
the lists below only with a measurement, never with a guess about the hardware.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional, Union

import torch

logger = logging.getLogger(__name__)

__all__ = ["apply_device_patches"]

Device = Optional[Union[str, torch.device]]

_MARK = "_thenoise_device_patch"

# comfy_kitchen picks its INT8 GEMM tile schedule by architecture. gfx1150 measured
# ~1.5x faster on the tiled one, bitwise identical; gfx1151 measured a wash.
TILED_INT8_ARCHES = ("gfx1150",)


def _gfx_arch(device: Device) -> Optional[str]:
    """``gfx1151`` for ``device``, without the ``:xnack-`` or stepping suffixes."""
    if device is None or not torch.cuda.is_available() or not torch.version.hip:
        return None
    try:
        name = torch.cuda.get_device_properties(torch.device(device).index).gcnArchName
    except Exception:
        return None
    match = re.match(r"gfx\d+", name)
    return match.group(0) if match else None


def _tiled_int8_gemm(device: Device) -> bool:
    """Use comfy_kitchen's tiled INT8 GEMM here, keeping its own shape limits."""
    arch = _gfx_arch(device)
    if arch not in TILED_INT8_ARCHES:
        return False
    try:
        from comfy_kitchen.backends import hip
        pick = hip._use_tiled_int8
    except (ImportError, AttributeError):
        logger.warning("%s is listed for the tiled INT8 GEMM but comfy_kitchen no longer "
                       "asks _use_tiled_int8; running with the schedule it chose", arch)
        return False
    if getattr(pick, _MARK, False):
        return True

    def tiled(m: int, n: int, k: int, where: torch.device) -> bool:
        return bool(pick(m, n, k, where)) or (m > 8 and n > 64 and k > 0 and k % 128 == 0)

    setattr(tiled, _MARK, True)
    hip._use_tiled_int8 = tiled
    logger.info("INT8 GEMM: tiled schedule enabled for %s", arch)
    return True


PATCHES: tuple[tuple[str, Callable[[Device], bool]], ...] = (("int8_tiled_gemm", _tiled_int8_gemm),)


def apply_device_patches(device: Device = None) -> dict[str, bool]:
    """Apply every patch that fits ``device``. Idempotent; a missing patch costs speed only."""
    done = {}
    for name, patch in PATCHES:
        try:
            done[name] = bool(patch(device))
        except Exception:
            logger.warning("device patch %s failed on %s", name, device, exc_info=True)
            done[name] = False
    return done
