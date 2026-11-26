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


# Below sm_75, comfy_kitchen's cuda kernels give wrong output without an error, Triton
# (and so torch.compile) needs sm_70, and torch._int_mm is not supported.
PRE_TURING_CAPABILITY = (7, 5)


def _cuda_capability(device: Device) -> Optional[tuple[int, int]]:
    if device is None or not torch.cuda.is_available() or torch.version.hip:
        return None
    dev = torch.device(device)
    if dev.type != "cuda":
        return None
    try:
        return torch.cuda.get_device_capability(dev.index if dev.index is not None else 0)
    except Exception:
        return None


_BACKEND_MIN_CAPABILITY = {"cuda": PRE_TURING_CAPABILITY, "triton": (7, 0)}


def _per_device_backends() -> None:
    """Answer comfy_kitchen's ``is_available`` for the current card, so newer cards in
    the same process keep their backends."""
    try:
        from comfy_kitchen.registry import registry
    except ImportError:
        return
    native = registry.is_available
    if getattr(native, _MARK, False):
        return

    def is_available(backend_name: str) -> bool:
        if not native(backend_name):
            return False
        need = _BACKEND_MIN_CAPABILITY.get(backend_name)
        if need is None or not torch.cuda.is_available() or torch.version.hip:
            return True
        try:
            return torch.cuda.get_device_capability(torch.cuda.current_device()) >= need
        except Exception:
            return True

    setattr(is_available, _MARK, True)
    registry.is_available = is_available


def _pre_turing_fallbacks(device: Device) -> bool:
    """Pre-Turing NVIDIA: eager comfy_kitchen kernels, no torch.compile, fp32 int-matmul."""
    cap = _cuda_capability(device)
    if cap is None or cap >= PRE_TURING_CAPABILITY:
        return False
    _per_device_backends()
    # force_eager rather than dynamo.config.disable: the latter fails fullgraph=True checks.
    torch.compiler.set_stance("force_eager")
    if not getattr(torch._int_mm, _MARK, False):
        native = torch._int_mm

        def int_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            dev = a.device
            if dev.type == "cuda" and torch.cuda.get_device_capability(dev) < PRE_TURING_CAPABILITY:
                # exact while the sum stays below 2^24
                return torch.matmul(a.float(), b.float()).round_().to(torch.int32)
            return native(a, b)

        setattr(int_mm, _MARK, True)
        torch._int_mm = int_mm
    logger.info("pre-Turing GPU (sm_%d%d) on %s: comfy_kitchen eager kernels on that card, "
                "torch.compile off, fp32 int-matmul", cap[0], cap[1], device)
    return True


PATCHES: tuple[tuple[str, Callable[[Device], bool]], ...] = (
    ("int8_tiled_gemm", _tiled_int8_gemm),
    ("pre_turing_fallbacks", _pre_turing_fallbacks),
)


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
