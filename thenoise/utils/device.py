"""Device helpers that are agnostic to the active backend (CUDA/ROCm, XPU, MPS)."""
from typing import Optional, Union
import gc
import torch

def clean_memory_on_device(device: Optional[Union[str, torch.device]]):
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "cpu":
        pass
    elif device.type == "mps":  # not tested
        torch.mps.empty_cache()


def release_device_memory(device: Optional[Union[str, torch.device]]):
    """Give back what a just-dropped model or pool entry was holding. Collecting first
    is what makes it work: a tensor frees its allocator block only when unreferenced.
    """
    gc.collect()
    clean_memory_on_device(device)


def get_device_memory(device: Optional[Union[str, torch.device]]) -> Optional[int]:
    """Total device memory in bytes for a compute device, or ``None`` if unknown.

    Only ``cuda`` is supported (ROCm aliases ``cuda`` -> hip).
    """
    if device is None:
        return None
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory
    return None


def get_free_memory(device: Optional[Union[str, torch.device]]) -> Optional[int]:
    """Bytes the driver would hand out on ``device`` right now, None if it will not say.

    Unlike the allocator's measure, this counts the compute context and other processes.
    """
    if device is None:
        return None
    if isinstance(device, str):
        device = torch.device(device)
    if device.type != "cuda":
        return None
    try:
        return torch.cuda.mem_get_info(device)[0]
    except (RuntimeError, AttributeError):
        return None


def synchronize_device(device: Optional[Union[str, torch.device]]):
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

# torch._int_mm needs integer tensor cores (sm_75) on NVIDIA; Triton, and so
# torch.compile, needs sm_70.
INT_MM_MIN_CAPABILITY = (7, 5)
COMPILE_MIN_CAPABILITY = (7, 0)


@torch.compiler.assume_constant_result
def nvidia_capability(device: Union[str, torch.device]) -> Optional[tuple[int, int]]:
    """NVIDIA compute capability; None for other backends or an unavailable device."""
    dev = torch.device(device)
    if torch.version.hip or dev.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(dev)


def int_mm_supported(device: torch.device) -> bool:
    """Whether to keep the existing INT8 path; only older NVIDIA needs a fallback."""
    cap = nvidia_capability(device)
    return not (cap is not None and cap < INT_MM_MIN_CAPABILITY)


def compile_supported(device: Union[str, torch.device]) -> bool:
    """False when ``device`` is an NVIDIA GPU too old for Triton."""
    cap = nvidia_capability(device)
    return not (cap is not None and cap < COMPILE_MIN_CAPABILITY)
