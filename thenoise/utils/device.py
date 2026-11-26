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


def _host_free() -> Optional[int]:
    """What the kernel would give a new allocation, from the one figure it keeps for it."""
    try:
        with open("/proc/meminfo") as info:
            for line in info:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def get_free_memory(device: Optional[Union[str, torch.device]]) -> Optional[int]:
    """Bytes the machine would hand out on ``device`` right now, None if it will not say."""
    if device is None:
        return None
    if isinstance(device, str):
        device = torch.device(device)
    if device.type != "cuda":
        return _host_free()
    try:
        return torch.cuda.mem_get_info(device)[0]
    except (RuntimeError, AttributeError):
        return None


def spare_bytes(device: Optional[Union[str, torch.device]]) -> Optional[int]:
    """What ``device`` could still hand out, counting bytes its allocator holds unused."""
    free = get_free_memory(device)
    if free is None:
        return None
    if device is not None and torch.device(device).type != "cuda":
        return free
    try:
        kept = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    except (RuntimeError, AttributeError):
        return free
    return free + max(0, kept)


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