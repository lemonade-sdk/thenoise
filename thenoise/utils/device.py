"""Device helpers that are agnostic to the active backend (CUDA/ROCm, XPU, MPS)."""
from typing import Optional, Union
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


def get_device_memory(device: Optional[Union[str, torch.device]]) -> Optional[int]:
    """Total device memory in bytes for a compute device, or ``None`` if unknown.

    Only ``cuda`` (ROCm aliases ``cuda`` -> hip) is currently supported; other
    device kinds (cpu, meta, mps, xpu) return ``None``. Used by the offload
    auto-detection, which compares the expected resident weight bytes against this.
    """
    if device is None:
        return None
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory
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