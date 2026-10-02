import numpy as np
import torch
import json
import struct
from pathlib import Path
from typing import Dict, Union, Optional

from thenoise.utils.setup_logging import setup_logging
from thenoise.utils.device import synchronize_device

setup_logging()
import logging

logger = logging.getLogger(__name__)

class MemoryEfficientSafeOpen:
    """Reader for safetensors files that memory-maps large tensors."""

    def __init__(self, filename, use_numpy_memmap=True):
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()
        self.use_numpy_memmap = use_numpy_memmap

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        """All tensor names in the file (excludes metadata)."""
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        return self.header.get("__metadata__", {})

    def _read_header(self):
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        header_json = self.file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def get_tensor(self, key: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Load a single tensor.

        **Note:** for a CUDA target the transfer is non-blocking, so the caller must
        synchronize before using the tensor.
        """
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]
        num_bytes = offset_end - offset_start

        original_dtype = self._get_torch_dtype(metadata["dtype"])
        target_dtype = dtype if dtype is not None else original_dtype

        if num_bytes == 0:
            return torch.empty(metadata["shape"], dtype=target_dtype, device=device)

        non_blocking = device is not None and device.type == "cuda"

        tensor_offset = self.header_size + 8 + offset_start

        # Memmap only for non-CPU targets: a CPU tensor would keep the file locked
        # for no benefit, and only large tensors make the mapping worth it.
        if self.use_numpy_memmap and num_bytes > 10 * 1024 * 1024 and device is not None and device.type != "cpu":
            mm = np.memmap(self.filename, mode="c", dtype=np.uint8, offset=tensor_offset, shape=(num_bytes,))
            byte_tensor = torch.from_numpy(mm)  # zero copy
            del mm

            cpu_tensor = self._deserialize_tensor(byte_tensor, metadata)
            del byte_tensor

            gpu_tensor = cpu_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)
            del cpu_tensor
            return gpu_tensor

        self.file.seek(tensor_offset)
        numpy_array = np.fromfile(self.file, dtype=np.uint8, count=num_bytes)
        byte_tensor = torch.from_numpy(numpy_array)
        del numpy_array

        deserialized_tensor = self._deserialize_tensor(byte_tensor, metadata)
        del byte_tensor

        return deserialized_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)

    def _deserialize_tensor(self, byte_tensor: torch.Tensor, metadata: Dict):
        """View the raw bytes as the tensor's dtype/shape."""
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

# Wrapper prefixes that repackagings prepend to *every* tensor name, e.g.
# ``net.`` and ``model.diffusion_model.``. Shared by key detection and loading so
# raw and repackaged checkpoints resolve AND load identically.
WRAP_PREFIXES = ("model.diffusion_model.", "net.")


def unwrap_key(key: str) -> str:
    """Return a tensor name with any generic wrapper prefix stripped."""
    for prefix in WRAP_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def strip_wrap_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a new state dict with generic wrapper prefixes stripped from keys."""
    return {unwrap_key(key): value for key, value in state_dict.items()}


def checkpoint_keys(path: str) -> set[str]:
    """The tensor names in the safetensors header of ``path``, wrapper-normalized.

    Reads the header only (no tensor data), so it is cheap enough to run at load
    time.
    """
    with MemoryEfficientSafeOpen(path) as f:
        return {unwrap_key(k) for k in f.keys()}


def load_dit_safetensors(
    path: str,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    drop_keys: Optional[tuple[str, ...]] = None,
) -> dict[str, torch.Tensor]:
    """Load a DiT checkpoint state dict, stripping generic repackaging wrapper
    prefixes so raw and repackaged checkpoints load identically.
    """
    sd = load_safetensors(path, device=device, dtype=dtype)
    sd = strip_wrap_prefixes(sd)
    if drop_keys:
        sd = {k: v for k, v in sd.items() if not k.startswith(drop_keys)}
    return sd


def load_safetensors(
    path: str,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> dict[str, torch.Tensor]:
    """Load a safetensors file into a state dict using the memory-efficient reader."""
    state_dict = {}
    device = torch.device(device) if device is not None else None
    with MemoryEfficientSafeOpen(path) as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key, device=device, dtype=dtype)
        synchronize_device(device)
    return state_dict


UPSCALE_WEIGHT_DIR = Path(__file__).resolve().parent.parent / "upscale" / "weights"

def upscale_weight_path(filename: str) -> Path:
    """Path to a latent-upscaler weight file vendored into the package.

    Raises rather than handing back a nonexistent path, which can only mean the
    package was installed without its package-data.
    """
    path = UPSCALE_WEIGHT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"upscaler weights not found at {path}; "
            "the package was not installed with its package-data"
        )
    return path

