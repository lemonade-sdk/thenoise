import numpy as np
import torch
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Union, Optional

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


#: A model's identity, in tensor names.
@dataclass(frozen=True)
class Signature:
    """What makes a checkpoint this model's, stated in names rather than in code.

    ``keys`` are exact tensor names, ``prefixes`` need at least one key beneath each,
    and ``absent`` names must not appear — the last is how a file that is a superset of
    this model (a sibling stream, a layout this loader cannot build) is refused instead
    of half-loaded. All three are wrapper-normalized.

    One declaration answers both the catalog's ``detect`` and the geometry read, so a
    file cannot be claimed by a loader that would then fail to build it.
    """

    name: str
    keys: Tuple[str, ...] = ()
    prefixes: Tuple[str, ...] = ()
    absent: Tuple[str, ...] = ()


class Header:
    """One checkpoint's header — names, shapes, dtypes, metadata — parsed exactly once.

    Names come out wrapper-normalized (see :func:`unwrap_key`), so a repackaged file is
    read in the same names as the module tree it loads onto. The file's own spelling is
    kept in ``raw_entries`` for the one caller that has to read a tensor back by raw
    key: a quantized layer's LoRA undo.

    Build one with :func:`read_header`; carry it to everything that would otherwise open
    the file again.
    """

    def __init__(
        self,
        path: str,
        shapes: Dict[str, Tuple[int, ...]],
        dtypes: Dict[str, str],
        raw_entries: Dict[str, dict],
        metadata: Dict[str, str],
    ):
        self.path = path
        self.shapes = shapes
        self.dtypes = dtypes
        self.raw_entries = raw_entries
        self.metadata = metadata
        # The catalog entry that claimed this file, set by `thenoise.models.resolve`: the
        # parse that identified the checkpoint travels with it instead of being redone.
        self.model_cls: Optional[type] = None

    def __contains__(self, name: str) -> bool:
        return name in self.shapes

    @property
    def names(self) -> Tuple[str, ...]:
        """Every tensor name, wrapper-normalized, in the file's own order."""
        return tuple(self.shapes)

    def shape(self, name: str) -> Tuple[int, ...]:
        """The shape stored under ``name``, with the file named when it is not there.

        A signature check is what normally rules that out; this is the one that catches
        a geometry key no signature mentioned.
        """
        try:
            return self.shapes[name]
        except KeyError:
            raise KeyError(f"{self.path} carries no '{name}'") from None

    def metadata_json(self, section: str) -> dict:
        """One JSON section of the file's ``__metadata__``, empty when absent or broken.

        A foreign or hand-repacked file can carry anything under ``config``, and a
        section that will not parse only means the geometry has to come from shapes.
        """
        raw = self.metadata.get(section)
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("Could not parse the checkpoint's %r metadata", section)
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def block_count(self, prefix: str) -> int:
        """How many ``<prefix><i>.`` blocks the file spans (0 when it has none).

        Depth is the one architectural knob no released checkpoint stores as a number.
        """
        indices = [
            int(head)
            for name in self.shapes
            if name.startswith(prefix)
            and (head := name[len(prefix):].split(".", 1)[0]).isdigit()
        ]
        return max(indices) + 1 if indices else 0

    def identifies(self, signature: Signature) -> bool:
        """Whether this file is ``signature``'s model, without explaining itself.

        An empty signature claims nothing: a model that declared no names and wrote no
        ``detect`` must not swallow every checkpoint in the catalog.
        """
        if not (signature.keys or signature.prefixes):
            return False
        return not self.missing(signature) and not self.unwanted(signature)

    def require(self, signature: Signature) -> None:
        """Refuse a file that is not ``signature``'s model, naming what gave it away.

        Geometry read out of a header is only as good as the claim that the file is
        what it says.
        """
        if missing := self.missing(signature):
            raise ValueError(
                f"{self.path} does not match the {signature.name} signature "
                f"(missing {missing})")
        if unwanted := self.unwanted(signature):
            raise ValueError(
                f"{self.path} does not match the {signature.name} signature: "
                f"it carries {unwanted}")

    def missing(self, signature: Signature) -> list[str]:
        """The names ``signature`` asks for that this file does not carry."""
        gone = [key for key in signature.keys if key not in self.shapes]
        return gone + [prefix for prefix in signature.prefixes
                       if not any(name.startswith(prefix) for name in self.shapes)]

    def unwanted(self, signature: Signature) -> list[str]:
        """The names ``signature`` rules out that this file does carry."""
        return [key for key in signature.absent if key in self.shapes]


    def relabel(self, key_map: Callable[[str], str]) -> "Header":
        """A view of this header under renamed names, without reading the file again.

        For a loader whose geometry lives under the names the model uses where the
        checkpoint spells a leaf differently.
        """
        shapes: Dict[str, Tuple[int, ...]] = {}
        dtypes: Dict[str, str] = {}
        for name, shape in self.shapes.items():
            renamed = key_map(name)
            if renamed in shapes:
                raise RuntimeError(f"{self.path}: two checkpoint names resolve to {renamed!r}")
            shapes[renamed] = shape
            dtypes[renamed] = self.dtypes[name]
        return Header(self.path, shapes, dtypes, self.raw_entries, self.metadata)


def read_header(path: str, key_map: Optional[Callable[[str], str]] = None) -> Header:
    """Parse ``path``'s header once. The only place a checkpoint header is opened.

    ``key_map`` renames on top of the wrapper stripping, for a loader whose geometry is
    read under the names the model uses instead of the checkpoint's.
    """
    with MemoryEfficientSafeOpen(path) as f:
        raw_entries = {k: f.header[k] for k in f.keys()}
        metadata = f.metadata()
    header = Header(
        path,
        {unwrap_key(raw): tuple(entry["shape"]) for raw, entry in raw_entries.items()},
        {unwrap_key(raw): entry["dtype"] for raw, entry in raw_entries.items()},
        raw_entries,
        metadata,
    )
    return header if key_map is None else header.relabel(key_map)


def as_header(checkpoint: Union[str, Header]) -> Header:
    """``checkpoint``, read if it is still a path.

    Lets a helper answer either a caller that already parsed the file and one that only
    has a name, without either of them knowing which it got.
    """
    return checkpoint if isinstance(checkpoint, Header) else read_header(checkpoint)


def require_divisible(value: int, divisor: int, what: str, by: str) -> None:
    """Refuse a width that does not divide, in the same words every time.

    A trunk that is not a whole number of heads predicts a plausible-but-wrong
    velocity rather than failing.
    """
    if value % divisor:
        raise ValueError(f"{what} {value} is not a multiple of {by} {divisor}")


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

