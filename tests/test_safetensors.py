"""Safetensors header/reader helpers.

Everything here runs on tiny throwaway files: header parsing (keys, metadata,
dtype/shape-only reads), the generic wrapper-prefix stripping shared by detection
and loading, and both ``load_safetensors`` strategies.
"""
from __future__ import annotations

import pytest
import torch

from conftest import int8_tensors, write_safetensors
from thenoise.utils.safetensors import (
    WRAP_PREFIXES,
    MemoryEfficientSafeOpen,
    load_dit_safetensors,
    load_safetensors,
    strip_wrap_prefixes,
)


# ------------------------------------------------------------- header/reader


def test_reader_lists_keys_and_metadata(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "m.safetensors"
    save_file({"a.weight": torch.ones(2, 3), "b.bias": torch.zeros(3)}, str(path),
              metadata={"format": "pt", "checkpoint": "test"})

    with MemoryEfficientSafeOpen(str(path)) as f:
        assert f.keys() == ["a.weight", "b.bias"]  # ``__metadata__`` is not a tensor
        assert f.metadata() == {"format": "pt", "checkpoint": "test"}


def test_reader_reads_dtype_shape_and_values(tmp_path):
    tensors = {
        "f32": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "bf16": torch.ones(4, dtype=torch.bfloat16),
        "i8": torch.tensor([-1, 2, 3], dtype=torch.int8),
        "u8": torch.tensor([1, 2], dtype=torch.uint8),
        "fp8": torch.ones(3, dtype=torch.float8_e4m3fn),
        "empty": torch.empty(0, dtype=torch.float32),
    }
    path = write_safetensors(tmp_path / "dt.safetensors", tensors)

    with MemoryEfficientSafeOpen(path) as f:
        for key, expected in tensors.items():
            got = f.get_tensor(key, device=torch.device("cpu"))
            assert got.shape == expected.shape, key
            assert got.dtype == expected.dtype, key
            if expected.numel():
                assert torch.equal(got.float(), expected.float()), key


def test_reader_can_cast_on_read(tmp_path):
    path = write_safetensors(tmp_path / "cast.safetensors", {"w": torch.ones(2, 2)})
    with MemoryEfficientSafeOpen(path) as f:
        assert f.get_tensor("w", device=torch.device("cpu"), dtype=torch.bfloat16).dtype == torch.bfloat16


def test_reader_missing_key_raises(tmp_path):
    path = write_safetensors(tmp_path / "one.safetensors", {"w": torch.ones(1)})
    with MemoryEfficientSafeOpen(path) as f:
        with pytest.raises(KeyError, match="Tensor 'nope' not found"):
            f.get_tensor("nope", device=torch.device("cpu"))


# ---------------------------------------------------------- wrapper prefixes


@pytest.mark.parametrize("prefix", WRAP_PREFIXES)
def test_strip_wrap_prefixes_removes_each_known_wrapper(prefix):
    stripped = strip_wrap_prefixes({f"{prefix}blocks.0.attn.weight": torch.ones(1)})
    assert list(stripped) == ["blocks.0.attn.weight"]


def test_strip_wrap_prefixes_leaves_bare_and_similar_keys_alone():
    keys = ["blocks.0.attn.weight", "model.other.weight", "nets.foo.weight", "net"]
    assert list(strip_wrap_prefixes({k: torch.ones(1) for k in keys})) == keys


def test_strip_wrap_prefixes_returns_a_new_dict():
    source = {"net.w": torch.ones(1), "w2": torch.ones(1)}
    stripped = strip_wrap_prefixes(source)
    assert "net.w" in source  # the input dict is never mutated
    assert set(stripped) == {"w", "w2"}


# --------------------------------------------------------------- load helpers


def test_load_safetensors(tmp_path):
    """The memory-efficient reader returns the expected tensors."""
    tensors = {"w": torch.randn(4, 4), "b": torch.arange(4)}
    path = write_safetensors(tmp_path / "s.safetensors", tensors)

    out = load_safetensors(path, device="cpu")
    assert set(out) == {"w", "b"}
    assert torch.equal(out["w"], tensors["w"])
    assert out["b"].dtype == torch.long

    cast = load_safetensors(path, device="cpu", dtype=torch.bfloat16)
    assert all(t.dtype == torch.bfloat16 for t in cast.values())


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model.", "net."])
def test_load_dit_safetensors_strips_the_wrapper(tmp_path, prefix):
    path = write_safetensors(
        tmp_path / "dit.safetensors", {f"{prefix}blocks.0.weight": torch.ones(2, 2)}
    )
    assert list(load_dit_safetensors(path, device="cpu")) == ["blocks.0.weight"]


def test_load_dit_safetensors_drop_keys(tmp_path):
    tensors = {
        "blocks.0.weight": torch.ones(2),
        "last.down.weight": torch.ones(2),
        "last.up.weight": torch.ones(2),
        "last.other": torch.ones(2),
    }
    path = write_safetensors(tmp_path / "dit.safetensors", tensors)

    kept = load_dit_safetensors(path, device="cpu", drop_keys=("last.down", "last.up"))
    assert set(kept) == {"blocks.0.weight", "last.other"}


def test_load_dit_safetensors_drop_keys_on_a_quantized_file(tmp_path):
    """Drop happens after wrapper stripping, so a wrapped file is handled too."""
    path = write_safetensors(
        tmp_path / "int8.safetensors",
        int8_tensors(prefix="model.diffusion_model.", extra={"last.down.w": torch.ones(1)}),
    )
    kept = load_dit_safetensors(path, device="cpu", drop_keys=("last.down",))
    assert "q.weight" in kept
    assert not any(k.startswith("last.down") for k in kept)
