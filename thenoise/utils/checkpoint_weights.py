"""Strict one-to-one floating-point Safetensors loading.

Reusable for models whose checkpoint keys map bijectively to an nn.Module
state_dict. Quantized or sharded checkpoints need a different loading policy;
this helper deliberately rejects unsupported tensor dtypes.
"""
from __future__ import annotations

from typing import Callable, Collection

import torch
from torch import nn


def validated_key_map(
    keys: set[str], mapper: Callable[[str], str], expected: set[str]
) -> dict[str, str]:
    """Require a bijection; never silently discard or overwrite checkpoint data."""
    mapped = {raw: mapper(raw) for raw in keys}
    values = set(mapped.values())
    if len(values) != len(mapped) or values != expected:
        missing = sorted(expected - values)
        extra = sorted(values - expected)
        duplicates = len(mapped) - len(values)
        raise ValueError(
            f"checkpoint keys mismatch: missing={missing[:6]}, "
            f"extra={extra[:6]}, duplicates={duplicates}"
        )
    return mapped


def load_mapped_weights(
    source, model: nn.Module, mapper: Callable[[str], str], *,
    dtype_for: Callable[[str], torch.dtype],
    excluded: Collection[str],
    check_cancel: Callable[[], None] | None,
    label: str,
) -> None:
    """Validate the *whole* checkpoint before assigning any model tensor.

    Only one tensor at a time is materialized, so large video checkpoints need
    no duplicate state dictionary in RAM. A caller supplies its per-key dtype
    policy and non-weight metadata keys.
    """
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    mapping = validated_key_map(set(source.keys()) - set(excluded), mapper, set(expected))
    for raw, key in mapping.items():
        entry = source.get_slice(raw)
        if tuple(entry.get_shape()) != expected[key]:
            raise ValueError(f"{label} shape mismatch: {raw} -> {key}")
        if entry.get_dtype() not in ("F16", "BF16", "F32"):
            raise ValueError(f"{label} unsupported tensor dtype: {raw}")

    for i, (raw, key) in enumerate(sorted(mapping.items())):
        if check_cancel is not None and i % 8 == 0:
            check_cancel()
        model.load_state_dict(
            {key: source.get_tensor(raw).to(dtype=dtype_for(key))},
            strict=False, assign=True,
        )
    if any(param.is_meta for param in model.parameters()):
        raise RuntimeError(f"{label} contains unloaded meta parameters")
