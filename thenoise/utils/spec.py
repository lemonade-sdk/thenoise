"""What every medium's request spec repeats.

A spec is what arrives off the wire before the model has been asked anything: field
shapes the same in each medium, the seed rule, and the merge with the model's defaults
that is the whole of ``request → checkpoint marker → model default``.
"""
from __future__ import annotations

import secrets
from typing import Annotated, Any, Callable, Mapping

from pydantic import AfterValidator, BaseModel, Field

Positive = Annotated[int, Field(gt=0)]
PositiveFinite = Annotated[float, Field(gt=0, allow_inf_nan=False)]
#: A picture as it was sent, before anything has tried to read it.
Picture = Annotated[bytes, Field(min_length=1)]


def _seed_in_generator_range(value: int | None) -> int | None:
    # torch's generator takes a signed 64-bit value; a wider one is refused rather
    # than wrapped into a different run than the caller asked for.
    if value is not None and not -(2**63) <= value < 2**64:
        raise ValueError("seed is outside torch.Generator's supported range")
    return value


#: A seed, checked on the way in. ``None`` and -1 both ask for a random one, which is
#: drawn at resolve time rather than here so a rejected request draws nothing.
Seed = Annotated[int | None, AfterValidator(_seed_in_generator_range)]


def resolve_seed(value: int | None) -> int:
    """The seed a run draws on: an unset one, or -1, asks for a random one."""
    return secrets.randbits(32) if value is None or value == -1 else value


def prefer(spec: BaseModel, defaults: Mapping[str, Any]) -> Callable[..., Any]:
    """``pref(key)`` answers a request field, or the model's default when it was unset.

    A spec field is ``None`` exactly where the request said nothing, and an unset
    default is a bug rather than a ``None`` to pass on, so this looks the key up.
    ``default_key`` reads a default under another name than the field's.
    """

    def pref(key: str, default_key: str | None = None) -> Any:
        value = getattr(spec, key)
        return defaults[default_key or key] if value is None else value

    return pref


__all__ = ["Picture", "Positive", "PositiveFinite", "Seed", "prefer", "resolve_seed"]
