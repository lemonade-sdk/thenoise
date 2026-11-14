"""Role-keyed LoRA specs: the shape the wire carries, and the rules that need no model.

A *role* names one LoRA-able component of a model, by the name the model registers it
under for residency (``dit``, ``dit_low``). Keeping the normalization here rather than
in a model or a router means the wire, the pipeline and the adapter agree on one
spelling without importing each other.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

#: The role of a model's single LoRA-able component, and what a bare list means.
DEFAULT_LORA_ROLE = "dit"

#: What a client may send: the specs for the default role, or them grouped by role.
LoraSpecs = list[str] | dict[str, list[str]] | None


def lora_specs_by_role(specs: LoraSpecs) -> dict[str, list[str]]:
    """The wire form -> ``{role: specs}``.

    A role with an empty list means the same as not naming it, so nothing downstream has
    to distinguish "no LoRA" from "LoRA, none of them".
    """
    if specs is None:
        return {}
    if isinstance(specs, list):
        return {DEFAULT_LORA_ROLE: list(specs)} if specs else {}
    return {role: list(values) for role, values in specs.items() if values}


def check_lora_roles(specs: Mapping[str, list[str]], roles: Sequence[str]) -> None:
    """Refuse roles this model does not offer, naming the ones it does."""
    unknown = sorted(role for role in specs if role not in roles)
    if unknown:
        raise ValueError(
            f"lora_specs names {unknown}, which this model does not offer; its "
            f"LoRA-able components are {list(roles)}"
        )


def flat_lora_specs(specs: Mapping[str, list[str]]) -> list[str]:
    """Every spec in the mapping, for the places that record what was applied."""
    return [spec for values in specs.values() for spec in values]
