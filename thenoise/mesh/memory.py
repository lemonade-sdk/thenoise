from __future__ import annotations

from typing import Union

import torch

from thenoise.utils.device import spare_bytes

CUT_BYTES_PER_CORNER = 48


def amount(nbytes: int) -> str:
    """Bytes in the unit a refusal should be read in."""
    return (f"{nbytes / (1 << 30):.1f} GiB" if nbytes >= 1 << 30
            else f"{nbytes / (1 << 20):.0f} MiB")


def volume_bytes(resolution: int) -> int:
    """What a cut at ``resolution`` asks the machine to have, per corner of its grid."""
    return (resolution + 1) ** 3 * CUT_BYTES_PER_CORNER


def room(device: Union[str, torch.device], need: int, saying: str, advice: str) -> None:
    """Refuse work the machine cannot take, before it is asked for."""
    spare = spare_bytes(device)
    if spare is not None and need > spare:
        raise ValueError(f"{saying} needs {amount(need)} and only {amount(spare)} is free: "
                         f"{advice}")


__all__ = ["amount", "room", "volume_bytes"]
