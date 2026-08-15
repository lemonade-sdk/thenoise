"""Convert a Real-ESRGAN ``*.pth`` checkpoint to ``.safetensors`` for thenoise.

The official ``xinntao/Real-ESRGAN`` releases ship ``*.pth`` torch pickles whose
state dict is wrapped under a ``params_ema`` (or ``params``) key, which
thenoise's safetensors-only loader cannot open. This unwraps one and saves it as
a ``.safetensors`` file that ``thenoise.upscale.esrgan`` accepts directly.

The official weights already use the ComfyUI key naming (``body.*``,
``conv_first``, ``conv_body``, ``conv_up1``, ``conv_up2``, ``conv_hr``,
``conv_last``) that the loader expects, so no key remapping is needed — only the
wrapper is removed.

Usage:
    python scripts/convert_esrgan_pth.py RealESRGAN_x2plus.pth -o models/esrgan
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

# Wrapper keys used by the official releases, in preference order.
_WRAPPERS = ("params_ema", "params", "state_dict")


def _unwrap(obj: Any) -> dict:
    """Return the bare state dict from a torch-loaded checkpoint ``obj``."""
    if not isinstance(obj, dict):
        return obj
    for key in _WRAPPERS:
        if key in obj:
            return obj[key]
    return obj


def convert(pth: Path, out: Path) -> Path:
    """Load ``pth`` (torch pickle), unwrap it and save as safetensors at ``out``."""
    state = _unwrap(torch.load(pth, map_location="cpu", weights_only=False))
    save_file(state, str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert a Real-ESRGAN .pth checkpoint to .safetensors"
    )
    ap.add_argument("pth", type=Path, help="source .pth checkpoint")
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output .safetensors path (default: same name next to the .pth)",
    )
    args = ap.parse_args()

    out = args.out or args.pth.with_suffix(".safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    convert(args.pth, out)
    print(f"saved ESRGAN safetensors to {out}")


if __name__ == "__main__":
    main()
