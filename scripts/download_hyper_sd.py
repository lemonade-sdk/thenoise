"""Download the Hyper-SD SDXL 8-step CFG-preserved LoRA for faster generation.

Source: https://huggingface.co/ByteDance/Hyper-SD
  Hyper-SDXL-8steps-CFG-lora.safetensors

Hyper-SD is a trajectory-segmented consistency distillation that lets SDXL
render in 8 steps instead of ~28 while keeping quality. This is the
CFG-preserved variant (supports guidance 5-8), which fits thenoise's CFG-based
sampler. The SDXL model auto-converts its diffusers-keyed UNet LoRA to
our LDM key naming, so it works as a normal ``--lora``.

Use at 8 steps::
    python -m thenoise generate --lora Hyper-SDXL-8steps-CFG-lora.safetensors --steps 8

Optional: also grabs the 4-step LoRA used by Fooocus.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "ByteDance/Hyper-SD"
FILES = {
    "8step_cfg": "Hyper-SDXL-8steps-CFG-lora.safetensors",
    "4step": "Hyper-SDXL-4steps-lora.safetensors",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Hyper-SD SDXL LoRAs")
    ap.add_argument(
        "--out", default="./models/sdxl/lora",
        help="output directory (usable as the model's --lora-dir)",
    )
    ap.add_argument(
        "--variants", default="8step_cfg", choices=list(FILES) + ["all"],
        help="which Hyper-SD LoRA to fetch (default: 8-step CFG)",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names = list(FILES) if args.variants == "all" else [args.variants]
    for name in names:
        filename = FILES[name]
        path = hf_hub_download(repo_id=REPO, filename=filename, local_dir=out)
        print(f"saved Hyper-SD LoRA ({name}) to {path}")


if __name__ == "__main__":
    main()
