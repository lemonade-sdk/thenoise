"""Download the Real-ESRGAN x4 model (ComfyUI repackage) for pixel upscaling.

Source: https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged
  RealESRGAN_x4plus.safetensors

Used by thenoise's ``no-refiner`` pixel-upscale path, and by the ``refined``
path when ``--upscale-factor`` exceeds the latent 2x. Optional: if absent only
the refiner (latent) upscale is available. Drop the downloaded file into a
``--upscaler-dir`` (serve) or pass its full path via ``--pixel-upscaler``
(generate).

Usage:
    python scripts/download_esrgan.py --out ./models/esrgan
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "Comfy-Org/Real-ESRGAN_repackaged"
FILE = "RealESRGAN_x4plus.safetensors"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the Real-ESRGAN x4 model")
    ap.add_argument(
        "--out", default="./models/esrgan",
        help="output directory (usable as --upscaler-dir)",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(repo_id=REPO, filename=FILE, local_dir=out)
    print(f"saved ESRGAN model to {path}")


if __name__ == "__main__":
    main()
