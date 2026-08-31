"""Download the SesquiLSR SDXL latent upscaler and convert fp32 -> bf16.

thenoise's latent (refined) upscaler is SesquiLSR. SDXL uses a 4-channel VAE, so
it needs the ``upscaler_SDXL.safetensors`` weights, which are committed bf16 in
``thenoise/upscale/weights/`` (where the ``"sdxl"`` format registry reads from).
This fetches the upstream fp32 weights from ``LoganBooker/SesquiLSR`` and
converts them in place, mirroring ``scripts/download_zimage.py``'s flux upscaler
step.

Usage:
    python scripts/download_sesqui_sdxl.py

Re-running is a no-op when ``thenoise/upscale/weights/upscaler_SDXL.safetensors``
already exists.
"""
from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

#: Upstream fp32 SDXL latent upscaler (MIT-licensed).
SESQUI_SDXL_UPSCALER_URL = (
    "https://github.com/LoganBooker/SesquiLSR/raw/main/models/upscaler_SDXL.safetensors"
)

#: Package weights dir the upscaler format registry reads from.
UPSCALER_WEIGHTS_DIR = (
    Path(__file__).resolve().parents[1] / "thenoise" / "upscale" / "weights"
)


def download_sdxl_upscaler() -> Path:
    """Download the SesquiLSR SDXL upscaler and convert fp32 -> bf16 in place."""
    dest = UPSCALER_WEIGHTS_DIR / "upscaler_SDXL.safetensors"
    if dest.is_file():
        print(f"{'SDXL upscaler':13s} -> {dest} (already present)")
        return dest

    UPSCALER_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "upscaler_SDXL.safetensors"
        print("Downloading SDXL upscaler from SesquiLSR (fp32)")
        urllib.request.urlretrieve(SESQUI_SDXL_UPSCALER_URL, fp32)

        print(f"Converting {fp32.name} fp32 -> bf16")
        sd = {k: v.to(torch.bfloat16) for k, v in load_file(str(fp32)).items()}
        save_file(sd, str(dest))

    print(f"{'SDXL upscaler':13s} -> {dest}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download + convert the SesquiLSR SDXL latent upscaler (bf16)."
    )
    parser.parse_args()
    download_sdxl_upscaler()


if __name__ == "__main__":
    main()
