"""Download the Anima (Cosmos-Predict2 2B text2image) model artifacts.

Source: https://huggingface.co/circlestone-labs/Anima

  DiT      split_files/diffusion_models/anima-<variant>.safetensors
           (variants: base-v1.0, aesthetic-v1.1, turbo-v1.0, ...)
  Text enc split_files/text_encoders/qwen_3_06b_base.safetensors
  VAE      split_files/vae/qwen_image_vae.safetensors

The Qwen3 / T5 tokenizer configs are packaged under ``thenoise/dit/anima/configs/``, so
they are not downloaded.

Pass ``--int8-convrot`` to fetch the int8-convrot DiT instead. The DiT lives in
``Bedovyy/Anima-INT8`` (``anima-<variant>-int8convrot.safetensors``); the text
encoder and VAE still come from ``circlestone-labs/Anima``.

Usage:
    python scripts/download_anima.py --out ./models/anima
    python scripts/download_anima.py --out ./models/anima --variant aesthetic-v1.1
    python scripts/download_anima.py --out ./models/anima --int8-convrot
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "circlestone-labs/Anima"
INT8_REPO = "Bedovyy/Anima-INT8"
DEFAULT_VARIANT = "turbo-v1.0"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Anima model artifacts")
    ap.add_argument("--out", default="./models/anima", help="output directory")
    ap.add_argument(
        "--variant", default=DEFAULT_VARIANT,
        help=f"DiT variant to download (default: {DEFAULT_VARIANT})",
    )
    ap.add_argument(
        "--int8-convrot", action="store_true",
        help="download the int8-convrot DiT instead of bf16",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.int8_convrot:
        dit = (INT8_REPO, f"anima-{args.variant}-int8convrot.safetensors")
    else:
        dit = (REPO, f"split_files/diffusion_models/anima-{args.variant}.safetensors")

    artifacts = [
        ("dit", *dit),
        ("text_encoder", REPO, "split_files/text_encoders/qwen_3_06b_base.safetensors"),
        ("vae", REPO, "split_files/vae/qwen_image_vae.safetensors"),
    ]
    for name, repo, path in artifacts:
        dest = hf_hub_download(repo, path, local_dir=str(out))
        print(f"{name:14s} -> {dest}")


if __name__ == "__main__":
    main()
