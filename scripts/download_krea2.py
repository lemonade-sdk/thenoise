"""Download the Krea 2 model artifacts into a local directory.

Uses huggingface_hub. All files come from `Comfy-Org/Krea-2`, which needs no
authentication. By default we fetch the bf16 variants:

  DiT (Turbo)  diffusion_models/krea2_turbo_bf16.safetensors
  DiT (RAW)    diffusion_models/krea2_raw_bf16.safetensors   (--include-raw)
  VAE          vae/qwen_image_vae.safetensors
  Text encoder text_encoders/qwen3vl_4b_bf16.safetensors

Pass `--int8-convrot` to swap the DiT(s) for the int8-convrot checkpoints
instead (same repo):

  DiT (Turbo)  diffusion_models/krea2_turbo_int8_convrot.safetensors
  DiT (RAW)    diffusion_models/krea2_raw_int8_convrot.safetensors

The Qwen3-VL tokenizer is fetched automatically (by repo id) at first
text-encoder load.

Usage:
    python scripts/download_krea2.py --out ./models/krea2
    python scripts/download_krea2.py --out ./models/krea2 --include-raw
    python scripts/download_krea2.py --out ./models/krea2 --int8-convrot
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "Comfy-Org/Krea-2"

ARTIFACTS = {
    "dit_turbo": "diffusion_models/krea2_turbo_bf16.safetensors",
    "vae": "vae/qwen_image_vae.safetensors",
    "text_encoder": "text_encoders/qwen3vl_4b_bf16.safetensors",
    "dit_raw": "diffusion_models/krea2_raw_bf16.safetensors",
}

INT8_ARTIFACTS = {
    "dit_turbo": "diffusion_models/krea2_turbo_int8_convrot.safetensors",
    "vae": "vae/qwen_image_vae.safetensors",
    "text_encoder": "text_encoders/qwen3vl_4b_bf16.safetensors",
    "dit_raw": "diffusion_models/krea2_raw_int8_convrot.safetensors",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Krea 2 model artifacts")
    ap.add_argument("--out", default="./models/krea2", help="output directory")
    ap.add_argument("--include-raw", action="store_true", help="also download the RAW DiT")
    ap.add_argument(
        "--int8-convrot", action="store_true",
        help="download int8-convrot DiTs instead of bf16",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    artifacts = INT8_ARTIFACTS if args.int8_convrot else ARTIFACTS
    items = list(artifacts.items())
    if not args.include_raw:
        items = [i for i in items if i[0] != "dit_raw"]

    for name, path in items:
        dest = hf_hub_download(REPO, path, local_dir=str(out))
        print(f"{name:14s} -> {dest}")


if __name__ == "__main__":
    main()
