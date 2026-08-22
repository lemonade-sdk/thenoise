"""Download the Flux.2 (Flux Klein) model artifacts into a local directory.

The DiT and text encoder use single-file bf16 checkpoints from Comfy-Org where
available (the project's preferred format).

  Variant     DiT                                    Text encoder
  ---------   ------------------------------------   --------------
  4b          4b (distilled)                         Qwen3-4B
  4b-base     base (CFG, 50 steps)                   Qwen3-4B
  9b          9b (distilled)                         Qwen3-8B
  9b-base     base (CFG, 50 steps)                   Qwen3-8B

Usage:
    python scripts/download_klein.py --out ./models/klein --variant 4b
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

#: Comfy-Org single-file text encoder / VAE repos (one per DiT size).
COMFY_4B = "Comfy-Org/vae-text-encorder-for-flux-klein-4b"
COMFY_9B = "Comfy-Org/vae-text-encorder-for-flux-klein-9b"

#: DiT size -> (text encoder repo, text encoder file).
TEXT_ENCODERS = {
    "4b": (COMFY_4B, "split_files/text_encoders/qwen_3_4b.safetensors"),
    "9b": (COMFY_9B, "split_files/text_encoders/qwen_3_8b.safetensors"),
}

#: Shared Flux.2 VAE (both sizes use the same file).
VAE = (COMFY_4B, "split_files/vae/flux2-vae.safetensors")

#: variant -> (repo, DiT path).
DITS = {
    "4b": (COMFY_4B, "split_files/diffusion_models/flux-2-klein-4b.safetensors"),
    "4b-base": (COMFY_4B, "split_files/diffusion_models/flux-2-klein-base-4b.safetensors"),
    "9b": ("unsloth/FLUX.2-klein-9B", "flux-2-klein-9b.safetensors"),
    "9b-base": ("unsloth/FLUX.2-klein-base-9B", "flux-2-klein-base-9b.safetensors"),
}

#: variant -> DiT size key used to pick the text encoder.
_DIT_SIZE = {"4b": "4b", "4b-base": "4b", "9b": "9b", "9b-base": "9b"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Flux.2 (Flux Klein) model artifacts")
    ap.add_argument("--out", default="./models/klein", help="output directory")
    ap.add_argument(
        "--variant",
        choices=sorted(DITS),
        default="4b",
        help="model variant: 4b / 4b-base (distilled / base 4B), 9b / 9b-base (distilled / base 9B)",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    size = _DIT_SIZE[args.variant]
    te_repo, te_path = TEXT_ENCODERS[size]
    vae_repo, vae_path = VAE
    dit_repo, dit_path = DITS[args.variant]

    jobs = [
        ("dit", dit_repo, dit_path),
        ("vae", vae_repo, vae_path),
        ("text_encoder", te_repo, te_path),
    ]
    for name, repo, path in jobs:
        dest = hf_hub_download(repo, path, local_dir=str(out))
        print(f"{name:13s} -> {dest}")

if __name__ == "__main__":
    main()
