"""Download the Flux.2 (Flux Klein) model artifacts into a local directory.

The DiT and text encoder use single-file bf16 checkpoints from Comfy-Org where
available (the project's preferred format).

  Variant     DiT                                    Text encoder
  ---------   ------------------------------------   --------------
  4b          4b (distilled)                         Qwen3-4B
  4b-base     base (CFG, 50 steps)                   Qwen3-4B
  9b          9b (distilled)                         Qwen3-8B
  9b-base     base (CFG, 50 steps)                   Qwen3-8B

Pass ``--int8-convrot`` to fetch the int8-convrot DiT instead of bf16. The 4B
DiT comes from ``wraps/FLUX.2-klein-4B-INT8-ConvRot-ComfyUI``; the 9B DiT is
only published on Civitai and is downloaded from there directly. Base variants
have no int8-convrot release and are rejected.

Usage:
    python scripts/download_klein.py --out ./models/klein --variant 4b
    python scripts/download_klein.py --out ./models/klein --variant 4b --int8-convrot
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

#: int8-convrot 4B DiT (HuggingFace).
INT8_4B = ("wraps/FLUX.2-klein-4B-INT8-ConvRot-ComfyUI", "flux-2-klein-4b-int8-convrot.safetensors")

#: int8-convrot 9B DiT (Civitai direct download URL + human link).
INT8_9B_URL = "https://civitai.com/api/download/models/3079984?fileId=2959248"
INT8_9B_LINK = "https://civitai.com/models/2738890/flux-2-klein-9b-int8"


def download_civitai(url: str, dest: Path) -> None:
    """Download a model from Civitai, printing the web link if login is required."""
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlretrieve(url, dest)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        if getattr(e, "code", 0) in (401, 403):
            print("Civitai requires login to download this model.")
            print(f"Open {INT8_9B_LINK} in your browser and download the file manually.")
            raise SystemExit(1) from e
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Flux.2 (Flux Klein) model artifacts")
    ap.add_argument("--out", default="./models/klein", help="output directory")
    ap.add_argument(
        "--variant",
        choices=sorted(DITS),
        default="4b",
        help="model variant: 4b / 4b-base (distilled / base 4B), 9b / 9b-base (distilled / base 9B)",
    )
    ap.add_argument(
        "--int8-convrot", action="store_true",
        help="download the int8-convrot DiT instead of bf16",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    size = _DIT_SIZE[args.variant]
    te_repo, te_path = TEXT_ENCODERS[size]
    vae_repo, vae_path = VAE

    if args.int8_convrot:
        if args.variant in ("4b-base", "9b-base"):
            ap.error(f"--int8-convrot is not available for variant {args.variant}")

        if args.variant == "4b":
            dit_repo, dit_path = INT8_4B
            jobs = [("dit", dit_repo, dit_path)]
        else:  # 9b
            jobs = []
            dest = out / "flux-2-klein-9b-int8-convrot.safetensors"
            print("Downloading 9B int8-convrot DiT from Civitai")
            download_civitai(INT8_9B_URL, dest)
            print(f"{'dit':13s} -> {dest}")
    else:
        dit_repo, dit_path = DITS[args.variant]
        jobs = [("dit", dit_repo, dit_path)]

    jobs += [("vae", vae_repo, vae_path), ("text_encoder", te_repo, te_path)]
    for name, repo, path in jobs:
        dest = hf_hub_download(repo, path, local_dir=str(out))
        print(f"{name:13s} -> {dest}")


if __name__ == "__main__":
    main()
