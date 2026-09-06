"""Download Qwen-Image / Qwen-Image-Edit model artifacts into a local directory.

Everything comes from Comfy-Org (no auth). Both variants share the same Qwen2.5-VL-7B
text encoder and Qwen-Image VAE, which live in the image repo. The DiTs are split
across two repos and are versioned by date — only the **latest** dated bf16
checkpoints are fetched (the unversioned / older releases are skipped).

  Qwen-Image DiT (bf16)      Comfy-Org/Qwen-Image_ComfyUI
                             split_files/diffusion_models/qwen_image_2512_bf16.safetensors
  Qwen-Image DiT (int8)      obsxrver/ComfyUI-Native-INT8_ConvRot
                             diffusion_models/qwen-image-2512-int8-ConvRot.safetensors
                             (--int8-convrot)
  Qwen-Image-Edit DiT (bf16) Comfy-Org/Qwen-Image-Edit_ComfyUI
                             split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors
  Qwen-Image-Edit DiT (int8) split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors
                             (--int8-convrot)
  Text encoder               split_files/text_encoders/qwen_2.5_vl_7b.safetensors   (shared)
  VAE                        split_files/vae/qwen_image_vae.safetensors            (shared)

Pass ``--int8-convrot`` to swap **both** DiTs for their int8-convrot checkpoints
instead of bf16 (the int8 releases live in separate repos). The tokenizer config
files are vendored under ``thenoise/dit/qwen_image/configs/tokenizer/``, so no
tokenizer is downloaded; loading is fully offline (``local_files_only=True``).

Usage:
    python scripts/download_qwen_image.py --out ./models/qwen_image
    python scripts/download_qwen_image.py --out ./models/qwen_image --int8-convrot
    python scripts/download_qwen_image.py --out ./models/qwen_image --edit-only
    python scripts/download_qwen_image.py --out ./models/qwen_image --image-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

IMAGE_REPO = "Comfy-Org/Qwen-Image_ComfyUI"
EDIT_REPO = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
INT8_REPO = "obsxrver/ComfyUI-Native-INT8_ConvRot"

#: Latest dated bf16 DiTs (versioned checkpoints; older/unversioned releases skipped).
IMAGE_DIT_BF16 = (IMAGE_REPO, "split_files/diffusion_models/qwen_image_2512_bf16.safetensors")
EDIT_DIT_BF16 = (EDIT_REPO, "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors")

#: int8-convrot DiTs (separate repos; the plain image int8 release is on Civitai-mirroring HF).
IMAGE_DIT_INT8 = (INT8_REPO, "diffusion_models/qwen-image-2512-int8-ConvRot.safetensors")
EDIT_DIT_INT8 = (EDIT_REPO, "split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors")

#: Shared across both variants.
TEXT_ENCODER = (IMAGE_REPO, "split_files/text_encoders/qwen_2.5_vl_7b.safetensors")
VAE = (IMAGE_REPO, "split_files/vae/qwen_image_vae.safetensors")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Qwen-Image / Qwen-Image-Edit model artifacts")
    ap.add_argument("--out", default="./models/qwen_image", help="output directory")
    ap.add_argument(
        "--image-only", action="store_true",
        help="download only the Qwen-Image DiT (plus shared text encoder and VAE)",
    )
    ap.add_argument(
        "--edit-only", action="store_true",
        help="download only the Qwen-Image-Edit DiT (plus shared text encoder and VAE)",
    )
    ap.add_argument(
        "--int8-convrot", action="store_true",
        help="download the int8-convrot DiTs instead of bf16",
    )
    args = ap.parse_args()

    if args.image_only and args.edit_only:
        ap.error("--image-only and --edit-only are mutually exclusive")

    jobs = []
    if not args.edit_only:
        image_dit = IMAGE_DIT_INT8 if args.int8_convrot else IMAGE_DIT_BF16
        jobs.append(("dit (image)", *image_dit))
    if not args.image_only:
        edit_dit = EDIT_DIT_INT8 if args.int8_convrot else EDIT_DIT_BF16
        jobs.append(("dit (edit)", *edit_dit))
    jobs += [("text_encoder", *TEXT_ENCODER), ("vae", *VAE)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, repo, path in jobs:
        dest = hf_hub_download(repo, path, local_dir=str(out))
        print(f"{name:13s} -> {dest}")


if __name__ == "__main__":
    main()
