"""Download the Z-Image-Turbo model artifacts into a local directory.

Everything comes from ``Comfy-Org/z_image_turbo`` (single-file, no auth): the
DiT, the Flux VAE, and the Qwen3-4B text encoder are each one safetensors file.
By default the bf16 DiT is fetched; pass ``--int8-convrot`` to fetch the
int8-convrot DiT instead (same repo). The tokenizer config files are vendored
under ``thenoise/dit/zimage/configs/tokenizer/`` (they carry the Qwen chat
template used by the caption encoder), so no tokenizer is downloaded; the
model's vendored config means no ``config.json`` is needed next to the text
encoder.

The Flux latent upscaler for the Z-Image upscale path is fetched from
``LoganBooker/SesquiLSR`` and converted fp32 -> bf16 into the package's committed
``thenoise/upscale/weights/`` directory (the format registry expects
``upscaler_flux.safetensors`` there).

  DiT (bf16)     split_files/diffusion_models/z_image_turbo_bf16.safetensors
  DiT (int8)     split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors  (--int8-convrot)
  VAE            split_files/vae/ae.safetensors                             (Flux VAE)
  Text encoder   split_files/text_encoders/qwen_3_4b.safetensors           (Qwen3-4B)
  Tokenizer      vendored in thenoise/dit/zimage/configs/tokenizer/       (not downloaded)
  Flux upscaler  thenoise/upscale/weights/upscaler_flux.safetensors        (bf16)

Usage:
    python scripts/download_zimage.py --out ./models/zimage
    python scripts/download_zimage.py --out ./models/zimage --int8-convrot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

COMFY_REPO = "Comfy-Org/z_image_turbo"

# Package weights dir the upscaler format registry reads from.
UPSCALER_WEIGHTS_DIR = (
    Path(__file__).resolve().parents[1] / "thenoise" / "upscale" / "weights"
)

# Single-file DiT / VAE / text encoder from the ComfyUI export.
SINGLE_FILES = [
    ("dit", COMFY_REPO, "split_files/diffusion_models/z_image_turbo_bf16.safetensors"),
    ("vae", COMFY_REPO, "split_files/vae/ae.safetensors"),
    ("text_encoder", COMFY_REPO, "split_files/text_encoders/qwen_3_4b.safetensors"),
]

#: The int8-convrot DiT lives in the same repo.
INT8_DIT = (
    "dit", COMFY_REPO, "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors"
)


def download_flux_upscaler() -> Path:
    """Download the SesquiLSR Flux upscaler and convert fp32 -> bf16 in place.

    The Flux VAE shares the 16-channel Sesqui architecture with Wan21, so only a
    dtype conversion is needed; the converted weights land next to the committed
    Wan21 weights where ``load_upscaler("flux")`` expects them.
    """
    import tempfile
    import urllib.request

    dest = UPSCALER_WEIGHTS_DIR / "upscaler_flux.safetensors"
    if dest.is_file():
        print(f"{'flux upscaler':14s} -> {dest} (already present)")
        return dest

    UPSCALER_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "upscaler_Flux.safetensors"
        print(f"Downloading Flux upscaler from SesquiLSR (fp32)")
        urllib.request.urlretrieve(SESQUI_FLUX_UPSCALER_URL, fp32)

        print(f"Converting {fp32.name} fp32 -> bf16")
        sd = {k: v.to(torch.bfloat16) for k, v in load_file(str(fp32)).items()}
        save_file(sd, str(dest))

    print(f"{'flux upscaler':14s} -> {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Z-Image-Turbo model artifacts")
    ap.add_argument("--out", default="./models/zimage", help="output directory")
    ap.add_argument(
        "--int8-convrot", action="store_true",
        help="download the int8-convrot DiT instead of bf16",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = SINGLE_FILES
    if args.int8_convrot:
        files = [INT8_DIT] + [f for f in SINGLE_FILES if f[0] != "dit"]

    for name, repo, path in files:
        dest = hf_hub_download(repo, path, local_dir=str(out))
        print(f"{name:14s} -> {dest}")

    download_flux_upscaler()


if __name__ == "__main__":
    main()
