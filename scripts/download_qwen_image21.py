"""Download the Qwen-Image 2.1 model artifacts into a local directory.

Everything comes from `Comfy-Org/Qwen-Image-2.1` (no auth). Qwen-Image 2.1 ships
**one** DiT that does both text-to-image and editing (unlike Qwen-Image 1, whose
image and edit variants are separate checkpoints), a full Qwen3-VL-8B conditioner
(LM *and* vision tower — an edit's reference goes into both the encoder and the DiT)
and a Wan 2.2-layout 64-channel/16x RGBA VAE:

  Artifact      Path                                                          Size
  ------------  ------------------------------------------------------------  -------
  DiT (bf16)    diffusion_models/qwen_image_2.1_bf16.safetensors              14.2 GB
  DiT (int8)    diffusion_models/qwen_image_2.1_int8_convrot.safetensors       7.3 GB
  TE (bf16)     text_encoders/qwen3vl_8b_bf16.safetensors                     17.5 GB
  TE (int8)     text_encoders/qwen3vl_8b_int8_convrot.safetensors              9.4 GB
  VAE           vae/qwen_image_2.1_vae_bf16.safetensors                        0.7 GB

``--int8-convrot`` swaps both bf16 rows for their int8-convrot counterparts
(~17 GB total instead of ~32 GB). The VAE is never quantized.

The repo also publishes a w4a8 build of the Qwen3-VL encoder and two Qwen3.5-9B
"prompt extend" encoders (`qwen3.5_9b_qwen_image_2.1_pe_t2i/_pe_i2i`); neither is
fetched — the loader has no w4a8 support yet, and prompt extension is not part of
the pipeline.

The tokenizer/processor configs are vendored under
``thenoise/utils/text_encoder/configs/qwen25_tokenizer/`` (plus
``thenoise/utils/qwen_configs.py`` for the model config), so no tokenizer is
downloaded and loading is fully offline (``local_files_only=True``).

Usage:
    python scripts/download_qwen_image21.py --out ./models/qwen_image21
    python scripts/download_qwen_image21.py --out ./models/qwen_image21 --int8-convrot
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "Comfy-Org/Qwen-Image-2.1"

#: name -> path, bf16 (default).
ARTIFACTS = {
    "dit": "diffusion_models/qwen_image_2.1_bf16.safetensors",
    "vae": "vae/qwen_image_2.1_vae_bf16.safetensors",
    "text_encoder": "text_encoders/qwen3vl_8b_bf16.safetensors",
}

#: int8-convrot variants (`--int8-convrot`). Same VAE: it is never quantized.
INT8_ARTIFACTS = {
    "dit": "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
    "vae": "vae/qwen_image_2.1_vae_bf16.safetensors",
    "text_encoder": "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Qwen-Image 2.1 model artifacts")
    ap.add_argument("--out", default="./models/qwen_image21", help="output directory")
    ap.add_argument(
        "--int8-convrot", action="store_true",
        help="download the int8-convrot DiT and text encoder instead of bf16",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    artifacts = INT8_ARTIFACTS if args.int8_convrot else ARTIFACTS
    for name, path in artifacts.items():
        dest = hf_hub_download(REPO, path, local_dir=str(out))
        print(f"{name:13s} -> {dest}")


if __name__ == "__main__":
    main()
