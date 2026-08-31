"""Download and split the Illustrious-XL v2.0 model artifacts.

The official ``OnomaAIResearch/Illustrious-XL-v2.0`` repo ships a single combined
6.94GB ``.safetensors`` (SDXL full model: UNet + VAE + both CLIP text encoders
concatenated). thenoise expects separate ``--dit`` / ``--vae`` /
``--text-encoder`` files, so this downloads the combined file and splits it once
into its components:

  UNet          model.diffusion_model.*   -> split_files/diffusion_models/sdxl_unet.safetensors
  VAE (decode)  first_stage_model.decoder / post_quant_conv
                                            -> split_files/vae/sdxl_vae.safetensors
  CLIP-L        conditioner.embedders.0.transformer.*
                                            -> clip_l.  in split_files/text_encoders/clip_l_g.safetensors
  CLIP-G        conditioner.embedders.1.model.*
                                            -> clip_g.  in split_files/text_encoders/clip_l_g.safetensors
  Tokenizer     openai/clip-vit-large-patch14 -> <out>/tokenizer/  (CLIP BPE)

The two text encoders are combined into one ``clip_l_g.safetensors`` (prefixed
``clip_l.`` / ``clip_g.``) so the CLI's single ``--text-encoder`` path carries
both. The CLIP tokenizer is fetched into ``<out>/tokenizer/`` and loaded offline
next to the text encoders (falling back to the package's vendored copy).

Usage:
    python scripts/download_sdxl.py --out ./models/sdxl
    python scripts/download_sdxl.py --out ./models/sdxl --hyper-sd
      (also fetches the optional Hyper-SD 8-step LoRA into ./models/sdxl/lora)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

REPO = "OnomaAIResearch/Illustrious-XL-v2.0"
COMBINED_FILENAME = "Illustrious-XL-v2.0.safetensors"

#: Optional Hyper-SD 8-step CFG-preserved SDXL LoRA (fast testing; ~750MB).
#: Downloaded only with ``--hyper-sd``; see ``scripts/download_hyper_sd.py``.
HYPER_SD_REPO = "ByteDance/Hyper-SD"
HYPER_SD_8STEP_CFG = "Hyper-SDXL-8steps-CFG-lora.safetensors"

CLIP_TOKENIZER_REPO = "openai/clip-vit-large-patch14"
CLIP_TOKENIZER_FILES = [
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

#: Component key prefixes within the combined file (Stability-AI SDXL layout).
UNET_PREFIX = "model.diffusion_model."
VAE_PREFIX = "first_stage_model."
CLIP_L_PREFIX = "conditioner.embedders.0.transformer."
CLIP_G_PREFIX = "conditioner.embedders.1.model."

#: Prediction-type marker tensors (``v_pred``, ``edm_mean``/``edm_std``, ...) that
#: some SDXL checkpoints carry at the top level. Preserved into the dit split so
#: ``SdxlModel`` can autodetect the prediction type. Mirrors ComfyUI's
#: ``SDXL.model_type`` markers.
PREDICTION_MARKERS = frozenset(
    {
        "v_pred",
        "ztsnr",
        "edm_mean",
        "edm_std",
        "edm_vpred.sigma_max",
        "edm_vpred.sigma_min",
    }
)


def _partition(sd: dict) -> dict[str, dict]:
    """Partition the combined state dict into the four component groups."""
    unet = {k[len(UNET_PREFIX):]: v for k, v in sd.items() if k.startswith(UNET_PREFIX)}
    # Keep the prediction-type markers (bare keys) alongside the UNet weights.
    unet.update({k: v for k, v in sd.items() if k in PREDICTION_MARKERS})
    vae = {
        k[len(VAE_PREFIX):]: v
        for k, v in sd.items()
        if k.startswith(VAE_PREFIX) and (k.startswith(VAE_PREFIX + "decoder.") or "post_quant_conv" in k)
    }
    clip_l = {k[len(CLIP_L_PREFIX):]: v for k, v in sd.items() if k.startswith(CLIP_L_PREFIX)}
    clip_g = {k[len(CLIP_G_PREFIX):]: v for k, v in sd.items() if k.startswith(CLIP_G_PREFIX)}
    for name, part in [
        ("unet", unet),
        ("vae", vae),
        ("clip_l", clip_l),
        ("clip_g", clip_g),
    ]:
        if not part:
            raise ValueError(
                f"partition {name!r} is empty: upstream layout for {REPO} may have "
                "changed (expected the Stability-AI SDXL key prefixes)"
            )
    return unet, vae, clip_l, clip_g


def download_and_split(out: Path, keep_combined: bool = False, hyper_sd: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s/%s", REPO, COMBINED_FILENAME)
    combined = hf_hub_download(REPO, COMBINED_FILENAME, local_dir=str(out / "combined"))
    combined = Path(combined)

    print(f"Loading combined checkpoint ({combined.stat().st_size / 1e9:.1f} GB) ...")
    sd = load_file(str(combined))
    unet, vae, clip_l, clip_g = _partition(sd)
    del sd

    for name, part in [
        ("unet", unet),
        ("vae", vae),
        ("clip_l", clip_l),
        ("clip_g", clip_g),
    ]:
        print(f"  {name:8s}: {len(part)} keys")

    dit_dir = out / "split_files" / "diffusion_models"
    dit_dir.mkdir(parents=True, exist_ok=True)
    _save(dit_dir / "sdxl_unet.safetensors", unet)

    vae_dir = out / "split_files" / "vae"
    vae_dir.mkdir(parents=True, exist_ok=True)
    _save(vae_dir / "sdxl_vae.safetensors", vae)

    te_dir = out / "split_files" / "text_encoders"
    te_dir.mkdir(parents=True, exist_ok=True)
    combined_te = {**{f"clip_l.{k}": v for k, v in clip_l.items()},
                   **{f"clip_g.{k}": v for k, v in clip_g.items()}}
    _save(te_dir / "clip_l_g.safetensors", combined_te)

    tok_dir = out / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    for f in CLIP_TOKENIZER_FILES:
        dest = hf_hub_download(CLIP_TOKENIZER_REPO, f, local_dir=str(tok_dir))
        print(f"{'tokenizer':14s} -> {dest}")

    if not keep_combined:
        _remove_combined(combined, out / "combined")

    if hyper_sd:
        lora_dir = out / "lora"
        lora_dir.mkdir(parents=True, exist_ok=True)
        dest = hf_hub_download(HYPER_SD_REPO, HYPER_SD_8STEP_CFG, local_dir=str(lora_dir))
        print(f"{'hyper-sd':12s} -> {dest}")

    print("\nDone. Point thenoise at:")
    print(f"  --dit            {dit_dir / 'sdxl_unet.safetensors'}")
    print(f"  --vae            {vae_dir / 'sdxl_vae.safetensors'}")
    print(f"  --text-encoder   {te_dir / 'clip_l_g.safetensors'}")
    print(f"  --lora-dir       (optional)")


def _save(path: Path, sd: dict) -> None:
    print(f"{path.name:22s} -> {path}")
    save_file(sd, str(path))


def _remove_combined(combined: Path, combined_dir: Path) -> None:
    """Delete the 6.94GB combined file (and its now-empty dir) after splitting."""
    try:
        combined.unlink()
        print(f"Removed combined checkpoint {combined}")
    except OSError as e:
        print(f"Warning: could not remove {combined}: {e}")
    try:
        combined_dir.rmdir()  # only succeeds if empty
    except OSError:
        pass


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="Download + split Illustrious-XL v2.0")
    ap.add_argument("--out", default="./models/sdxl", help="output directory")
    ap.add_argument(
        "--keep-combined-safetensors",
        action="store_true",
        help="keep the downloaded combined single-file safetensors after splitting",
    )
    ap.add_argument(
        "--hyper-sd",
        action="store_true",
        help="also download the Hyper-SD 8-step CFG LoRA into <out>/lora/ "
        "(optional, ~750MB; enables fast 8-step testing)",
    )
    args = ap.parse_args()
    download_and_split(
        Path(args.out),
        keep_combined=args.keep_combined_safetensors,
        hyper_sd=args.hyper_sd,
    )


if __name__ == "__main__":
    main()
