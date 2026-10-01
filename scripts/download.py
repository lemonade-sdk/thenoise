"""Download model artifacts for TheNoise.

One script for every model TheNoise can run. Pick the model with ``--model``;
per-model options (variants, int8-convrot, ...) are enabled for the models that
support them. Files are fetched with ``huggingface_hub`` (install the ``scripts``
extra: ``uv pip install -e ".[scripts]"``) and kept in the same directory layout
as before, so existing ``--dit`` / ``--vae`` / ``--text-encoder`` paths keep
working.

Models:
  anima           Anima (Cosmos-Predict2 2B) text-to-image
                  https://huggingface.co/circlestone-labs/Anima
  krea2           Krea 2 Turbo (bf16 / raw)
                  https://huggingface.co/Comfy-Org/Krea-2
  zimage          Z-Image Turbo (distilled, 8-step) or Z-Image (base)
                  https://huggingface.co/Comfy-Org/z_image_turbo
                  https://huggingface.co/Comfy-Org/z_image
  klein           Flux.2 Klein 4B / 9B (distilled or base)
  ming-image      Ming-Image 0.1 (Design)
                  https://huggingface.co/Comfy-Org/Ming-Image
  qwen-image      Qwen-Image / Qwen-Image-Edit (latest dated checkpoints)
  qwen-image-2.1  Qwen-Image 2.1 (t2i + edit in one DiT)
                  https://huggingface.co/Comfy-Org/Qwen-Image-2.1
  mage-flow       Microsoft Mage-Flow (t2i + edit; base / RL / turbo)
                  https://huggingface.co/Comfy-Org/Mage-Flow
  esrgan          Real-ESRGAN x4 pixel upscaler (optional, for upscaling)
                  https://huggingface.co/Comfy-Org/Real-ESRGAN_repackaged

Tokenizer/processor configs are vendored inside the ``thenoise`` package, so no
tokenizer is ever downloaded; loading is fully offline.

Usage:
    python scripts/download.py --model anima
    python scripts/download.py --model anima --variant aesthetic-v1.1
    python scripts/download.py --model krea2 --int8-convrot
    python scripts/download.py --model krea2 --include-raw
    python scripts/download.py --model zimage
    python scripts/download.py --model zimage --variant base
    python scripts/download.py --model klein --variant 9b
    python scripts/download.py --model klein --variant 4b --int8-convrot
    python scripts/download.py --model ming-image
    python scripts/download.py --model ming-image --int8-convrot
    python scripts/download.py --model qwen-image --edit-only
    python scripts/download.py --model qwen-image-2.1
    python scripts/download.py --model mage-flow
    python scripts/download.py --model mage-flow --variant base
    python scripts/download.py --model mage-flow --int8-convrot --edit-only
    python scripts/download.py --model esrgan
    python scripts/download.py --model krea2 --dry-run
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download

#: An artifact: (label, huggingface repo, in-repo file path).
Artifact = tuple[str, str, str]
JobsFn = Callable[["argparse.Namespace"], list[Artifact]]


# ---------------------------------------------------------------------------
# Artifact registries (repo, in-repo path). Layouts match the per-model
# download scripts this file replaces, byte for byte.
# ---------------------------------------------------------------------------

def _anima_jobs(args: argparse.Namespace) -> list[Artifact]:
    """Anima: DiT (per variant, bf16 or int8-convrot) + Qwen3-06B TE + Qwen VAE."""
    repo = "circlestone-labs/Anima"
    if args.int8_convrot:
        # The int8 DiT is published at the repo root (no split_files prefix).
        dit = ("Bedovyy/Anima-INT8", f"anima-{args.variant}-int8convrot.safetensors")
    else:
        dit = (repo, f"split_files/diffusion_models/anima-{args.variant}.safetensors")
    return [
        ("dit", *dit),
        ("text_encoder", repo, "split_files/text_encoders/qwen_3_06b_base.safetensors"),
        ("vae", repo, "split_files/vae/qwen_image_vae.safetensors"),
    ]


def _krea2_jobs(args: argparse.Namespace) -> list[Artifact]:
    """Krea 2: Turbo DiT (+ optional RAW DiT) + Qwen VAE + Qwen3-VL-4B TE."""
    repo = "Comfy-Org/Krea-2"
    suffix = "int8_convrot" if args.int8_convrot else "bf16"
    jobs = [
        ("dit (turbo)", repo, f"diffusion_models/krea2_turbo_{suffix}.safetensors"),
    ]
    if args.include_raw:
        jobs.append(("dit (raw)", repo, f"diffusion_models/krea2_raw_{suffix}.safetensors"))
    jobs += [
        ("vae", repo, "vae/qwen_image_vae.safetensors"),
        ("text_encoder", repo, "text_encoders/qwen3vl_4b_bf16.safetensors"),
    ]
    return jobs


def _zimage_jobs(args: argparse.Namespace) -> list[Artifact]:
    """Z-Image: Turbo (distilled) or base, from the matching Comfy-Org repo."""
    repo = "Comfy-Org/z_image_turbo" if args.variant == "turbo" else "Comfy-Org/z_image"
    if args.variant == "turbo":
        name = "z_image_turbo"
    else:
        name = "z_image"
    if args.int8_convrot:
        dit = f"split_files/diffusion_models/{name}_int8_convrot.safetensors"
    else:
        dit = f"split_files/diffusion_models/{name}_bf16.safetensors"
    return [
        ("dit", repo, dit),
        ("vae", repo, "split_files/vae/ae.safetensors"),
        ("text_encoder", repo, "split_files/text_encoders/qwen_3_4b.safetensors"),
    ]


#: Flux.2 Klein: (variant) -> DiT (repo, path).
_KLEIN_DITS = {
    "4b": ("Comfy-Org/vae-text-encorder-for-flux-klein-4b",
           "split_files/diffusion_models/flux-2-klein-4b.safetensors"),
    "4b-base": ("Comfy-Org/vae-text-encorder-for-flux-klein-4b",
                "split_files/diffusion_models/flux-2-klein-base-4b.safetensors"),
    "9b": ("unsloth/FLUX.2-klein-9B", "flux-2-klein-9b.safetensors"),
    "9b-base": ("unsloth/FLUX.2-klein-base-9B", "flux-2-klein-base-9b.safetensors"),
}
#: Distilled variants with an int8-convrot release (base variants: none).
_KLEIN_INT8_DITS = {
    "4b": ("wraps/FLUX.2-klein-4B-INT8-ConvRot-ComfyUI",
           "flux-2-klein-4b-int8-convrot.safetensors"),
    "9b": ("obsxrver/ComfyUI-Native-INT8_ConvRot",
           "diffusion_models/flux-2-klein-9b_int8_convrot.safetensors"),
}
#: DiT size -> (text encoder repo, path).
_KLEIN_TEXT_ENCODERS = {
    "4b": ("Comfy-Org/vae-text-encorder-for-flux-klein-4b",
           "split_files/text_encoders/qwen_3_4b.safetensors"),
    "9b": ("Comfy-Org/vae-text-encorder-for-flux-klein-9b",
           "split_files/text_encoders/qwen_3_8b.safetensors"),
}
#: Shared Flux.2 VAE (both sizes use the same file).
_KLEIN_VAE = ("Comfy-Org/vae-text-encorder-for-flux-klein-4b",
              "split_files/vae/flux2-vae.safetensors")


def _klein_jobs(args: argparse.Namespace) -> list[Artifact]:
    """Flux.2 Klein: DiT (per variant, bf16 or int8-convrot) + Qwen VAE + TE."""
    if args.int8_convrot:
        if args.variant not in _KLEIN_INT8_DITS:
            raise SystemExit(
                f"error: --int8-convrot is not available for variant '{args.variant}' "
                "(no int8-convrot release for base variants)"
            )
        dit_repo, dit_path = _KLEIN_INT8_DITS[args.variant]
    else:
        dit_repo, dit_path = _KLEIN_DITS[args.variant]
    size = "4b" if args.variant.startswith("4b") else "9b"
    te_repo, te_path = _KLEIN_TEXT_ENCODERS[size]
    return [
        ("dit", dit_repo, dit_path),
        ("vae", *_KLEIN_VAE),
        ("text_encoder", te_repo, te_path),
    ]


def _ming_image_jobs(args: argparse.Namespace) -> list[Artifact]:
    """Ming-Image 0.1: Design DiT + Ling-mini-2.0 TE + VAE."""
    repo = "Comfy-Org/Ming-Image"
    suffix = "int8_convrot" if args.int8_convrot else "bf16"
    return [
        ("dit", repo, f"diffusion_models/ming_image_0.1_design_{suffix}.safetensors"),
        ("text_encoder", repo, f"text_encoders/ming_image_0.1_ling_mini_2.0_{suffix}.safetensors"),
        ("vae", repo, "vae/ming_image_vae_bf16.safetensors"),
    ]


def _qwen_image_jobs(args: argparse.Namespace) -> list[Artifact]:
    """Qwen-Image / Qwen-Image-Edit: latest dated DiT(s) + shared TE + VAE."""
    image_repo = "Comfy-Org/Qwen-Image_ComfyUI"
    edit_repo = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
    int8_repo = "obsxrver/ComfyUI-Native-INT8_ConvRot"

    image_dit = (
        (int8_repo, "diffusion_models/qwen-image-2512-int8-ConvRot.safetensors")
        if args.int8_convrot
        else (image_repo, "split_files/diffusion_models/qwen_image_2512_bf16.safetensors")
    )
    edit_dit = (
        (edit_repo, "split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors")
        if args.int8_convrot
        else (edit_repo, "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors")
    )

    jobs: list[Artifact] = []
    if not args.edit_only:
        jobs.append(("dit (image)", *image_dit))
    if not args.image_only:
        jobs.append(("dit (edit)", *edit_dit))
    jobs += [
        ("text_encoder", image_repo, "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"),
        ("vae", image_repo, "split_files/vae/qwen_image_vae.safetensors"),
    ]
    return jobs


def _qwen_image_21_jobs(args: argparse.Namespace) -> list[Artifact]:
    """Qwen-Image 2.1: DiT + Qwen3-VL-8B TE + VAE (VAE is never quantized)."""
    repo = "Comfy-Org/Qwen-Image-2.1"
    if args.int8_convrot:
        return [
            ("dit", repo, "diffusion_models/qwen_image_2.1_int8_convrot.safetensors"),
            ("vae", repo, "vae/qwen_image_2.1_vae_bf16.safetensors"),
            ("text_encoder", repo, "text_encoders/qwen3vl_8b_int8_convrot.safetensors"),
        ]
    return [
        ("dit", repo, "diffusion_models/qwen_image_2.1_bf16.safetensors"),
        ("vae", repo, "vae/qwen_image_2.1_vae_bf16.safetensors"),
        ("text_encoder", repo, "text_encoders/qwen3vl_8b_bf16.safetensors"),
    ]


#: Mage-Flow variant -> (t2i DiT file stem, edit DiT file stem, int8-convrot release).
#: The un-distilled ``base`` checkpoints were never published in int8-convrot.
_MAGE_FLOW_REPO = "Comfy-Org/Mage-Flow"
_MAGE_FLOW_VARIANTS = {
    "base": ("mage_flow_base", "mage_flow_edit_base", False),
    "rl": ("mage_flow", "mage_flow_edit", True),
    "turbo": ("mage_flow_turbo", "mage_flow_edit_turbo", True),
}


def _mage_flow_jobs(args: argparse.Namespace) -> list[Artifact]:
    """Mage-Flow: t2i and/or edit DiT + shared Qwen3-VL-4B TE + Mage VAE."""
    stem, edit_stem, has_int8 = _MAGE_FLOW_VARIANTS[args.variant]
    if args.int8_convrot and not has_int8:
        raise SystemExit(
            f"error: --int8-convrot is not available for variant '{args.variant}' "
            "(no int8-convrot release for the base checkpoints)"
        )
    suffix = "int8_convrot" if args.int8_convrot else "bf16"

    jobs: list[Artifact] = []
    if not args.edit_only:
        jobs.append(("dit (t2i)", _MAGE_FLOW_REPO,
                     f"diffusion_models/{stem}_{suffix}.safetensors"))
    if not args.image_only:
        jobs.append(("dit (edit)", _MAGE_FLOW_REPO,
                     f"diffusion_models/{edit_stem}_{suffix}.safetensors"))
    jobs += [
        ("text_encoder", _MAGE_FLOW_REPO, "text_encoders/qwen3vl_4b_bf16.safetensors"),
        ("vae", _MAGE_FLOW_REPO, "vae/mage_flow_vae_bf16.safetensors"),
    ]
    return jobs


def _esrgan_jobs(args: argparse.Namespace) -> list[Artifact]:
    """Real-ESRGAN x4 pixel upscaler (no DiT/TE/VAE)."""
    return [
        ("pixel_upscaler", "Comfy-Org/Real-ESRGAN_repackaged",
         "RealESRGAN_x4plus.safetensors"),
    ]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    default_out: str
    help: str
    jobs: JobsFn
    has_int8: bool = True
    #: Whether --variant is accepted. If so, variant_choices=None means free-form.
    variant_supported: bool = False
    variant_choices: tuple[str, ...] | None = None
    default_variant: str | None = None


MODELS: dict[str, ModelSpec] = {
    "anima": ModelSpec(
        "anima", "./models/anima",
        "Anima (Cosmos-Predict2 2B) text-to-image",
        _anima_jobs,
        variant_supported=True,  # free-form variant name (see --variant help)
        default_variant="turbo-v1.0",
    ),
    "krea2": ModelSpec(
        "krea2", "./models/krea2",
        "Krea 2 Turbo (bf16 / raw)",
        _krea2_jobs,
    ),
    "zimage": ModelSpec(
        "zimage", "./models/zimage",
        "Z-Image (distilled Turbo 8-step, or base)",
        _zimage_jobs,
        variant_supported=True,
        variant_choices=("turbo", "base"),
        default_variant="turbo",
    ),
    "klein": ModelSpec(
        "klein", "./models/klein",
        "Flux.2 Klein 4B / 9B (distilled or base)",
        _klein_jobs,
        variant_supported=True,
        variant_choices=("4b", "4b-base", "9b", "9b-base"),
        default_variant="4b",
    ),
    "ming-image": ModelSpec(
        "ming-image", "./models/ming_image",
        "Ming-Image 0.1 (Design)",
        _ming_image_jobs,
    ),
    "qwen-image": ModelSpec(
        "qwen-image", "./models/qwen_image",
        "Qwen-Image / Qwen-Image-Edit (latest dated checkpoints)",
        _qwen_image_jobs,
    ),
    "qwen-image-2.1": ModelSpec(
        "qwen-image-2.1", "./models/qwen_image21",
        "Qwen-Image 2.1 (t2i + edit in one DiT)",
        _qwen_image_21_jobs,
    ),
    "mage-flow": ModelSpec(
        "mage-flow", "./models/mage_flow",
        "Microsoft Mage-Flow (t2i + edit; base / RL / turbo)",
        _mage_flow_jobs,
        variant_supported=True,
        variant_choices=("base", "rl", "turbo"),
        default_variant="turbo",
    ),
    "esrgan": ModelSpec(
        "esrgan", "./models/esrgan",
        "Real-ESRGAN x4 pixel upscaler (optional; used for upscaling)",
        _esrgan_jobs,
        has_int8=False,
    ),
}


def _epilog() -> str:
    lines = ["models:"]
    for key in sorted(MODELS):
        spec = MODELS[key]
        note = f" (default variant: {spec.default_variant})" if spec.default_variant else ""
        lines.append(f"  {key + ':':18s} {spec.help}{note}")
    lines += [
        "",
        "examples:",
        "  python scripts/download.py --model anima",
        "  python scripts/download.py --model krea2 --int8-convrot",
        "  python scripts/download.py --model zimage --variant base",
        "  python scripts/download.py --model klein --variant 9b",
        "  python scripts/download.py --model ming-image",
        "  python scripts/download.py --model ming-image --int8-convrot",
        "  python scripts/download.py --model qwen-image --edit-only",
        "  python scripts/download.py --model qwen-image-2.1 --int8-convrot",
        "  python scripts/download.py --model mage-flow",
        "  python scripts/download.py --model mage-flow --variant base",
        "  python scripts/download.py --model mage-flow --int8-convrot --edit-only",
        "  python scripts/download.py --model esrgan",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download model artifacts for TheNoise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog(),
    )
    ap.add_argument(
        "--model", required=True, choices=sorted(MODELS),
        help="model to download",
    )
    ap.add_argument(
        "--out", default=None,
        help="output directory (default: the model's standard dir, e.g. ./models/krea2)",
    )
    ap.add_argument(
        "--int8-convrot", action="store_true",
        help="download the int8-convrot DiT(s) instead of bf16 (where supported)",
    )
    ap.add_argument(
        "--variant", default=None,
        help=(
            "anima: DiT variant name (default: turbo-v1.0; others include base-v1.0, "
            "aesthetic-v1.1) | zimage: turbo | base (default: turbo) | "
            "klein: 4b | 4b-base | 9b | 9b-base (default: 4b) | "
            "mage-flow: turbo | rl | base (default: turbo)"
        ),
    )
    ap.add_argument(
        "--include-raw", action="store_true",
        help="krea2: also download the RAW (non-turbo) DiT",
    )
    ap.add_argument(
        "--image-only", action="store_true",
        help="qwen-image / mage-flow: only the t2i DiT (shared TE + VAE are still fetched)",
    )
    ap.add_argument(
        "--edit-only", action="store_true",
        help="qwen-image / mage-flow: only the edit DiT (shared TE + VAE are still fetched)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="print the download plan without downloading",
    )
    args = ap.parse_args()

    spec = MODELS[args.model]

    # ---- per-model validation ---------------------------------------------
    if args.variant is None and spec.default_variant is not None:
        args.variant = spec.default_variant
    if args.variant is not None:
        if not spec.variant_supported:
            ap.error(f"--variant is not supported by --model {args.model}")
        if spec.variant_choices is not None and args.variant not in spec.variant_choices:
            ap.error(
                f"--variant {args.variant!r} is not valid for --model {args.model} "
                f"(choices: {', '.join(spec.variant_choices)})"
            )
    if args.include_raw and args.model != "krea2":
        ap.error("--include-raw is only supported by --model krea2")
    if (args.image_only or args.edit_only) and args.model not in ("qwen-image", "mage-flow"):
        ap.error("--image-only / --edit-only are only supported by --model mage-flow, qwen-image")
    if args.image_only and args.edit_only:
        ap.error("--image-only and --edit-only are mutually exclusive")
    if args.int8_convrot and not spec.has_int8:
        ap.error(f"--int8-convrot is not supported by --model {args.model}")

    # ---- download -----------------------------------------------------------
    out_dir = args.out or spec.default_out
    jobs = spec.jobs(args)

    if args.dry_run:
        print(f"plan for --model {args.model} -> {out_dir}")
        for name, repo, path in jobs:
            print(f"  {name:14s} {repo} :: {path}")
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.model} artifacts -> {out}")
    for name, repo, path in jobs:
        dest = hf_hub_download(repo, path, local_dir=str(out))
        print(f"  {name:14s} -> {dest}")
    print(f"done: {args.model} ({len(jobs)} artifact(s))")


if __name__ == "__main__":
    main()
