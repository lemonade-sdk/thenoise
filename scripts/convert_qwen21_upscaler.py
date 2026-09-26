"""Convert the upstream Qwen-Image 2.1 latent upscaler to thenoise's Sesqui layout.

Source: https://github.com/kinfolk0117/ComfyUI-QwenImage21-LatentUpscale — MIT,
``(c) SesquiLSR Contributors`` + ``(c) kinfolk0117``. The file to convert is its
``models/qwen21-latent-2x-v1.safetensors``: a SesquiLSR trained for the
Qwen-Image 2.1 VAE, exported in two ways that are specific to ComfyUI and to a
fixed 2x factor. Rewriting it into the layout
``thenoise.upscale.sesqui_net.SesquiLSRNet`` already loads means no second network
class, no second strategy and no second latent format beyond a registry entry.

1. It wraps the network in a ``Qwen21Upscaler`` module — keys prefixed ``net.`` —
   which also carries ``mean``/``std`` buffers and computes
   ``out = std * net((x - mean) / std) + mean``. That normalization is a shim for
   ComfyUI, whose latent is the *raw* VAE latent. thenoise's canonical latent is
   already ``(raw - mean) / std`` with these very statistics — checked against
   ``thenoise.vae.wan22.LATENT_STATS[64]`` below, and asserted, because dropping
   the shim is only lossless if they agree — so handing the trunk the canonical
   latent is exactly equivalent. The buffers are therefore dropped, not
   reimplemented, and the format's adaptor is the identity.

2. Its coordinate-conditioned reassembly head is folded into constant taps
   (``upsample.filters`` / ``upsample.gains``). That fold is exact, not an
   approximation: Sesqui applies the reassembly filter *after* the 2x pixel
   shuffle, so at 2x every output coordinate has ``frac == 0`` and ``log_foot == 0``
   and the predicted filter cannot vary by position. The fold is undone key-wise —
   ``filters`` -> ``base_taps``, ``gains`` -> ``rank_gain`` (our head applies
   ``sigmoid``, so it is stored as the logit) — and the predictor tensors the fold
   deleted (``filter_net.*``, ``pe.freqs``, ``axis_emb``, ``rank_emb``) are written
   zeroed, which makes our head reduce to exactly those fixed taps.

Usage:
    .venv/bin/python scripts/convert_qwen21_upscaler.py \
        models/qwen21/qwen21-latent-2x-v1.safetensors
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from thenoise.upscale.sesqui_net import SesquiLSRNet
from thenoise.vae.wan22 import QWEN_IMAGE21_LATENTS_MEAN, QWEN_IMAGE21_LATENTS_STD

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The raw-VAE latent width of the Qwen-Image 2.1 VAE (= its ``z_dim``).
CHANNELS = 64

DEFAULT_BF16_OUT = REPO_ROOT / "thenoise" / "upscale" / "weights" / "upscaler_qwen21.safetensors"
DEFAULT_FP32_OUT = REPO_ROOT / "models" / "qwen21-latent-2x-v1-fp32.safetensors"

SOURCE = "https://github.com/kinfolk0117/ComfyUI-QwenImage21-LatentUpscale"


def check_normalization(sd: dict) -> None:
    """Fail unless the wrapper's ``mean``/``std`` ARE the VAE's latent statistics.

    The upstream module normalizes with these buffers and un-normalizes on the way
    out, so handing it the already-normalized canonical latent is only equivalent
    to handing the trunk the canonical latent if the two statistics are the same
    numbers. In the released file they agree to fp32 precision (the buffers are
    fp32, the VAE's constants are float64); if a future export ever differs, the
    buffers cannot be dropped, so this is a hard check rather than a note.
    """
    want_mean = torch.tensor(QWEN_IMAGE21_LATENTS_MEAN, dtype=torch.float64).view(1, -1, 1, 1)
    want_std = torch.tensor(QWEN_IMAGE21_LATENTS_STD, dtype=torch.float64).view(1, -1, 1, 1)
    mean, std = sd["mean"].double(), sd["std"].double()
    if mean.shape != want_mean.shape or std.shape != want_std.shape:
        raise ValueError(
            f"expected (1, {CHANNELS}, 1, 1) mean/std buffers, got "
            f"{tuple(mean.shape)}/{tuple(std.shape)}"
        )
    d_mean = (mean - want_mean).abs().max().item()
    d_std = (std - want_std).abs().max().item()
    print(f"  normalization vs LATENT_STATS[64]: max diff {d_mean:.3e} / {d_std:.3e}")
    if d_mean > 1e-5 or d_std > 1e-5:
        raise ValueError(
            "the file's mean/std differ from the VAE's latent statistics; the "
            "buffers cannot be dropped, keep the normalization (in fp32) instead"
        )


def fold(sd: dict) -> dict:
    """Rewrite the upstream state dict into the ``SesquiLSRNet`` key layout."""
    # Shapes are read off the target module, so this stays honest if it is retuned.
    probe = SesquiLSRNet(in_channels=CHANNELS).state_dict()

    out = {}
    for key, value in sd.items():
        if key in ("mean", "std"):
            continue  # raw-latent shim, see check_normalization
        if not key.startswith("net."):
            raise ValueError(f"unexpected key {key!r}: not wrapped in the net module")
        key = key[len("net."):]
        if key == "upsample.filters":
            # [rank, y|x, trunk width, taps] — the same tensor our head keeps as
            # the fixed part of its predicted filter.
            want = tuple(probe["upsample.base_taps"].shape)
            if tuple(value.shape) != want:
                raise ValueError(f"upsample.filters has shape {tuple(value.shape)}, want {want}")
            out["upsample.base_taps"] = value
        elif key == "upsample.gains":
            want = tuple(probe["upsample.rank_gain"].shape)
            if tuple(value.shape) != want:
                raise ValueError(f"upsample.gains has shape {tuple(value.shape)}, want {want}")
            if not bool(((value > 0) & (value < 1)).all()):
                raise ValueError("upsample.gains must be the already-sigmoid'd rank gains")
            # Our head applies sigmoid(), so store the logit.
            out["upsample.rank_gain"] = torch.log(value / (1.0 - value))
        else:
            out[key] = value

    # The folded head has no coordinate predictor; ours does, and with its output
    # layer zeroed the predicted residual is identically zero, leaving exactly the
    # constant taps above. Every other missing key would be an architecture change.
    missing = sorted(set(probe) - set(out))
    if any(not k.startswith("upsample.") for k in missing):
        raise ValueError(f"keys this converter cannot synthesize: {missing}")
    extra = sorted(set(out) - set(probe))
    if extra:
        raise ValueError(f"keys SesquiLSRNet does not know: {extra}")
    for key in missing:
        out[key] = torch.zeros_like(probe[key])
    print(f"  zeroed {len(missing)} folded-away predictor tensors: {', '.join(missing)}")
    return out


def metadata_for(upstream: dict) -> dict:
    """Upstream metadata (all of it is strings) plus this conversion's provenance."""
    meta = {k: str(v) for k, v in (upstream or {}).items()}
    meta.update(
        {
            "source": SOURCE,
            "license": "MIT (SesquiLSR Contributors, kinfolk0117)",
            "converted_by": "scripts/convert_qwen21_upscaler.py",
            "note": (
                "coordinate-conditioned reassembly filters folded to constant taps "
                "(exact at 2x), filter_net zeroed; the raw-latent mean/std "
                "normalization was dropped because the canonical latent is already "
                "normalized with these statistics"
            ),
        }
    )
    return meta


def save(sd: dict, path: Path, dtype: torch.dtype, metadata: dict) -> Path:
    tensors = {k: v.to(dtype).contiguous().clone() for k, v in sd.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)
    print(f"  wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {dtype})")
    return path


def verify(path: Path) -> None:
    """Re-open the written file the way the engine will: strict load, one forward."""
    for dtype in (torch.float32, torch.bfloat16):
        net = SesquiLSRNet(in_channels=CHANNELS)
        net.load_state_dict(load_file(str(path)), strict=True)
        net.to(dtype).eval().requires_grad_(False)
        z = torch.randn(1, CHANNELS, 8, 8, dtype=dtype)
        with torch.no_grad():
            up = net(z, (16, 16))
        print(f"  verify {dtype}: 8x8 -> {tuple(up.shape)[-2:]}, finite={bool(torch.isfinite(up).all())}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert the upstream Qwen-Image 2.1 latent upscaler to "
        "thenoise's SesquiLSRNet layout (bf16 for the repo, fp32 for interchange)"
    )
    ap.add_argument("checkpoint", type=Path, help="upstream qwen21-latent-2x-v1.safetensors")
    ap.add_argument("--bf16-out", type=Path, default=DEFAULT_BF16_OUT,
                    help=f"committed engine weight (default: {DEFAULT_BF16_OUT})")
    ap.add_argument("--fp32-out", type=Path, default=DEFAULT_FP32_OUT,
                    help=f"fp32 copy, not committed (default: {DEFAULT_FP32_OUT}); "
                         "pass --no-fp32 to skip")
    ap.add_argument("--no-fp32", action="store_true", help="skip the fp32 copy")
    args = ap.parse_args()

    sd = load_file(str(args.checkpoint))
    with safe_open(str(args.checkpoint), framework="pt") as f:
        upstream_meta = dict(f.metadata() or {})
    print(f"loaded {args.checkpoint} ({len(sd)} tensors)")
    check_normalization(sd)
    folded = fold(sd)
    meta = metadata_for(upstream_meta)

    print("converting:")
    save(folded, args.bf16_out, torch.bfloat16, meta)
    if not args.no_fp32:
        save(folded, args.fp32_out, torch.float32, meta)

    print("checking:")
    verify(args.bf16_out)


if __name__ == "__main__":
    main()
