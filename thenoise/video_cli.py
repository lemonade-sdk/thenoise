"""CLI video: one model, one clip — what ``generate`` is to a still image."""
from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

from .utils.lora_roles import DEFAULT_LORA_ROLE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _frame(flag: str, path: str) -> bytes:
    """One reference image's bytes; the spec takes bytes, not a path."""
    try:
        return Path(path).read_bytes()
    except OSError as e:
        raise SystemExit(f"video: {flag}: {e}") from e


def _lora_specs(args) -> dict[str, list[str]] | None:
    """LoRAs by role; a role the model has no component for is refused by the request
    check."""
    specs = {DEFAULT_LORA_ROLE: args.lora, "dit_low": args.lora_low}
    return {role: list(values) for role, values in specs.items() if values} or None


def _container(name: str, requested: str | None) -> str | None:
    """The container asked for, by flag or by ``--out``'s extension; None is the model's
    own default."""
    from .video.spec import OUTPUT_FORMATS

    if requested:
        return requested
    suffix = Path(name).suffix[1:]
    return suffix if suffix in OUTPUT_FORMATS else None


def _out_path(name: str, output_format: str) -> Path:
    """``--out`` extended with the container written; a name promising another is
    refused rather than renamed."""
    suffix = Path(name).suffix
    if not suffix:
        return Path(f"{name}.{output_format}")
    if suffix != f".{output_format}":
        raise SystemExit(
            f"video: --out {name} ends in {suffix} but this clip is {output_format}")
    return Path(name)


class _Progress:
    """One bar per stage; ``decoding`` is the controller's own to bar."""

    def __init__(self) -> None:
        self._bars: dict[str, tqdm] = {}

    def __call__(self, stage: str, step: int, of: int) -> None:
        if stage == "decoding":
            return
        bar = self._bars.get(stage)
        if bar is None:
            bar = self._bars[stage] = tqdm(total=of, desc=stage, leave=False)
        bar.update(step - bar.n)
        if of and step >= of:
            bar.close()

    def close(self) -> None:
        for bar in self._bars.values():
            bar.close()


def run_video(args) -> None:
    """Load one model, resolve one request against it, render one clip."""
    from pydantic import ValidationError

    from .jobs import Uncancellable
    from .runtime import ModelPaths, Runtime, Settings
    from .video.spec import VideoSpec

    # The spec refuses this pair too, but only on its way to the model's defaults.
    if args.end_image and not args.image:
        raise SystemExit("video: --end-image needs --image"
                         ": a clip cannot end before it starts")

    try:
        spec = VideoSpec(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width, height=args.height,
            frames=args.frames, duration=args.duration, fps=args.fps,
            steps=args.steps, guidance_scale=args.guidance_scale,
            flow_shift=args.flow_shift, sampler=args.sampler, seed=args.seed,
            output_format=_container(args.out, args.output_format),
            image=_frame("--image", args.image) if args.image else None,
            end_image=_frame("--end-image", args.end_image) if args.end_image else None,
            lora_specs=_lora_specs(args),
        )
    except ValidationError as e:
        raise SystemExit("video: " + "; ".join(err["msg"] for err in e.errors())) from e

    runtime = Runtime(Settings(device=args.device))
    runtime.load(ModelPaths(
        dit_path=args.dit, vae_path=args.vae, text_encoder_path=args.text_encoder,
        lora_dir=args.lora_dir, dit_low_path=args.dit_low,
    ))
    if not runtime.video:
        raise SystemExit(
            f"video: {runtime.model_name} renders stills; use `generate`")

    # Resolved first: the numbers land on the log before the minutes they take, and a
    # request this variant cannot serve fails here rather than mid-run.
    try:
        resolved = runtime.run(lambda controller: controller.resolve(spec))
    except ValueError as e:
        raise SystemExit(f"video: {e}") from e
    out = _out_path(args.out, resolved.output_format)
    logger.info("%s %s: %dx%d, %d frames at %d FPS, %d %s steps, seed %s",
                runtime.model_name, resolved.mode, resolved.width, resolved.height,
                resolved.frame_count, resolved.fps, resolved.steps, resolved.sampler,
                resolved.seed)

    progress = _Progress()
    try:
        runtime.run(lambda controller: controller.render(
            resolved, Uncancellable(), progress, out))
    finally:
        progress.close()
    logger.info("saved %s", out)
