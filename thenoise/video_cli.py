"""CLI video: one model, one clip — what ``generate`` is to a still image."""
from __future__ import annotations

import logging

from .one_shot import check_solver, container_of, render_one
from .utils.lora_roles import DEFAULT_LORA_ROLE
from .utils.media import read_media

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _lora_specs(args) -> dict[str, list[str]] | None:
    """LoRAs by role; a role the model has no component for is refused by the request
    check."""
    specs = {DEFAULT_LORA_ROLE: args.lora, "dit_low": args.lora_low}
    return {role: list(values) for role, values in specs.items() if values} or None


def _summary(resolved) -> str:
    return (f"{resolved.width}x{resolved.height}, {resolved.frame_count} frames at "
            f"{resolved.fps} FPS, {resolved.steps} {resolved.sampler} steps, "
            f"seed {resolved.seed}")


def one_reference(value: str) -> tuple[str, int]:
    """``PATH`` or ``PATH@FRAME``, the frame being the one the reference belongs to.

    The frame travels with the path because a reference is meaningless without it: the
    packing pins it to a frame of the clip. A trailing ``@`` that is not a number stays
    part of the path, since paths are allowed to contain one.
    """
    path, sep, tail = value.rpartition("@")
    if sep and (tail.lstrip("-").isdigit() and tail != "-"):
        return path, int(tail)
    return value, 0


def run_video(args) -> None:
    """Load one model, resolve one request against it, render one clip."""
    from pydantic import ValidationError

    from .video.spec import OUTPUT_FORMATS, Guide, VideoSpec

    # The spec refuses this pair too, but only on its way to the model's defaults.
    if args.end_image and not args.image:
        raise SystemExit("video: --end-image needs --image"
                         ": a clip cannot end before it starts")

    check_solver(args.sampler, "video")
    try:
        spec = VideoSpec(
            prompt=args.prompt, negative_prompt=args.negative_prompt,
            width=args.width, height=args.height,
            frames=args.frames, duration=args.duration, fps=args.fps,
            steps=args.steps, guidance_scale=args.guidance_scale,
            flow_shift=args.flow_shift, sampler=args.sampler, seed=args.seed,
            output_format=container_of(args.out, args.output_format, OUTPUT_FORMATS),
            image=read_media(args.image, "video") if args.image else None,
            end_image=(read_media(args.end_image, "video", "--end-image")
                       if args.end_image else None),
            references=[Guide(pixels=read_media(path, "video", "--reference"),
                              frame=frame)
                        for path, frame in map(one_reference, args.reference)],
            lora_specs=_lora_specs(args),
        )
    except ValidationError as e:
        raise SystemExit("video: " + "; ".join(err["msg"] for err in e.errors())) from e

    render_one(spec, args, medium="video", command="video", noun="clip",
               summary=_summary)
