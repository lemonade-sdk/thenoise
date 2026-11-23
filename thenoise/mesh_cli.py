"""CLI mesh: one model, one shape."""
from __future__ import annotations

import logging

from .one_shot import check_solver, container_of, render_one
from .utils.media import read_media

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _summary(resolved) -> str:
    return (f"{resolved.latents} tokens, {resolved.steps} {resolved.sampler} steps, "
            f"volume {resolved.octree_resolution}, seed {resolved.seed}")


def run_mesh(args) -> None:
    """Load one model, resolve one request against it, render one shape."""
    from pydantic import ValidationError

    from .mesh.spec import MeshSpec, OUTPUT_FORMATS

    check_solver(args.sampler, "mesh")
    try:
        spec = MeshSpec(
            prompt=args.prompt, negative_prompt=args.negative_prompt,
            image=[read_media(path, "mesh") for path in args.image], latents=args.latents,
            steps=args.steps, guidance_scale=args.guidance_scale, sampler=args.sampler,
            seed=args.seed,
            output_format=container_of(args.out, args.output_format, OUTPUT_FORMATS),
            octree_resolution=args.octree_resolution, bounds=args.bounds,
            mc_level=args.mc_level, scale=args.scale, lora_specs=args.lora or None,
        )
    except ValidationError as e:
        raise SystemExit("mesh: " + "; ".join(err["msg"] for err in e.errors())) from e

    render_one(spec, args, medium="mesh", command="mesh", noun="shape",
               summary=_summary)
