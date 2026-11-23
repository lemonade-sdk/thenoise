"""The shared body of the one-shot media commands."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def check_solver(sampler: str | None, command: str) -> None:
    """Refuse a bad solver name before a model costs minutes to load."""
    if sampler is None:
        return
    from .samplers import check_sampler

    try:
        check_sampler(sampler)
    except ValueError as e:
        raise SystemExit(f"{command}: {e}") from e


def container_of(name: str, requested: str | None, formats: Sequence[str]) -> str | None:
    """The container asked for, by flag or by ``--out``'s extension; None is the default."""
    if requested:
        return requested
    suffix = Path(name).suffix[1:]
    return suffix if suffix in formats else None


def load(args, medium: str, command: str):
    """The runtime for a one-shot command's model paths, refusing another medium's."""
    from .cli import component_paths
    from .runtime import ModelPaths, Runtime, Settings

    runtime = Runtime(Settings(device=args.device))
    # A command declares only the checkpoint flags its medium can read.
    runtime.load(ModelPaths(
        dit_path=args.dit, vae_path=getattr(args, "vae", ""),
        text_encoder_path=getattr(args, "text_encoder", ""),
        lora_dir=args.lora_dir, dit_low_path=getattr(args, "dit_low", ""),
        extra_paths=component_paths(args),
    ))
    if runtime.medium != medium:
        other = "generate" if runtime.medium in ("image", None) else runtime.medium
        raise SystemExit(f"{command}: {runtime.model_name} does not serve {medium}; "
                         f"use `{other}`")
    return runtime


def render_one(spec, args, *, medium: str, command: str, noun: str,
               summary: Callable[[Any], str]) -> None:
    """Resolve one request before any weight moves, then render it to ``--out``."""
    from .jobs import idle, Uncancellable
    from .utils.paths import named_output

    runtime = load(args, medium, command)
    try:
        resolved = runtime.run(lambda controller: controller.resolve(spec))
    except ValueError as e:
        raise SystemExit(f"{command}: {e}") from e
    out = named_output(args.out, resolved.output_format, noun, command)
    logger.info("%s %s: %s", runtime.model_name, resolved.mode, summary(resolved))

   runtime.run(lambda controller: controller.render(resolved, Uncancellable(), idle, out))
    logger.info("saved %s", out)


__all__ = ["check_solver", "container_of", "load", "render_one"]
