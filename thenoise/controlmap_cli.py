"""CLI control-map extraction: no diffusion model, one input image, one PNG per type."""
from __future__ import annotations

import logging
import os

from .utils.paths import ensure_png_extension
from .utils.png import save_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_preprocess(args) -> None:
    from .runtime import Settings, Runtime
    from .utils.image_tensor import load_image

    runtime = Runtime(Settings(device=args.device, preprocessor_dir=args.preprocessor_dir))  # no load(): extraction is model-free

    if args.list:
        for kind, names in runtime.preprocessors.available().items():
            logger.info("%s: %s", kind, ", ".join(names))
        # A kind whose weights are only half there is worth naming: it is on disk,
        # just not runnable, and saying so beats an empty list with no reason.
        for kind, reason in runtime.preprocessors.missing().items():
            logger.info("%s: unavailable, %s", kind, reason)
        return
    if not args.input or not args.types:
        raise SystemExit("preprocess: --input and --type are required (or --list)")

    image = load_image(args.input)
    stem, extension = os.path.splitext(ensure_png_extension(args.out))
    for kind in args.types:
        try:
            out = runtime.preprocess.run(image, kind, _options(runtime, kind, args), args.model)
        except ValueError as e:
            raise SystemExit(f"preprocess: {e}")
        # More than one map requested: each gets a file named after its type.
        suffix = "" if len(args.types) == 1 else f"_{kind}"
        path = save_output(out, stem + suffix + extension)
        logger.info("saved %s (type=%s, model=%s)", path, kind, args.model or kind)


def _options(runtime, kind: str, args):
    """The kind's own ``--<kind>-<option>`` flags, un-namespaced: which flags a kind
    accepts is argparse's business, so this stays generic as the pool grows."""
    prefix = f"{kind}_"
    return runtime.preprocessors.options_for(
        kind,
        **{
            name[len(prefix):]: getattr(args, name)
            for name in vars(args)
            if name.startswith(prefix)
        },
    )


__all__ = ["run_preprocess"]
