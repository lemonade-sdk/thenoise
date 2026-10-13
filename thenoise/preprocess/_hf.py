"""The only place ``transformers`` names and conventions are referenced.

The engine pins ``transformers>=4.57.6`` and runs on v5, where a few keyword names
changed (``dtype`` for ``torch_dtype``, ``backend`` for ``use_fast``). The helpers
here pick the right one once, so the preprocessors never touch the version drift.
"""
from __future__ import annotations

import os

import transformers

#: v5 renamed a few ``from_pretrained`` keywords; the helpers below pick the right one
#: once, so the preprocessors never touch the version drift.
_IS_V5 = transformers.__version__.startswith("5")


def _dtype_kwargs(dtype) -> dict:
    """The ``from_pretrained`` keyword for a torch dtype: ``dtype`` on v5."""
    return {"dtype": dtype} if _IS_V5 else {"torch_dtype": dtype}


def _backend_kwargs() -> dict:
    """The image-processor backend keyword: ``backend`` on v5, ``use_fast`` on v4."""
    return {"backend": "pil"} if _IS_V5 else {"use_fast": False}


def require_depth_files(path: str) -> None:
    """Raise with the remedy when a depth checkpoint directory is incomplete."""
    needed = ("config.json", "model.safetensors", "preprocessor_config.json")
    missing = [f for f in needed if not os.path.isfile(os.path.join(path, f))]
    if missing:
        raise ValueError(
            f"depth preprocessor at {path} is missing {', '.join(missing)}; "
            "run 'scripts/download.py --model preprocess' to fetch them"
        )


def load_depth_model(path: str, dtype):
    """A Depth-Anything model from a checkpoint directory."""
    from transformers import AutoModelForDepthEstimation

    return AutoModelForDepthEstimation.from_pretrained(path, **_dtype_kwargs(dtype))


def load_depth_processor(path: str):
    """The Depth-Anything image processor, PIL-backed (no torchvision dependency)."""
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(path, **_backend_kwargs())


__all__ = ["load_depth_model", "load_depth_processor", "require_depth_files"]
