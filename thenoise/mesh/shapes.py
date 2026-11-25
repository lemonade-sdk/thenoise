"""The latents behind a finished shape, kept so a cut can be re-made without sampling."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from thenoise.jobs import JOB_ID_PREFIX

logger = logging.getLogger(__name__)

SHAPES_LIMIT = 10


class ShapeStore:
    """``<ident>.pt`` in one directory: latents, and nothing else."""

    def __init__(self, root: Path, limit: int = SHAPES_LIMIT):
        self._root = Path(root)
        self._limit = limit

    @property
    def root(self) -> Path:
        """The directory the store owns, which is a queue's artifact directory."""
        return self._root

    def path(self, ident: str) -> Path:
        if not ident.startswith(JOB_ID_PREFIX) or not ident.isidentifier():
            raise ValueError(f"a shape is addressed by its job id, not {ident!r}")
        return self._root / f"{ident}.pt"

    def save(self, ident: str, latents: torch.Tensor) -> Path:
        target = self.path(ident)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"latents": latents.detach().to("cpu")}, target)
        logger.info("shape %s kept: %s of latents", ident, _bytes(target.stat().st_size))
        self.prune()
        return target

    def load(self, ident: str) -> torch.Tensor:
        try:
            record = torch.load(self.path(ident), map_location="cpu", weights_only=True)
        except FileNotFoundError:
            raise ValueError(
                f"shape {ident!r} has no saved latents: only the newest {self._limit} "
                f"are kept, and none of them survive a restart") from None
        latents = record["latents"]
        logger.info("shape %s reused: %d latent tokens, no sampling",
                    ident, latents.shape[-2])
        return latents

    def kept(self) -> list[str]:
        """The shapes a request can still name, newest first."""
        return [one for one, _ in self.shapes()]

    def shapes(self) -> list[tuple[str, float]]:
        """What is on disk, newest first, as ``(ident, mtime)``."""
        found = (self._root.glob(f"{JOB_ID_PREFIX}*.pt")
                 if self._root.is_dir() else ())
        return sorted(((one.stem, one.stat().st_mtime) for one in found),
                      key=lambda one: one[1], reverse=True)

    def prune(self) -> None:
        for stale, _ in self.shapes()[self._limit:]:
            self.path(stale).unlink(missing_ok=True)


def _bytes(nbytes: int) -> str:
    return f"{nbytes / (1 << 20):.1f} MiB" if nbytes >= 1 << 20 else f"{nbytes} B"


__all__ = ["SHAPES_LIMIT", "ShapeStore"]
