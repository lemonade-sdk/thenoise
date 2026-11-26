from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar, List, Optional, Sequence, Tuple

import torch

from thenoise.models.base import DiffusionModel
from thenoise.models.config import ModelConfig
from thenoise.upscale import LatentUpscaler

if TYPE_CHECKING:  # pragma: no cover - only for annotations
    from PIL import Image


class PixelModel(DiffusionModel):
    """Base class for adapters whose output is an image, or frames of one."""

    DEFAULT_PREFS = {**DiffusionModel.DEFAULT_PREFS,
                     "width": 1024,
                     "height": 1024,
                     "ref_method": "index"}

    CAPABILITIES = {**DiffusionModel.CAPABILITIES,
                    "image": {"t2i": True, "i2i": False},
                    "upscale": True}

    UPSCALE_SCALE = 2
    REFINE_STEPS = 1
    REFINE_DENOISE = 0.25

    def __init__(self, *, config: ModelConfig):
        super().__init__(config=config)
        self._upscaler: Optional[LatentUpscaler] = None

    # ------------------------------------------------------------ size
    def resolve_size(self, width: int, height: int) -> tuple[int, int]:
        """Return the effective (width, height). Override to round/validate."""
        return width, height

    #: Area an edit is sampled at when the request names no size.
    EDIT_DEFAULT_AREA: ClassVar[int] = 1024 * 1024

    def default_edit_size(self, width: int, height: int) -> Tuple[int, int]:
        """The result size of an edit that names none, for a ``width`` x ``height`` reference.

        The reference's aspect ratio at :data:`EDIT_DEFAULT_AREA`; ``resolve_size`` still
        rounds the answer onto this model's latent cell. Override it when a model was
        trained at particular sizes rather than at any size.
        """
        scale = (self.EDIT_DEFAULT_AREA / (width * height)) ** 0.5
        return round(width * scale), round(height * scale)

    def size_from_reference(self, width: int | None, height: int | None,
                            reference) -> tuple[int, int]:
        """What a request named, or the reference's size when it named neither.

        `default_edit_size` is what turns the reference into this model's answer.
        ``reference`` is a decoded image, or the ``(width, height)`` a file's header
        stated -- which is all a run needs from a reference that only names a size.
        """
        if width is None and height is None:
            return self.default_edit_size(*(reference.size if hasattr(reference, "size")
                                            else reference))
        return width, height

    # ------------------------------------------------------------ editing
    def fit_references(self, images: List[Image.Image],
                       target: Tuple[int, int]) -> List[Image.Image]:
        """Every reference of one edit, fitted for :meth:`encode_reference`.

        ``target`` is the size of the result. The default hands each image to
        :meth:`prepare_reference` and ignores it, because a reference normally keeps the
        shape it arrived in; a model whose edit lines up with its lone reference pixel
        for pixel overrides this to put that reference on the canvas's own grid.
        """
        return [self.prepare_reference(img) for img in images]

    def fit_guides(self, images: List[Image.Image],
                   target: Tuple[int, int]) -> List[Image.Image]:
        """Reference conditions, fitted for :meth:`encode_references`.

        Separate from :meth:`fit_references` because a model can be strict about the
        frames of its own clip and generous about the ones it is only asked to resemble.
        """
        return self.fit_references(images, target)

    def encode_reference(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode input pixels (``[C,H,W]`` in [-1, 1]) into the canonical latent."""
        raise NotImplementedError(f"{self.name} does not support reference editing")

    def encode_references(self, pixels: list[torch.Tensor],
                          frames: int = 1) -> list[torch.Tensor]:
        """The latents a run conditions on, from fitted images: one per image, unless
        the model takes its reference some other way.

        `frames` is the temporal extent the run spans. A model whose reference is one
        latent for a whole clip has to override this rather than be handed per-frame
        latents, because the codec is causal: the later latent frames of a clip are not
        the first one repeated.
        """
        return [self.encode_reference(image) for image in pixels]

    def pack_reference_latent(
        self,
        latents: torch.Tensor,
        method: str = "index",
        ref_index: int = 1,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Canonical reference latent -> model-internal (tokens, ids).

        ``ref_index`` is the 1-based position among the reference images.
        """
        return None

    # ------------------------------------------------------- latent upscaler
    def get_upscaler(self) -> LatentUpscaler:
        """This model's latent upscaler, built once on first use (under the lock)."""
        if self._upscaler is None:
            self._upscaler = self._create_upscaler()
        return self._upscaler

    @abstractmethod
    def _create_upscaler(self) -> LatentUpscaler:
        """Build this model's latent upscaler."""
        ...

    # ------------------------------------------------------------ pixel format
    @property
    def pixel_channels(self) -> int:
        """Pixel channels this model's VAE consumes and emits (3 = RGB, 4 = RGBA)."""
        return getattr(getattr(self, "vae", None), "pixel_channels", 3)

    # ------------------------------------------------------------ decode
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Shared VAE decode — the final generation step.

        Canonical 4D latent -> pixels ``[C, H, W]`` in [-1, 1] as an fp32 GPU tensor.
        """
        dev = torch.device(self.device)
        pixels = self.vae.decode_to_pixels(latents.to(dev, dtype=self.vae.dtype))
        if pixels.ndim == 5:  # [B, C, 1, H, W] -> [B, C, H, W]
            pixels = pixels.squeeze(2)
        pixels = pixels.to(torch.float32)
        return pixels[0]  # [C, H, W] in [-1, 1]


def expect_latent_width(dit_channels: int, codec_channels: int,
                        *, also: Sequence[int] = ()) -> None:
    """Refuse a codec whose latent this DiT cannot read: they arrive as two files.

    ``also`` lists the widths that pair as well, for a model that widens its trunk on
    purpose — Wan 2.2's I2V takes a reference concatenated onto the noise, and a codec
    and a transformer from two different releases otherwise meet only in an error about
    tensor shapes, halfway into a run.
    """
    if dit_channels in (codec_channels, *also):
        return
    pairs = f"{codec_channels} channels"
    if also:
        pairs += (", or " + " or ".join(str(width) for width in also)
                  + " for a reference concatenated onto the noise")
    raise ValueError(
        f"the --vae codec writes a {codec_channels}-channel latent, which this DiT "
        f"cannot read: it denoises a {dit_channels}-channel one. The two pair at {pairs}.")


__all__ = ["PixelModel", "expect_latent_width"]
