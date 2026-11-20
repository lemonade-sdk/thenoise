from __future__ import annotations

from typing import Sequence

import torch
from accelerate import init_empty_weights
from PIL import Image
from transformers import Dinov2Config, Dinov2Model

from thenoise.utils.image_tensor import pil_to_pixels
from thenoise.utils.loader import load_dit

from .keys import CONDITIONER_PREFIX, Hunyuan3DConditionerParams, conditioner_key_map, take_prefix

#: DINOv2's pretraining statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def dino_normalize(pixels: torch.Tensor) -> torch.Tensor:
    """``[C,H,W]`` or ``[B,C,H,W]`` in [-1,1] -> DINOv2's own input range."""
    channels = pixels.shape[-3]
    mean = torch.tensor(IMAGENET_MEAN[:channels], device=pixels.device,
                        dtype=pixels.dtype).view(*(1,) * (pixels.dim() - 3), channels, 1, 1)
    std = torch.tensor(IMAGENET_STD[:channels], device=pixels.device,
                       dtype=pixels.dtype).view(*(1,) * (pixels.dim() - 3), channels, 1, 1)
    return ((pixels + 1.0) / 2.0 - mean) / std


class Hunyuan3DImageEncoder(Dinov2Model):
    """DINOv2 over a reference picture, using the ``transformers`` tower so the saved names load verbatim."""

    def __init__(self, params: Hunyuan3DConditionerParams):
        super().__init__(Dinov2Config(
            hidden_size=params.hidden_size,
            num_hidden_layers=params.layers,
            num_attention_heads=params.num_heads,
            intermediate_size=params.mlp_dim,
            patch_size=params.patch_size,
            image_size=params.image_size,
            layer_norm_eps=params.norm_eps,
            qkv_bias=params.qkv_bias,
            drop_path_rate=0.0,
            use_mask_token=True,
        ))
        self.params = params

    def pixels(self, images: Sequence[Image.Image], dtype: torch.dtype) -> torch.Tensor:
        """Fitted PIL pictures -> the normalized batch the tower takes, one size at a time."""
        sizes = {image.size for image in images}
        if len(sizes) > 1:
            raise ValueError(f"the tower batches one picture size at a time, got {sorted(sizes)}")
        batch = torch.stack([pil_to_pixels(image) for image in images])
        return dino_normalize(batch.to(dtype))

    def forward(self, images: Sequence[Image.Image], dtype: torch.dtype) -> torch.Tensor:
        """``[B, tokens, hidden]``: the reference as cross-attention context."""
        state = super().forward(self.pixels(images, dtype)).last_hidden_state
        return state if self.params.use_cls_token else state[:, 1:]

    def unconditional(self, like: torch.Tensor) -> torch.Tensor:
        """The null context: zeros shaped like the conditioned one, since the picture decides its width."""
        return torch.zeros_like(like)


def load_hunyuan3d_image_encoder(conditioner_path: str,
                                 params: Hunyuan3DConditionerParams, device,
                                 dtype: torch.dtype) -> Hunyuan3DImageEncoder:
    """Build on meta from the header's geometry, then load the conditioner."""
    with init_empty_weights():
        model = Hunyuan3DImageEncoder(params)
    load_dit(model, conditioner_path, device=device, dtype=None,
             state_map=take_prefix(CONDITIONER_PREFIX), key_map=conditioner_key_map)
    model.to(dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "Hunyuan3DImageEncoder", "dino_normalize",
           "load_hunyuan3d_image_encoder"]
