"""Reusable quantized linear projection.

``QuantizedLinear`` is a drop-in replacement for ``nn.Linear``. A layer runs
either BF16 (default; ``weight`` is an ``nn.Parameter``) or a quantized scheme,
in which case ``weight`` becomes a ``comfy_kitchen.tensor.QuantizedTensor``
buffer. The module is fully layout-agnostic: ``forward`` just calls ``F.linear``,
and the ``QuantizedTensor``'s ``__torch_dispatch__`` routes the GEMM to the right
kernel for its layout (INT8+ConvRot, FP8, NVFP4, MXFP8, ...). Quantized layers
emit the activation dtype, so everything downstream is dtype-agnostic and the
rest of the model needs no changes.

Quantized weights live inside a ``QuantizedTensor`` buffer (not an
``nn.Parameter``) because PyTorch forbids gradients on integer tensors — and a
``QuantizedTensor`` subclass cannot be a Parameter at all. Loaders should use
``thenoise.utils.loader.load_quantized_state_dict`` to populate the model from a
ComfyUI-style checkpoint.

LoRAs on quantized layers are *baked in* at switch time (``bake_lora``): the
weight is dequantized to BF16, the LoRA delta is added, and the result is
requantized with the layer's preserved layout profile (``requantize_from_float``
carries over the ConvRot flag, group size, and scale granularity). The runtime
forward is therefore a single quantized GEMM with zero per-step LoRA cost. Undo
reloads the original weights from the checkpoint file (see
``thenoise.utils.loader.build_quantized_restore_map`` / ``restore_quantized_layer``).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy_kitchen.tensor import QuantizedTensor

from thenoise.utils.loader import restore_quantized_layer


class QuantizedLinear(nn.Module):
    """Linear projection that runs BF16 (default) or a quantized scheme."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # float32 default (matching ``nn.Linear``); the model is cast to the
        # compute dtype (bf16) by the adapter, or the weight is replaced at load.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()
        self._quantized = False

    def _reset_parameters(self) -> None:
        """Initialize like ``nn.Linear`` (kaiming on weight, uniform on bias)."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def load_quantized(self, qt: QuantizedTensor) -> None:
        """Switch this layer to a pre-quantized weight of any layout.

        Frees the BF16 ``weight`` parameter and registers ``qt`` as the
        ``weight`` buffer (the ``QuantizedTensor`` carries the layout profile —
        scale, ConvRot flag/group size, original dtype/shape).
        """
        del self.weight  # free the BF16 weights
        self.register_buffer("weight", qt)
        self._quantized = True

    def bake_lora(self, delta: torch.Tensor) -> None:
        """Bake a BF16 LoRA delta into the quantized weights (any layout).

        Args:
            delta: the LoRA delta ``[out, in]`` in BF16 (``multiplier * (up @
                down) * (alpha/r)``). Multiple LoRAs should be summed into one
                delta before calling, so the layer is dequantized/requantized
                only once.

        Dequantizes ``weight`` to BF16 (un-rotating ConvRot if active), adds the
        delta, and requantizes back with this layer's preserved layout profile
        (``requantize_from_float`` keeps ConvRot flag, group size, and scale
        granularity). The runtime forward stays a single quantized GEMM (zero
        per-step LoRA cost). The original weights are restored on undo by
        reloading from disk.
        """
        qt = self.weight
        weight = qt.dequantize()
        self._set_quantized(qt.requantize_from_float(weight + delta.to(weight.dtype)))

    def _set_quantized(self, qt: QuantizedTensor) -> None:
        """Overwrite the quantized ``weight`` buffer in place (preserving identity)."""
        self.weight.copy_(qt)

    def apply_lora(self, delta: torch.Tensor) -> None:
        """Apply a LoRA delta ``[out, in]`` in place.

        BF16 layers add ``delta`` to the weight parameter directly. Quantized
        layers bake it in (``bake_lora``), so the runtime forward stays a single
        quantized GEMM with zero per-step LoRA cost.
        """
        if self._quantized:
            self.bake_lora(delta)
        else:
            self.weight.data.add_(delta.to(self.weight.dtype))

    def undo_lora(
        self,
        delta: Optional[torch.Tensor],
        *,
        raw_key: Optional[str] = None,
        dit_path: Optional[str] = None,
    ) -> None:
        """Undo a previously applied LoRA delta.

        BF16 layers subtract ``delta`` from the weight parameter (exact, no
        compounding). Quantized layers reload the original quantized weights
        from the checkpoint file by ``raw_key`` (avoids compounding
        quantization errors from repeated dequantize/requantize). ``delta`` is
        unused for quantized layers.
        """
        if self._quantized:
            if raw_key is None:
                raise RuntimeError(
                    "cannot undo quantized LoRA: no raw checkpoint key was "
                    "recorded at load time"
                )
            if dit_path is None:
                raise RuntimeError(
                    "cannot undo quantized LoRA: no dit_path was recorded at "
                    "apply time"
                )
            restore_quantized_layer(self, dit_path, raw_key)
        else:
            self.weight.data.sub_(delta.to(self.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def replace_linears(model: nn.Module) -> None:
    """Replace every ``nn.Linear`` in ``model`` with a drop-in ``QuantizedLinear``.

    Do this while the model is still on meta (inside ``init_empty_weights``) so the
    fresh parameters stay meta.
    """
    for name, module in list(model.named_modules()):
        if module.__class__ is nn.Linear:
            parent_path, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(
                parent,
                attr,
                QuantizedLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                ),
            )


__all__ = ["QuantizedLinear", "replace_linears"]
