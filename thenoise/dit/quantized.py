"""Reusable INT8 linear projection.

``QuantizedLinear`` is a drop-in replacement for ``nn.Linear`` on layers a model
wants to run as symmetric INT8 with online ConvRot (Hadamard) activation rotation
via the ``comfy_kitchen`` INT8 GEMM (``torch.ops.comfy_kitchen.int8_linear``). The
compute path is chosen at load time: a layer that keeps its BF16 ``weight`` runs
``F.linear``; a layer switched with ``load_int8`` runs the INT8 kernel and frees
the BF16 weight. Quantized layers emit BF16 output, so everything downstream is
dtype-agnostic and the rest of the model needs no changes.

INT8 weights are stored in buffers (not parameters) because PyTorch forbids
gradients on integer tensors — ``load_state_dict(assign=True)`` cannot assign an
int8 tensor into an ``nn.Parameter``. Loaders should use
``thenoise.utils.int8.load_int8_state_dict`` to populate the model from a
ComfyUI-style INT8 checkpoint.

LoRAs on INT8 layers are *baked in* at switch time (``bake_lora``): the INT8
weight is dequantized to BF16, the LoRA delta is added, and the result is
requantized back to INT8 with the layer's ConvRot profile. The runtime forward
is therefore always a single INT8 GEMM with zero per-step LoRA cost. Undo
reloads the original INT8 weights from the checkpoint file (see
``thenoise.utils.int8.build_int8_restore_map``/``restore_int8_layer``).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy_kitchen


class QuantizedLinear(nn.Module):
    """Linear projection that runs BF16 (default) or INT8."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        convrot_groupsize: int = 256,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.convrot_groupsize = convrot_groupsize
        self.convrot = True  # set from the checkpoint's comfy_quant marker at load
        # float32 default (matching ``nn.Linear``); the model is cast to the compute
        # dtype (bf16) by the adapter, or the weight is replaced at load time.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()
        # INT8 output is always bf16; the raw op takes this as an int code (not a
        # torch.dtype), resolved once here so torch.compile never traces the dict
        # lookup. Using torch.ops.comfy_kitchen.int8_linear (a torch.library
        # custom op with a fake schema) instead of the comfy_kitchen.int8_linear
        # Python wrapper keeps the call traceable inside ``torch.compile``.
        self._out_dtype_code = comfy_kitchen.DTYPE_TO_CODE[torch.bfloat16]
        self._int8 = False

    def _reset_parameters(self) -> None:
        """Initialize like ``nn.Linear`` (kaiming on weight, uniform on bias)."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def load_int8(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        convrot: bool = True,
        convrot_groupsize: Optional[int] = None,
    ) -> None:
        """Switch this layer to INT8 using pre-quantized weights.

        Args:
            qweight: int8 weight tensor ``[out, in]``.
            scale: per-row F32 scale ``[out]`` (or ``[out, 1]``).
            convrot: whether the weights were ConvRot-rotated at export and so
                activations must be rotated at inference (from the checkpoint's
                ``comfy_quant`` marker).
            convrot_groupsize: Hadamard group size (defaults to 256).
        """
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        self.convrot = convrot
        if convrot_groupsize is not None:
            self.convrot_groupsize = convrot_groupsize
        self.register_parameter("weight", None)  # free the BF16 weights
        self._int8 = True

    def bake_lora(self, delta: torch.Tensor) -> None:
        """Bake a BF16 LoRA delta into the INT8 weights.

        Args:
            delta: the LoRA delta ``[out, in]`` in BF16 (``multiplier * (up @
                down) * (alpha/r)``). Multiple LoRAs should be summed into one
                delta before calling, so the layer is dequantized/requantized
                only once.

        Dequantizes ``qweight`` to BF16 (un-rotating ConvRot), adds the delta,
        and requantizes back to INT8 with this layer's ConvRot profile. The
        runtime forward stays a single INT8 GEMM (zero per-step LoRA cost). The
        original weights are restored on undo by reloading from disk.
        """
        weight = self._dequantize()
        self._set_int8(self._quantize(weight + delta.to(weight.dtype)))

    def _dequantize(self) -> torch.Tensor:
        """Return the BF16 dequantized weight (un-rotating ConvRot if active)."""
        if self.convrot:
            return torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
                self.qweight, self.scale, self.convrot_groupsize, self._out_dtype_code
            )
        return torch.ops.comfy_kitchen.dequantize_int8_simple_dtype(
            self.qweight, self.scale, self._out_dtype_code
        )

    def _quantize(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Requantize a BF16 weight to INT8 with this layer's ConvRot profile."""
        if self.convrot:
            return torch.ops.comfy_kitchen.quantize_int8_convrot_weight(
                weight, self.convrot_groupsize
            )
        return torch.ops.comfy_kitchen.quantize_int8_rowwise(weight)

    def _set_int8(self, quantized: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Overwrite ``qweight``/``scale`` buffers in place (preserving identity)."""
        q, s = quantized
        self.qweight.copy_(q)
        self.scale.copy_(s)

    def forward(self, x: torch.Tensor, input_act: Optional[str] = None) -> torch.Tensor:
        if self._int8:
            # Call the raw custom op (traceable by torch.compile) rather than the
            # registry-dispatching comfy_kitchen.int8_linear Python wrapper. A
            # LoRA is baked into ``qweight`` at switch time, so this is the whole
            # forward — no extra residual branch at runtime.
            return torch.ops.comfy_kitchen.int8_linear(
                x,
                self.qweight,
                self.scale,
                self.bias,
                self._out_dtype_code,
                self.convrot,
                self.convrot_groupsize,
                input_act,
            )
        return F.linear(x, self.weight, self.bias)


__all__ = ["QuantizedLinear"]
