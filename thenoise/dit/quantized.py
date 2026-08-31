"""Reusable quantized linear projection.

``QuantizedLinear`` is a drop-in replacement for ``nn.Linear``. A layer runs
either BF16 (default; ``weight`` is an ``nn.Parameter``) or a quantized scheme,
in which case ``weight`` becomes a ``comfy_kitchen.tensor.QuantizedTensor``
buffer. The module is layout-agnostic: the INT8+ConvRot path (the current
deployment) runs a raw-op INT8 GEMM that stays traceable inside
``torch.compile``; any other layout (FP8 E4M3/E5M2, NVFP4, MXFP8, ...) is
weight-only and falls back to dequantizing to the activation dtype and running a
plain BF16 linear. Quantized layers emit BF16 output, so everything downstream
is dtype-agnostic and the rest of the model needs no changes.

Quantized weights live inside a ``QuantizedTensor`` buffer (not an
``nn.Parameter``) because PyTorch forbids gradients on integer tensors — and a
``QuantizedTensor`` subclass cannot be a Parameter at all. Loaders should use
``thenoise.utils.loader.load_quantized_state_dict`` to populate the model from a
ComfyUI-style checkpoint.

LoRAs on quantized layers are *baked in* at switch time (``bake_lora``): the
weight is dequantized to BF16, the LoRA delta is added, and the result is
requantized with the layer's preserved layout profile (``requantize_from_float``
carries over the ConvRot flag, group size, and scale granularity). The runtime
forward is therefore a single quantized GEMM (for INT8) with zero per-step LoRA
cost. Undo reloads the original weights from the checkpoint file (see
``thenoise.utils.loader.build_quantized_restore_map`` / ``restore_quantized_layer``).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy_kitchen
from comfy_kitchen.tensor import QuantizedTensor
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout


class QuantizedLinear(nn.Module):
    """Linear projection that runs BF16 (default) or a quantized scheme."""

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
        self._convrot_groupsize = convrot_groupsize
        # float32 default (matching ``nn.Linear``); the model is cast to the
        # compute dtype (bf16) by the adapter, or the weight is replaced at load.
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
        self._quantized = False

    def _reset_parameters(self) -> None:
        """Initialize like ``nn.Linear`` (kaiming on weight, uniform on bias)."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def convrot(self) -> bool:
        """ConvRot flag of the active quantized weight (True while unquantized)."""
        return self.weight.params.convrot if self._quantized else True

    @property
    def convrot_groupsize(self) -> int:
        """Hadamard group size of the active quantized weight."""
        if self._quantized:
            return self.weight.params.convrot_groupsize
        return self._convrot_groupsize

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._quantized:
            return F.linear(x, self.weight, self.bias)
        qt = self.weight
        if qt.layout_cls is TensorWiseINT8Layout:
            # Raw-op INT8 GEMM (traceable by torch.compile); a LoRA is baked into
            # the weights at switch time, so this is the whole forward. We call
            # the custom op directly rather than the registry-dispatching
            # comfy_kitchen wrapper.
            qdata, scale = TensorWiseINT8Layout.get_plain_tensors(qt)
            return torch.ops.comfy_kitchen.int8_linear(
                x,
                qdata,
                scale,
                self.bias,
                self._out_dtype_code,
                qt.params.convrot,
                qt.params.convrot_groupsize,
            )
        # Any other layout (FP8 E4M3/E5M2, NVFP4, MXFP8, ...) is weight-only in
        # this engine: dequantize to the activation dtype and run a plain bf16
        # linear. (A faster FP8 scaled-MM activation-quant path can be added
        # behind a hardware check when validated on-target.)
        return F.linear(x, qt.dequantize(), self.bias)


__all__ = ["QuantizedLinear"]
