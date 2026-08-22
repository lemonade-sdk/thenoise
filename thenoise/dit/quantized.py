"""Reusable INT8+ConvRot linear projection.

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
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy_kitchen


class QuantizedLinear(nn.Module):
    """Linear projection that runs BF16 (default) or INT8+ConvRot."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        convrot_groupsize: int = 256,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.convrot_groupsize = convrot_groupsize
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)
        # INT8 output is always bf16; the raw op takes this as an int code (not a
        # torch.dtype), resolved once here so torch.compile never traces the dict
        # lookup. Using torch.ops.comfy_kitchen.int8_linear (a torch.library
        # custom op with a fake schema) instead of the comfy_kitchen.int8_linear
        # Python wrapper keeps the call traceable inside ``torch.compile``.
        self._out_dtype_code = comfy_kitchen.DTYPE_TO_CODE[torch.bfloat16]
        self._int8 = False
        # Rank-reduced bf16 LoRA residual branch (INT8 path only): ``lora_down``
        # is ``[r, in]`` (already scaled), ``lora_up`` is ``[out, r]``. Multiple
        # LoRAs are concatenated along the rank dim so forward keeps one fixed
        # structure. bf16 layers use the normal parameter-mutation LoRA path.
        self.lora_down: Optional[torch.Tensor] = None
        self.lora_up: Optional[torch.Tensor] = None
        self._lora = False

    def load_int8(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        convrot_groupsize: Optional[int] = None,
    ) -> None:
        """Switch this layer to INT8+ConvRot using pre-quantized weights.

        Args:
            qweight: int8 weight tensor ``[out, in]``.
            scale: per-row F32 scale ``[out]`` (or ``[out, 1]``).
            convrot_groupsize: Hadamard group size (defaults to 256).
        """
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)
        if convrot_groupsize is not None:
            self.convrot_groupsize = convrot_groupsize
        self.register_parameter("weight", None)  # free the BF16 weights
        self._int8 = True

    def apply_lora(self, down, up, alpha, multiplier, calc_device) -> None:
        """Store a rank-reduced bf16 LoRA residual branch (INT8 path only).

        Args:
            down: LoRA-down ``[r, in]`` (bf16/fp16).
            up: LoRA-up ``[out, r]`` (bf16/fp16).
            alpha: LoRA alpha (scalar, int, or 0-dim tensor).
            multiplier: LoRA weight multiplier.
            calc_device: device to store the factors on.

        The residual added in ``forward`` is ``x @ down^T @ up^T * (alpha/r) *
        multiplier``, matching ``_compute_lora_delta``'s ``multiplier * (up @
        down) * scale``. Multiple LoRAs are concatenated along the rank dim so
        the forward residual keeps a single fixed shape.
        """
        r = down.size(0)
        scale = (
            float(alpha.to(calc_device)) if isinstance(alpha, torch.Tensor) else float(alpha)
        ) / r * multiplier
        down = down.to(device=calc_device, dtype=torch.bfloat16)
        up = up.to(device=calc_device, dtype=torch.bfloat16)
        scaled_down = down * scale
        if self._lora:
            self.lora_down = torch.cat([self.lora_down, scaled_down], dim=0)
            self.lora_up = torch.cat([self.lora_up, up], dim=1)
        else:
            self.lora_down = scaled_down
            self.lora_up = up
            self._lora = True

    def clear_lora(self) -> None:
        self.lora_down = None
        self.lora_up = None
        self._lora = False

    def forward(self, x: torch.Tensor, input_act: Optional[str] = None) -> torch.Tensor:
        if self._int8:
            # Call the raw custom op (traceable by torch.compile) rather than the
            # registry-dispatching comfy_kitchen.int8_linear Python wrapper.
            out = torch.ops.comfy_kitchen.int8_linear(
                x,
                self.qweight,
                self.scale,
                self.bias,
                self._out_dtype_code,
                True,  # convrot
                self.convrot_groupsize,
                input_act,
            )
            if self._lora:
                # bf16 LoRA residual branch: x @ down^T @ up^T (scale folded into
                # lora_down). Keeps the int8 weight untouched.
                out = out + x @ self.lora_down.t() @ self.lora_up.t()
            return out
        return F.linear(x, self.weight, self.bias)


__all__ = ["QuantizedLinear"]
