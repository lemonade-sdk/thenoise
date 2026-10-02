"""Reusable quantized linear projection.

``QuantizedLinear`` is a drop-in replacement for ``nn.Linear``. A layer runs either
BF16 (``weight`` is an ``nn.Parameter``) or a quantized scheme, in which case
``weight`` is a ``comfy_kitchen.tensor.QuantizedTensor`` buffer — it cannot be a
Parameter, since PyTorch forbids gradients on integer tensors.

The module is layout-agnostic: ``forward`` calls ``F.linear`` and the tensor's
``__torch_dispatch__`` routes the GEMM to the right kernel for its layout.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

from thenoise.utils.loader import restore_quantized_layer
from thenoise.utils.lora import LoraFactors, LoraMode


class QuantizedLinear(nn.Module):
    """Linear projection that runs BF16 (default) or a quantized scheme.

    A LoRA is baked into the weight, or kept as a low-rank branch on top of the
    quantized GEMM when its delta is finer than this layer's requantization step.
    """

    #: Below this delta-RMS / ``quant_step()`` ratio a bake would not survive.
    #: Calibrated against real INT8+ConvRot checkpoints.
    LORA_BAKE_MIN_RATIO = 0.1

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Cast to the compute dtype by the adapter, or replaced at load time.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self._reset_parameters()
        self._quantized = False
        # Runtime LoRA branch: non-persistent buffers, so they follow the module
        # across devices and dtype casts but never reach a state_dict.
        self.register_buffer("_lora_down", None, persistent=False)
        self.register_buffer("_lora_up", None, persistent=False)

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def load_quantized(self, qt: QuantizedTensor) -> None:
        """Switch this layer to a pre-quantized weight of any layout.

        Frees the BF16 ``weight`` parameter and registers ``qt``, which carries its
        own layout profile, as the ``weight`` buffer.
        """
        del self.weight
        self.register_buffer("weight", qt)
        self._quantized = True

    # ------------------------------------------------------------------- LoRA

    def quant_step(self) -> Optional[torch.Tensor]:
        """This layer's requantization step per output row, or None if unknown.

        The int8 exporter stores ``scale = absmax / 127`` per row (ConvRot rotates
        within groups and an orthogonal rotation keeps norms), so the stored scale
        *is* the step; a per-tensor scale broadcasts.
        """
        qt = self.weight
        if not isinstance(qt, QuantizedTensor) or qt.layout_cls is not TensorWiseINT8Layout:
            return None
        scale = getattr(qt.params, "scale", None)
        if not isinstance(scale, torch.Tensor) or scale.numel() not in (1, self.out_features):
            return None
        step = scale.detach().to(torch.float32).reshape(-1)
        return step.expand(self.out_features) if step.numel() == 1 else step

    @staticmethod
    def _bake_ratio(factors: LoraFactors, step: torch.Tensor) -> torch.Tensor:
        """How wide a LoRA is against this layer's step: ``rms(delta) / step``.

        Row ``j`` of ``up @ down`` has squared norm ``up_j (down @ down.T) up_j.T``,
        so the row norms cost two ``[out, r] x [r, r]`` products instead of building
        the ``[out, in]`` delta. Only rows the delta actually touches are counted: a
        fused ``qkv``/``gate_up`` receives a partial delta.

        NaN (a zero delta) reads as "bake".
        """
        down = factors.down.to(torch.float32)
        up = factors.up.to(torch.float32)
        step = step.to(down.device).reshape(-1)
        row_sq = ((up @ (down @ down.t())) * up).sum(dim=1).clamp_min_(0)
        keep = (row_sq > 0) & (step > 0)
        rows = keep.sum().clamp_min(1)
        rms = (row_sq * keep).sum().div(rows).div(down.size(1)).sqrt()
        return rms / (step * keep).sum().div(rows)

    def apply_lora(self, factors: LoraFactors) -> LoraMode:
        """Carry a LoRA the best way this layer can, and report which way that was.

        Baking is free at every step, so it stays the default; only a delta finer
        than ``quant_step()`` goes to the runtime branch.
        """
        self.clear_runtime_lora()  # a leftover branch would stack on top of a bake
        if not self._quantized:
            self.weight.data.add_(factors.delta(self.weight.device, self.weight.dtype))
            return LoraMode.BAKED

        step = self.quant_step()
        if step is not None and self._bake_ratio(factors, step) < self.LORA_BAKE_MIN_RATIO:
            self.set_runtime_lora(factors.down, factors.up)
            return LoraMode.RUNTIME
        self.bake_lora(factors.delta(self.weight.device, self.weight.dtype))
        return LoraMode.BAKED_QUANTIZED

    def bake_lora(self, delta: torch.Tensor) -> None:
        """Bake a ``[out, in]`` delta into the quantized weight (any layout).

        Dequantizes, adds, and requantizes with this layer's preserved layout
        profile, so the runtime forward stays a single quantized GEMM.
        """
        if self._lora_down is not None:
            raise RuntimeError(
                "cannot bake a LoRA into a layer that already has a runtime LoRA"
            )
        qt = self.weight
        weight = qt.dequantize()
        self._set_quantized(qt.requantize_from_float(weight + delta.to(weight.dtype)))

    def set_runtime_lora(self, down: torch.Tensor, up: torch.Tensor) -> None:
        """Keep a LoRA as a low-rank add-on to the quantized GEMM.

        Costs two rank-sized GEMMs per step and ``r * (in + out)`` of memory, and
        leaves the weight bit-identical.
        """
        self._lora_down = down.detach().to(self.weight.device, self.weight.dtype)
        self._lora_up = up.detach().to(self.weight.device, self.weight.dtype)

    def clear_runtime_lora(self) -> None:
        """Drop the runtime branch, which restores the layer exactly."""
        self._lora_down = None
        self._lora_up = None

    def undo_lora(self, dit_path: Optional[str], raw_key: Optional[str]) -> None:
        """Undo a baked LoRA by reloading this layer's originals from the checkpoint."""
        if not dit_path or not raw_key:
            raise RuntimeError(
                "cannot undo a baked quantized LoRA: no dit_path and raw checkpoint "
                "key were recorded at apply time"
            )
        restore_quantized_layer(self, dit_path, raw_key)

    def _set_quantized(self, qt: QuantizedTensor) -> None:
        """Overwrite the quantized ``weight`` buffer in place (preserving identity)."""
        self.weight.copy_(qt)

    def _lora_branch(self, x: torch.Tensor) -> torch.Tensor:
        """The runtime branch alone; ``F.linear`` keeps it on the layer's GEMM path."""
        x = x.to(self._lora_down.dtype)
        return F.linear(F.linear(x, self._lora_down), self._lora_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._lora_down is not None:
            y = y + self._lora_branch(x)
        return y


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
