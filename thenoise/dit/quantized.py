"""Reusable quantized linear projection.

``QuantizedLinear`` is a drop-in replacement for ``nn.Linear``. A layer runs
either BF16 (default; ``weight`` is an ``nn.Parameter``) or a quantized scheme,
in which case ``weight`` becomes a ``comfy_kitchen.tensor.QuantizedTensor``
buffer. The module is fully layout-agnostic: ``forward`` calls ``F.linear``, and
the ``QuantizedTensor``'s ``__torch_dispatch__`` routes the GEMM to the right
kernel for its layout (INT8+ConvRot, FP8, NVFP4, MXFP8, ...). Quantized layers
emit the activation dtype, so everything downstream is dtype-agnostic and the
rest of the model needs no changes.

Quantized weights live inside a ``QuantizedTensor`` buffer (not an
``nn.Parameter``) because PyTorch forbids gradients on integer tensors — and a
``QuantizedTensor`` subclass cannot be a Parameter at all. Loaders should use
``thenoise.utils.loader.load_quantized_state_dict`` to populate the model from a
ComfyUI-style checkpoint.

LoRAs on quantized layers take one of two paths, picked per layer by
``thenoise.utils.lora`` from how wide the delta is against the layer's
requantization step (see ``QuantizedLinear.quant_step``):

* **Baked** (``bake_lora``, the default): the weight is dequantized to BF16, the
  LoRA delta is added, and the result is requantized with the layer's preserved
  layout profile (``requantize_from_float`` carries over the ConvRot flag, group
  size, and scale granularity). The runtime forward is a single quantized GEMM
  with zero per-step LoRA cost. Undo reloads the original weights from the
  checkpoint file (see ``thenoise.utils.loader.build_quantized_restore_map`` /
  ``restore_quantized_layer``).
* **Runtime** (``set_runtime_lora``): the weight is left untouched and the LoRA
  runs as a low-rank add-on to the quantized GEMM. Used when the delta is finer
  than the quantization step, where baking would round it away and land
  requantization noise instead of the LoRA. Undo just drops the factors.

The forward is layout-agnostic either way.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout

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
        # Runtime LoRA branch (see ``set_runtime_lora``). Non-persistent buffers so
        # they follow the module across devices/dtype casts but never reach a
        # state_dict. ``None`` slots registered up front keep ``forward`` a plain
        # attribute check instead of a ``getattr``.
        self.register_buffer("_lora_down", None, persistent=False)
        self.register_buffer("_lora_up", None, persistent=False)

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

    def quant_step(self) -> Optional[torch.Tensor]:
        """This layer's requantization step per output row, in weight units.

        The step is what a baked-in LoRA has to be wider than to survive: a delta
        below it rounds back to the stored codes, while the requantization
        re-derives the scales and re-rounds every other entry of the row too.
        ``thenoise.utils.lora.lora_bake_ratio`` compares a delta against it.

        The int8 exporter stores ``scale = absmax / 127`` per row (ConvRot rotates
        within groups but still scales per row, and an orthogonal rotation keeps
        norms, so the rotated step is a fair yardstick for an unrotated delta), so
        the stored scale *is* the step. Per-tensor int8 stores a 0-dim scale,
        broadcast to one value per row.

        Returns ``None`` for layouts with no cheap step estimate (anything but
        int8 — FP8/MXFP8/NVFP4 scale relative to each weight rather than to the
        row absmax, so reading a step off their single scale would be a guess).
        ``None`` means "no opinion": the caller bakes, which is what this module
        always did.
        """
        qt = self.weight
        if not isinstance(qt, QuantizedTensor) or qt.layout_cls is not TensorWiseINT8Layout:
            return None
        scale = getattr(qt.params, "scale", None)
        if not isinstance(scale, torch.Tensor) or scale.numel() == 0:
            return None
        step = scale.detach().to(torch.float32).reshape(-1)
        if step.numel() == 1:
            step = step.expand(self.out_features)
        elif step.numel() != self.out_features:
            # Not the per-row convention this reads (a per-group scale, say); a
            # wrong step is worse than no step.
            return None
        return step

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

        Only pays off when the delta is wide against ``quant_step()``; that check
        lives in ``thenoise.utils.lora``, which routes deltas that would not
        survive to ``set_runtime_lora`` instead.
        """
        if self._lora_down is not None:
            raise RuntimeError(
                "cannot bake a LoRA into a layer that already has a runtime LoRA"
            )
        qt = self.weight
        weight = qt.dequantize()
        self._set_quantized(qt.requantize_from_float(weight + delta.to(weight.dtype)))

    def set_runtime_lora(self, down: torch.Tensor, up: torch.Tensor) -> None:
        """Apply a LoRA at runtime as a low-rank add-on to the quantized GEMM.

        Args:
            down: ``[r, in_features]`` (the ``lora_down``/``lora_A`` factor).
            up: ``[out_features, r]`` with the LoRA's ``alpha/r * multiplier``
                already folded in.

        ``forward`` adds ``(x @ down.T) @ up.T`` to the quantized result, so the
        stored weight keeps its own quantization grid instead of swallowing a
        delta finer than its requantization step. Costs two rank-sized GEMMs per
        step and ``r * (in + out)`` of memory, leaves the weight untouched, and
        undoes by dropping the factors. Several LoRAs on one layer go in as one
        concatenated pair: the rank axis is summed over, so that is their exact
        sum.
        """
        self._lora_down = down.detach().to(self.weight.device, self.weight.dtype)
        self._lora_up = up.detach().to(self.weight.device, self.weight.dtype)

    def clear_runtime_lora(self) -> None:
        """Drop any runtime LoRA branch (the weight was never touched, so this
        restores the layer exactly)."""
        self._lora_down = None
        self._lora_up = None

    def _set_quantized(self, qt: QuantizedTensor) -> None:
        """Overwrite the quantized ``weight`` buffer in place (preserving identity)."""
        self.weight.copy_(qt)

    def apply_lora(self, delta: torch.Tensor) -> None:
        """Apply a LoRA delta ``[out, in]`` in place.

        BF16 layers add ``delta`` to the weight parameter directly. Quantized
        layers bake it in (``bake_lora``), so the runtime forward stays a single
        quantized GEMM with zero per-step LoRA cost. A caller that can decide per
        layer (``thenoise.utils.lora.apply_lora_to_model``) sends deltas that a
        bake would destroy to ``set_runtime_lora`` instead of calling this.
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
        unused for quantized layers. A runtime LoRA branch is dropped either way:
        it never touched the weight, so clearing it is the whole undo.
        """
        self.clear_runtime_lora()
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

    def _lora_branch(self, x: torch.Tensor) -> torch.Tensor:
        """The runtime LoRA add-on alone: ``(x @ down.T) @ up.T``.

        Two ``F.linear`` calls rather than raw matmuls, because that is exactly what
        they are (``down`` is ``[r, in]``, ``up`` is ``[out, r]``) — which puts the
        branch on the same GEMM path, and under the same Inductor epilogue fusion,
        as the layer's own projection. Kept as a method so a test can assert on
        precisely what ``forward`` adds.
        """
        x = x.to(self._lora_down.dtype)
        return F.linear(F.linear(x, self._lora_down), self._lora_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self._lora_down is not None:
            # Runtime LoRA branch (``set_runtime_lora``).
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
