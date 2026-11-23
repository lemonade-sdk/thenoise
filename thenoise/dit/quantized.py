"""Reusable quantized linear projection.

``QuantizedLinear`` is a drop-in replacement for ``nn.Linear``. A layer runs either
BF16 (``weight`` is an ``nn.Parameter``) or a quantized scheme, in which case
``weight`` is a ``comfy_kitchen.tensor.QuantizedTensor`` buffer — it cannot be a
Parameter, since PyTorch forbids gradients on integer tensors.

The module is layout-agnostic: the GEMM goes to whichever kernel the weight's layout
answers with. Two of those answers are plain Python functions whose device backends
take raw pointers, so they run inside a ``torch.library`` op here rather than under
``torch.compile``'s trace -- see ``_LINEAR_OPS``. A full-precision weight stays an
ordinary ``F.linear``, where inductor can fuse it with its neighbours.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from comfy_kitchen.tensor import (
    AsymW4A8Int8Layout,
    QuantizedTensor,
    TensorCoreFP8Layout,
    TensorWiseINT8Layout,
    w4a8_int8_linear,
)

from thenoise.utils.loader import restore_quantized_layer
from thenoise.utils.lora import LoraFactors, LoraMode

#: What a quantized GEMM emits, carried across the op boundary as an index because
#: ``torch.library`` has no schema type for ``torch.dtype``.
GEMM_DTYPES: tuple[torch.dtype, ...] = (torch.bfloat16, torch.float16, torch.float32)


@torch.library.custom_op("thenoise::w4a8_linear", mutates_args=())
def _w4a8_linear(
    x: torch.Tensor,
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: Optional[torch.Tensor],
    correction: Optional[torch.Tensor],
    decode_lut: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: int,
    group_size: int,
    convrot_groupsize: int,
    mma_packed: bool,
) -> torch.Tensor:
    """The grouped W4A8 GEMM, opaque to dynamo (see ``_LINEAR_OPS``)."""
    return w4a8_int8_linear(
        x, qdata, s_rel, s_channel, codebook=codebook, correction=correction, bias=bias,
        group_size=group_size, convrot_groupsize=convrot_groupsize,
        out_dtype=GEMM_DTYPES[out_dtype], mma_packed=mma_packed, decode_lut=decode_lut,
    )


@_w4a8_linear.register_fake
def _(
    x: torch.Tensor,
    qdata: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    codebook: Optional[torch.Tensor],
    correction: Optional[torch.Tensor],
    decode_lut: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out_dtype: int,
    group_size: int,
    convrot_groupsize: int,
    mma_packed: bool,
) -> torch.Tensor:
    return torch.empty(*x.shape[:-1], qdata.shape[0],
                       dtype=GEMM_DTYPES[out_dtype], device=x.device)


def _fp8_tensor(data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> QuantizedTensor:
    """Re-wrap FP8 storage so the layout's own linear dispatch answers for it."""
    return QuantizedTensor(data, "TensorCoreFP8Layout", TensorCoreFP8Layout.Params(
        scale=scale, orig_dtype=dtype, orig_shape=tuple(data.shape)))


@torch.library.custom_op("thenoise::fp8_linear", mutates_args=())
def _fp8_linear(
    x: torch.Tensor,
    x_scale: Optional[torch.Tensor],
    qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: int,
) -> torch.Tensor:
    """The FP8 GEMM, opaque to dynamo (see ``_LINEAR_OPS``). ``x`` is FP8 storage under
    ``x_scale`` when the activations were quantized, and the activation itself when they
    were not -- which is what decides between the FP8 kernel and a dequantized matmul."""
    dtype = GEMM_DTYPES[out_dtype]
    if x_scale is not None:
        x = _fp8_tensor(x, x_scale, dtype)
    return F.linear(x, _fp8_tensor(qdata, weight_scale, dtype), bias)


@_fp8_linear.register_fake
def _(
    x: torch.Tensor,
    x_scale: Optional[torch.Tensor],
    qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: int,
) -> torch.Tensor:
    return torch.empty(*x.shape[:-1], qdata.shape[0],
                       dtype=GEMM_DTYPES[out_dtype], device=x.device)


#: Layouts whose GEMM this module hides from dynamo.
#:
#: comfy_kitchen already exposes INT8 as a fake-registered custom op
#: (``comfy_kitchen::int8_linear``), so an INT8 weight is one opaque node on its own and
#: stays on ``F.linear``. W4A8 and FP8 reach their kernels through plain functions: under
#: ``torch.compile`` dynamo inlines them and the backend then asks a FakeTensor for its
#: data pointer -- a hard error for W4A8, and for FP8 an exception their own handler
#: swallows into a permanently dequantized graph.
#:
#: The ops take storage tensors, never the ``QuantizedTensor``: a tensor subclass crossing
#: a custom-op boundary arrives already dequantized, which would defeat the point silently.
_LINEAR_OPS = (AsymW4A8Int8Layout, TensorCoreFP8Layout)


#: Row pitches, in bytes, from which a full-precision GEMM pays for the alignment of its
#: operand, and the elements added to a row once it does.
#:
#: The penalty follows the reduction length, because that is the activation's row pitch.
#: Measured on gfx1151 across the blocks the engine ships: rows 32768 B apart cost about
#: half the rate the same block charges elsewhere — LTX's and krea2's 16384-wide FFN
#: activations, 1.48x and 1.54x in eager and 8-13% of the block — and rows 24576 B apart
#: about a sixth (flux2_klein9b and qwen_image21: 2.6% and 5.3%). Nothing at or below
#: 16384 B responded. Neither did rows 27648 B apart, wan22's, so the threshold does buy a
#: copy there: about 0.6% of that GEMM, against the sixth recovered wherever the pitch is
#: 24576.
#:
#: The gate is the backend rather than a device list because only ROCm is known to have the
#: penalty; on cuBLAS the copy would be pure cost. The quantized paths are untouched, since
#: what was measured is the rate of a GEMM reading a full-precision activation.
PITCH_MIN_BYTES = 24576
PITCH_PAD_ELEMENTS = 8
PITCH_PENALTY = torch.version.hip is not None


def pitch_too_wide(x: torch.Tensor) -> bool:
    """Whether ``x``'s rows are long enough, and laid out squarely enough, to want widening.

    Contiguity is part of the test: a row already widened by an earlier projection has a
    stride that says so, and copying behind someone else's copy earns nothing.
    """
    if x.dim() < 2:
        return False
    pitch = x.stride(-2) * x.element_size()
    return pitch >= PITCH_MIN_BYTES and pitch == x.shape[-1] * x.element_size()


def pitch_padded(x: torch.Tensor) -> torch.Tensor:
    """``x`` in rows ``PITCH_PAD_ELEMENTS`` wider, when its rows are wide in the slow way.

    Sixteen bytes per row break whatever bank or set conflict costs the GEMM its rate, for
    one pointwise write — which Inductor folds into the kernel that produced the activation,
    so in a compiled block this is usually no extra kernel at all but a wider allocation.
    """
    if not PITCH_PENALTY or x.device.type != "cuda" or not pitch_too_wide(x):
        return x
    wide = torch.empty((*x.shape[:-1], x.shape[-1] + PITCH_PAD_ELEMENTS),
                       dtype=x.dtype, device=x.device)
    wide[..., :x.shape[-1]] = x
    return wide[..., :x.shape[-1]]


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
        # The input scale belongs to the quantized layer, not to its loader.
        self.register_buffer("input_scale", None, persistent=False)
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

    def load_quantized(
        self, qt: QuantizedTensor, *, input_scale: torch.Tensor | None = None,
    ) -> None:
        """Install a quantized weight and its optional calibrated FP8 input scale.

        The weight becomes a buffer; the non-persistent input scale follows it
        across devices without becoming part of the checkpoint state.
        """
        del self.weight
        self.register_buffer("weight", qt)
        self._quantized = True
        self.input_scale = input_scale.reshape(()) if input_scale is not None else None

    # ------------------------------------------------------------------- LoRA

    def quant_step(self) -> Optional[torch.Tensor]:
        """This layer's requantization step per output row, or None if unknown.

        The int8 exporter stores ``scale = absmax / 127`` per row (ConvRot rotates
        within groups and an orthogonal rotation keeps norms), so the stored scale
        *is* the step; a per-tensor scale broadcasts.

        The grouped W4A8 export decodes a stored codebook under a per-group and a
        per-row scale, so its step is the codebook's mean spacing under the row's mean
        group scale. That puts the two layouts' steps on one footing: measured on real
        weights, each grid stops carrying a baked delta at about the same multiple of
        its own step. A 6-bit export ships no codebook, so there is nothing to measure.
        """
        qt = self.weight
        if not isinstance(qt, QuantizedTensor):
            return None
        if qt.layout_cls is TensorWiseINT8Layout:
            scale = getattr(qt.params, "scale", None)
            if not isinstance(scale, torch.Tensor) or scale.numel() not in (1, self.out_features):
                return None
            step = scale.detach().to(torch.float32).reshape(-1)
            return step.expand(self.out_features) if step.numel() == 1 else step
        if qt.layout_cls is AsymW4A8Int8Layout:
            params = qt.params
            book = params.codebook
            if (params.s_channel is None or book is None or book.numel() < 2
                    or params.scale.shape[0] != self.out_features):
                return None
            spacing = (book.max() - book.min()) / (book.numel() - 1)
            step = params.s_channel * params.scale.float().mean(dim=1) * spacing
            return step.detach().to(torch.float32).reshape(-1)
        return None

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

    def undo_lora(self, checkpoint: Optional[str], raw_key: Optional[str]) -> None:
        """Undo a baked LoRA by reloading this layer's originals from the checkpoint."""
        if not checkpoint or not raw_key:
            raise RuntimeError(
                "cannot undo a baked quantized LoRA: no checkpoint was recorded behind"
                " this layer, so it has no originals to reload"
            )
        restore_quantized_layer(self, checkpoint, raw_key)

    def _set_quantized(self, qt: QuantizedTensor) -> None:
        """Overwrite the quantized ``weight`` buffer in place (preserving identity)."""
        self.weight.copy_(qt)

    def _lora_branch(self, x: torch.Tensor) -> torch.Tensor:
        """The runtime branch alone; ``F.linear`` keeps it on the layer's GEMM path."""
        x = x.to(self._lora_down.dtype)
        return F.linear(F.linear(x, self._lora_down), self._lora_up)

    def _gemm(self, x: torch.Tensor, weight: QuantizedTensor, bias: Optional[torch.Tensor],
              out_dtype: torch.dtype) -> torch.Tensor:
        """``x @ weight.T + bias`` for a quantized weight, opaque where dynamo needs it.

        A layout outside ``_LINEAR_OPS`` (or one the file stored transposed, which no op
        here reads) keeps the plain dispatch, which is correct in eager and fails loudly
        under ``torch.compile`` rather than building the wrong graph.
        """
        params = weight.params
        if weight.layout_cls not in _LINEAR_OPS or getattr(params, "transposed", False):
            return F.linear(x, weight, bias)
        if weight.layout_cls is AsymW4A8Int8Layout:
            qdata, s_rel, s_channel, correction, codebook = (
                AsymW4A8Int8Layout.get_plain_tensors(weight))
            return torch.ops.thenoise.w4a8_linear(
                x, qdata, s_rel, s_channel, codebook, correction, params.decode_lut, bias,
                GEMM_DTYPES.index(out_dtype), params.group_size, params.convrot_groupsize,
                params.mma_packed)
        if weight.layout_cls is TensorCoreFP8Layout:
            qdata, weight_scale = TensorCoreFP8Layout.get_plain_tensors(weight)
            activations, x_scale = (x, None)
            if isinstance(x, QuantizedTensor) and x.layout_cls is TensorCoreFP8Layout:
                activations, x_scale = TensorCoreFP8Layout.get_plain_tensors(x)
            return torch.ops.thenoise.fp8_linear(
                activations, x_scale, qdata, weight_scale, bias,
                GEMM_DTYPES.index(out_dtype))
        return F.linear(x, weight, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if not isinstance(weight, QuantizedTensor):
            # Full precision stays an ordinary ``F.linear``: an op boundary here would cut
            # the layer off from the elementwise fusion around it. Widening the operand's
            # rows is not one — it is pointwise, it fuses into whatever produced them, and
            # it is this GEMM's own rate that wants it.
            y = F.linear(pitch_padded(x), weight, self.bias)
        elif self.input_scale is not None:
            # Only quantized layers loaded with calibrated FP8 input scaling take
            # this path; all others use the standard linear dispatch.
            scale = self.input_scale
            original_shape = x.shape
            flat = x.reshape(-1, original_shape[-1])
            # FP8 matrix kernels require FP16/BF16 bias, independent of
            # the checkpoint's original bias dtype.
            compute_dtype = (
                flat.dtype if flat.dtype in (torch.float16, torch.bfloat16)
                else torch.bfloat16
            )
            qx = QuantizedTensor.from_float(
                flat.to(compute_dtype), "TensorCoreFP8Layout", scale=scale
            )
            bias = self.bias.to(compute_dtype) if self.bias is not None else None
            y = self._gemm(qx, weight, bias, compute_dtype)
            if self._lora_down is not None:
                y = y + self._lora_branch(flat)
            return y.to(x.dtype).reshape(*original_shape[:-1], self.out_features)
        else:
            y = self._gemm(x, weight, self.bias, weight.params.orig_dtype)
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
