"""Shared fixtures and helpers for the test-suite.

Everything here is weight-free, GPU-free and working-directory independent.
Import the plain helpers directly (``from conftest import FakeHandle``); use the
``@pytest.fixture`` ones as usual.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Optional

import torch

# Tests exercise model math, not Inductor. ``Flux2.forward`` wraps every block in
# ``torch.compile(fullgraph=True)``, so the first forward in a process would pay
# the whole compile cost (~7s) to assert a shape and ``isfinite``. Disabling
# dynamo here (rather than via TORCH_COMPILE_DISABLE) cannot be silently unset by
# the environment and does not depend on this file being imported before torch.
torch._dynamo.config.disable = True

import pytest  # noqa: E402

from thenoise.dit.quantized import QuantizedLinear  # noqa: E402
from thenoise.models import (  # noqa: E402
    AnimaModel,
    FluxKleinModel,
    Krea2Model,
    QwenImageModel,
    ZImageModel,
)
from thenoise.models.base import Conditioning, DiffusionModel  # noqa: E402
from thenoise.models.config import ModelConfig, SamplingParams  # noqa: E402
from thenoise.samplers import Step  # noqa: E402
from thenoise.vae import AutoencoderKLFlux2  # noqa: E402


# --------------------------------------------------------------- safetensors helpers


class FakeHandle:
    """Mimics ``safetensors.safe_open``'s ``keys()`` for detection tests."""

    def __init__(self, keys):
        self._keys = list(keys)

    def keys(self):
        return self._keys


def write_safetensors(path, tensors: dict) -> str:
    """Write ``tensors`` as a safetensors file; return the path as ``str``."""
    from safetensors.torch import save_file

    save_file(tensors, str(path))
    return str(path)


def write_key_checkpoint(path, keys) -> str:
    """Write a throwaway checkpoint carrying one zero scalar per key name.

    Detection reads names only, so the values never matter.
    """
    return write_safetensors(path, {k: torch.zeros(1) for k in keys})


def comfy_quant(convrot: bool = True, groupsize: int = 256) -> torch.Tensor:
    """Build a ``comfy_quant`` marker tensor like ComfyUI's INT8 exporter."""
    conf = {"convrot": bool(convrot)}
    if convrot:
        conf["convrot_groupsize"] = int(groupsize)
    conf["per_row"] = True
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


# The canonical tiny DiT layout used across the quantized tests: one layer that
# loads quantized (``q``) and one that stays full precision (``plain``).
TINY_DIM = (512, 256)  # (out_features, in_features)


def int8_tensors(
    *,
    convrot: bool = True,
    groupsize: int = 256,
    marker="default",
    weight_dtype: torch.dtype = torch.int8,
    scale: Optional[torch.Tensor] = None,
    prefix: str = "",
    extra: Optional[dict] = None,
    drop: tuple[str, ...] = (),
) -> dict:
    """Tensor set of the canonical 5-tensor quantized mini checkpoint."""
    if weight_dtype == torch.int8:
        qweight, default_scale = int8_pair()
    else:
        qweight, default_scale = fp8_pair()
    scale = default_scale if scale is None else scale

    tensors = {
        "q.weight": qweight,
        "q.weight_scale": scale,
        "plain.weight": torch.randn(*TINY_DIM, dtype=torch.bfloat16),
        "plain.bias": torch.randn(TINY_DIM[0], dtype=torch.bfloat16),
    }
    if marker == "default":
        tensors["q.comfy_quant"] = comfy_quant(convrot=convrot, groupsize=groupsize)
    elif marker is not None:
        tensors["q.comfy_quant"] = marker
    if prefix:
        tensors = {f"{prefix}{k}": v for k, v in tensors.items()}
    if extra:
        tensors.update(extra)
    for key in drop:
        tensors.pop(key, None)
    return tensors


def int8_checkpoint(path, **kwargs) -> tuple[str, dict]:
    """Write :func:`int8_tensors` to ``path``; return ``(path, tensors)``.

    Keyword arguments are forwarded to :func:`int8_tensors`, so a test can ask for
    an FP8 / unparseable-marker / wrapper-prefixed variant of the same fixture.
    """
    tensors = int8_tensors(**kwargs)
    return write_safetensors(path, tensors), tensors


def bf16_tensors(*, prefix: str = "", extra: Optional[dict] = None) -> dict:
    """The full-precision counterpart of :func:`int8_tensors`."""
    tensors = {
        "q.weight": torch.randn(*TINY_DIM, dtype=torch.bfloat16),
        "plain.weight": torch.randn(*TINY_DIM, dtype=torch.bfloat16),
        "plain.bias": torch.randn(TINY_DIM[0], dtype=torch.bfloat16),
    }
    if prefix:
        tensors = {f"{prefix}{k}": v for k, v in tensors.items()}
    if extra:
        tensors.update(extra)
    return tensors


def int8_lora_state_dict(module: str = "q", rank: int = 8) -> dict:
    """A LoRA state dict targeting ``module`` (sd-scripts style factor names)."""
    out_f, in_f = TINY_DIM
    return {
        f"{module}.lora_down.weight": torch.randn(rank, in_f, dtype=torch.bfloat16),
        f"{module}.lora_up.weight": torch.randn(out_f, rank, dtype=torch.bfloat16),
        f"{module}.alpha": torch.tensor(float(rank)),
    }


class TinyDiT(torch.nn.Module):
    """Smallest DiT shape the quantized loader/LoRA paths accept.

    Layout matches :func:`int8_tensors` (``q`` loads quantized, ``plain`` stays
    full precision). ``quantized=True`` pre-loads ``q`` with an INT8 weight for
    the tests that never touch a checkpoint file.
    """

    def __init__(self, quantized: bool = False):
        super().__init__()
        out_f, in_f = TINY_DIM
        self.q = QuantizedLinear(in_f, out_f, bias=False)
        self.plain = QuantizedLinear(in_f, out_f, bias=True)
        if quantized:
            self.q.load_quantized(wrapped_int8_tensor())


def int8_qt(qweight, scale, convrot: bool = True, groupsize: int = 256):
    """Wrap an int8 weight + per-row scale into a ``TensorWiseINT8Layout`` tensor."""
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

    params = TensorWiseINT8Layout.Params(
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=tuple(qweight.shape),
        is_weight=True,
        convrot=convrot,
        convrot_groupsize=groupsize,
    )
    return QuantizedTensor(qweight, "TensorWiseINT8Layout", params)


def fp8_qt(qweight, scale):
    """Wrap an fp8 weight + per-tensor scale into a ``TensorCoreFP8Layout`` tensor."""
    from comfy_kitchen.tensor import QuantizedTensor
    from comfy_kitchen.tensor.fp8 import TensorCoreFP8Layout

    params = TensorCoreFP8Layout.Params(
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=tuple(qweight.shape),
    )
    return QuantizedTensor(qweight, "TensorCoreFP8Layout", params)


def fp8_pair(shape=TINY_DIM):
    """A random bf16 weight quantized to fp8 E4M3 plus its per-tensor scale."""
    bf16 = torch.randn(*shape, dtype=torch.bfloat16)
    scale = torch.tensor(bf16.abs().amax().item() / 448.0, dtype=torch.float32)
    return bf16.to(torch.float8_e4m3fn), scale


def int8_pair(shape=TINY_DIM):
    """A random int8 weight plus a per-row scale."""
    return (
        torch.randint(-127, 127, shape, dtype=torch.int8),
        torch.rand(shape[0], 1, dtype=torch.float32),
    )


def wrapped_int8_tensor(convrot: bool = True, groupsize: int = 256):
    """A ready-made ``TensorWiseINT8Layout`` QuantizedTensor matching :data:`TINY_DIM`."""
    qweight, scale = int8_pair()
    return int8_qt(qweight, scale, convrot=convrot, groupsize=groupsize)


def wrapped_fp8_tensor():
    """A ready-made ``TensorCoreFP8Layout`` QuantizedTensor matching :data:`TINY_DIM`."""
    qweight, scale = fp8_pair()
    return fp8_qt(qweight, scale)


# ------------------------------------------------------------------- model key-sets

# Key-set name -> (owning model class or ``None`` for "must match nothing", keys).
# The ``*_wrapped`` variants are the same checkpoints repackaged under ComfyUI's
# generic ``model.diffusion_model.`` wrapper: detection must strip the prefix
# (or the model fails to resolve and another falsely claims it).
MODEL_KEYSETS: dict[str, tuple[Optional[type[DiffusionModel]], list[str]]] = {
    "anima": (
        AnimaModel,
        [
            "model.diffusion_model.blocks.0.adaln_modulation_self_attn.1.weight",
            "model.diffusion_model.llm_adapter.layers.0.weight",
            "model.diffusion_model.x_embedder.linear.weight",
        ],
    ),
    # A generic adaLN DiT WITHOUT Anima's LLM adapter must not be claimed as Anima.
    "adaln_only": (
        None,
        [
            "model.diffusion_model.blocks.0.adaln_modulation_self_attn.1.weight",
            "model.diffusion_model.blocks.0.mlp.gate.weight",
        ],
    ),
    "krea2": (
        Krea2Model,
        [
            "x_embedder.linear.weight",
            "txtfusion.layerwise_blocks.0.attn.q_proj.weight",
            "blocks.0.mod.lin.weight",
            "txtmlp.1.weight",
        ],
    ),
    "krea2_wrapped": (
        Krea2Model,
        [
            "model.diffusion_model.x_embedder.linear.weight",
            "model.diffusion_model.txtfusion.layerwise_blocks.0.attn.q_proj.weight",
            "model.diffusion_model.blocks.0.mod.lin.weight",
            "model.diffusion_model.txtmlp.1.weight",
        ],
    ),
    "zimage": (
        ZImageModel,
        [
            "x_embedder.weight",
            "cap_embedder.1.weight",
            "context_refiner.0.attention_norm1.weight",
            "t_embedder.mlp.0.weight",
            "layers.0.attention_norm1.weight",
        ],
    ),
    "zimage_wrapped": (
        ZImageModel,
        [
            "model.diffusion_model.x_embedder.weight",
            "model.diffusion_model.cap_embedder.1.weight",
            "model.diffusion_model.context_refiner.0.attention_norm1.weight",
            "model.diffusion_model.layers.0.attention_norm1.weight",
        ],
    ),
    "flux_klein": (
        FluxKleinModel,
        [
            "double_stream_modulation_img.lin.weight",
            "double_stream_modulation_txt.lin.weight",
            "single_stream_modulation.lin.weight",
            "img_in.weight",
            "txt_in.weight",
            "final_layer.adaLN_modulation.1.weight",
        ],
    ),
    "flux_klein_wrapped": (
        FluxKleinModel,
        [
            "model.diffusion_model.double_stream_modulation_img.lin.weight",
            "model.diffusion_model.double_stream_modulation_txt.lin.weight",
            "model.diffusion_model.single_stream_modulation.lin.weight",
            "model.diffusion_model.img_in.weight",
            "model.diffusion_model.final_layer.adaLN_modulation.1.weight",
        ],
    ),
    "qwen_image": (
        QwenImageModel,
        [
            "img_in.weight",
            "txt_in.weight",
            "time_text_embed.1.weight",
            "layers.0.img_attn.qkv.weight",
            "layers.0.txt_attn.qkv.weight",
            "layers.0.ff.img.0.weight",
            "final_layer.1.weight",
        ],
    ),
    "qwen_image_wrapped": (
        QwenImageModel,
        [
            "model.diffusion_model.img_in.weight",
            "model.diffusion_model.txt_in.weight",
            "model.diffusion_model.time_text_embed.1.weight",
            "model.diffusion_model.layers.0.img_attn.qkv.weight",
            "model.diffusion_model.layers.0.txt_attn.qkv.weight",
            "model.diffusion_model.layers.0.ff.img.0.weight",
            "model.diffusion_model.final_layer.1.weight",
        ],
    ),
    "unknown": (None, ["some.random.key", "blocks.0.attn.gate.weight"]),
}

KEYSET_IDS = sorted(MODEL_KEYSETS)
CATALOG_IDS = [cls.name for cls in (Krea2Model, AnimaModel, ZImageModel, FluxKleinModel, QwenImageModel)]


# ------------------------------------------------------------------------ stub models


class StubModel(DiffusionModel):
    """A fully functional weight-free adapter for pipeline/sampler tests.

    Every kernel is real enough to drive the pipeline end to end on CPU fp32 and
    every one of them is counted in ``self.calls`` so cache behaviour, call
    counts and the refine schedule are directly observable.
    """

    name = "stub"
    DEFAULT_WIDTH = 64
    DEFAULT_HEIGHT = 64
    DEFAULT_STEPS = 2
    DEFAULT_GUIDANCE_SCALE = 1.0
    SAMPLER = "euler"
    LATENT_CHANNELS = 4
    _VAE_SCALE = 8
    UPSCALE_SCALE = 2
    REFINE_STEPS = 1
    REFINE_DENOISE = 0.1

    def __init__(
        self,
        *,
        config: Optional[ModelConfig] = None,
        supports_edit: Optional[bool] = None,
        lora_dir: Optional[str] = None,
    ):
        super().__init__(
            config=config
            or ModelConfig(
                dit_path="dit", vae_path="vae", text_encoder_path="te",
                device="cpu", dtype=torch.float32, lora_dir=lora_dir,
            )
        )
        if supports_edit is not None:
            self.supports_edit = supports_edit
        self.calls: Counter = Counter()
        self.dit = torch.nn.Identity()  # ``switch_loras`` is handed this module
        self.sizes: list[tuple[int, int]] = []
        self.params_seen: list[SamplingParams] = []
        # ``prepare_latent`` / ``switch_loras`` call log (see those methods).
        self.prepared: list[dict] = []
        self.lora_switches: list[Optional[list]] = []
        self.encode_prompt_args = None

    @staticmethod
    def detect(f) -> bool:
        return False

    # ---------------------------------------------------------------- kernels
    def encode_prompt(self, args) -> Conditioning:
        self.calls["encode_prompt"] += 1
        self.encode_prompt_args = args
        return Conditioning(cond=torch.zeros(1, 4, 8), null=None)

    def fuse_text(self, cond):
        self.calls["fuse_text"] += 1
        return cond

    def init_latents(self, params: SamplingParams) -> torch.Tensor:
        self.calls["init_latents"] += 1
        self.sizes.append((params.width, params.height))
        gen = torch.Generator(device="cpu").manual_seed(params.seed)
        return torch.randn(
            1,
            self.LATENT_CHANNELS,
            params.height // self._VAE_SCALE,
            params.width // self._VAE_SCALE,
            generator=gen,
        )

    def prepare_latent(self, latents, cond, params, ref=None, ref_method="index"):
        self.calls["prepare_latent"] += 1
        self.prepared.append({"latents": latents, "ref": ref, "method": ref_method})
        return latents

    def schedule(self, params: SamplingParams) -> list[Step]:
        self.calls["schedule"] += 1
        self.params_seen.append(params)
        grid = torch.linspace(1.0, 0.0, params.steps + 1)
        return [
            Step(t=grid[i], delta=grid[i] - grid[i + 1]) for i in range(params.steps)
        ]

    def denoise_step(self, latents, t, cond, guidance_scale, i):
        self.calls["denoise_step"] += 1
        return torch.zeros_like(latents)

    def encode_reference(self, pixels):
        self.calls["encode_reference"] += 1
        return pixels.unsqueeze(0).float()  # [1, C, H, W]

    def decode(self, latents):
        self.calls["decode"] += 1
        h, w = latents.shape[-2:]
        return torch.zeros(3, h * self._VAE_SCALE, w * self._VAE_SCALE)

    def switch_loras(self, lora_specs, dit):
        self.calls["switch_loras"] += 1
        self.lora_switches.append(list(lora_specs) if lora_specs else None)
        return super().switch_loras(lora_specs, dit)

    def _upscale_format(self) -> str:
        return "flux2"


class EditingStubModel(StubModel):
    """A stub that declares editing support (reference-latent path)."""

    supports_edit = True


# ---------------------------------------------------------------------------- fixtures


@pytest.fixture
def stub_model_cls():
    """The minimal concrete :class:`DiffusionModel` subclass.

    Use ``object.__new__(cls)`` to get an instance without ``__init__`` when only
    the pure helper methods are under test.
    """

    class _Stub(DiffusionModel):
        name = "test"

        @staticmethod
        def detect(f):
            return False

        def encode_prompt(self, args):
            pass

        def init_latents(self, params):
            pass

        def schedule(self, params):
            pass

        def denoise_step(self, latents, t, cond, guidance_scale, i):
            pass

        def _upscale_format(self):
            return "wan21"

    return _Stub


@pytest.fixture
def fake_model_cls():
    """Factory for stand-in entries of ``thenoise.models.MODEL_CATALOG``.

    ``make(name=..., constructed=list)`` returns a class whose ``__init__`` records
    the kwargs it was built with (that is what the runtime hands to an adapter).
    """

    def make(name: str = "fake", constructed: Optional[list] = None, **attrs):
        def __init__(self, **kwargs):
            if constructed is not None:
                constructed.append(kwargs)

        namespace = {
            "name": name,
            "__init__": __init__,
            "detect": staticmethod(lambda f: False),
            **attrs,
        }
        return type("FakeModel", (), namespace)

    return make


@pytest.fixture
def stub_model():
    """A fresh :class:`StubModel` (weight-free, CPU, fp32)."""
    return StubModel()


@pytest.fixture
def editing_stub_model():
    """A fresh :class:`StubModel` with ``supports_edit`` enabled."""
    return EditingStubModel()


@pytest.fixture(scope="module")
def flux2_vae():
    """One random-init Flux.2 VAE per module (building it costs ~0.2s)."""
    return AutoencoderKLFlux2()
