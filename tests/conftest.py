"""Shared fixtures and helpers for the test-suite.

Everything here is weight-free, GPU-free and working-directory independent.
Import the plain helpers directly (``from conftest import FakeHandle``); use the
``@pytest.fixture`` ones as usual.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import torch

# Every tensor in the suite is toy-sized, and the intra-op thread pool spends more in
# barrier time than it can win back: one tiny Lumina forward is ~0.25s on 24 threads
# and ~0.01s on one, which is most of the suite's wall time. One thread is also a
# deterministic reduction order.
torch.set_num_threads(1)

# Tests exercise model math, not Inductor. ``Flux2.forward`` wraps every block in
# ``torch.compile(fullgraph=True)``, so the first forward in a process would pay
# the whole compile cost (~7s) to assert a shape and ``isfinite``. Disabling
# dynamo here (rather than via TORCH_COMPILE_DISABLE) cannot be silently unset by
# the environment and does not depend on this file being imported before torch.
torch._dynamo.config.disable = True

import pytest  # noqa: E402

from thenoise.dit.quantized import QuantizedLinear  # noqa: E402
from thenoise.models import (  # noqa: E402
    MODEL_CATALOG,
    AnimaModel,
    Fibo2Model,
    Flux3Model,
    FluxKleinModel,
    H3ImageModel,
    Krea2Model,
    MageFlowModel,
    MingImageModel,
    MossImageModel,
    QwenImageModel,
    QwenImage21Model,
    VideoModel,
    Wan22Model,
    ZImageModel,
)
from thenoise.models.base import Conditioning, DiffusionModel  # noqa: E402
from thenoise.models.config import ModelConfig, SamplingParams  # noqa: E402
from thenoise.samplers import Step  # noqa: E402
from thenoise.upscale import LatentUpscaler  # noqa: E402
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


# ------------------------------------------------------------------ Wan 2.2 helpers

#: The inverse of ``wan_dit_key``: this repository's module paths back to the names a
#: Comfy-Org Wan checkpoint carries.
_WAN_RAW_KEYS = {
    "condition_embedder.text_linear_1.": "text_embedding.0.",
    "condition_embedder.text_linear_2.": "text_embedding.2.",
    "condition_embedder.time_linear_1.": "time_embedding.0.",
    "condition_embedder.time_linear_2.": "time_embedding.2.",
    "condition_embedder.time_proj.": "time_projection.1.",
    "proj_out.": "head.head.",
    "norm_out.": "head.norm.",
    "scale_shift_table": "head.modulation",
}

_WAN_RAW_BLOCK_KEYS = (
    ("attn1.to_q.", "self_attn.q."), ("attn1.to_k.", "self_attn.k."),
    ("attn1.to_v.", "self_attn.v."), ("attn1.to_out.", "self_attn.o."),
    ("attn1.norm_q.", "self_attn.norm_q."), ("attn1.norm_k.", "self_attn.norm_k."),
    ("attn2.to_q.", "cross_attn.q."), ("attn2.to_k.", "cross_attn.k."),
    ("attn2.to_v.", "cross_attn.v."), ("attn2.to_out.", "cross_attn.o."),
    ("attn2.norm_q.", "cross_attn.norm_q."), ("attn2.norm_k.", "cross_attn.norm_k."),
    ("ffn_in.", "ffn.0."), ("ffn_out.", "ffn.2."),
)


def wan_raw_key(key: str) -> str:
    """One module path as a Wan checkpoint would spell it.

    ComfyUI's ``norm2`` is the FFN norm and ``norm3`` the cross-attention one, the
    reverse of this repository's, and a block's modulation table is its
    ``scale_shift_table``.
    """
    for dest, src in _WAN_RAW_KEYS.items():
        if key.startswith(dest):
            return src + key[len(dest):]
    if not key.startswith("blocks."):
        return key
    head, suffix = ".".join(key.split(".", 2)[:2]) + ".", key.split(".", 2)[2]
    if suffix == "scale_shift_table":
        return head + "modulation"
    if suffix.startswith("norm2."):
        return head + "norm3." + suffix[len("norm2."):]
    for dest, src in _WAN_RAW_BLOCK_KEYS:
        if suffix.startswith(dest):
            return head + src + suffix[len(dest):]
    return key


def wan_tiny_params(channels: int = 48):
    """A Wan transformer small enough to build in a test.

    Still legal geometry for ``detect_params``: one 128-wide head, two blocks.
    """
    from thenoise.dit.wan_video.model import Wan22Params

    return Wan22Params(in_channels=channels, out_channels=channels, dim=128,
                       num_layers=2, ffn_dim=256, freq_dim=32, text_dim=32,
                       patch_size=(1, 2, 2))


def write_wan_dit(path, *, channels: int = 48) -> str:
    """Write a tiny but shape-valid Wan 2.2 DiT under Comfy-Org's key names.

    Anything that asks a transformer how big it is needs real shapes in the header, so
    a throwaway file of zero scalars will not do.
    """
    from thenoise.dit.wan_video.model import WanNativeDiT

    with torch.random.fork_rng():
        torch.manual_seed(123)
        source = WanNativeDiT(wan_tiny_params(channels))
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.uniform_(-0.1, 0.1)
    return write_safetensors(path, {
        wan_raw_key(name): value.detach().half().contiguous()
        for name, value in source.state_dict().items()
    })


def write_wan_vae(path, *, channels: int = 48) -> str:
    """Write a file whose header claims a Wan codec of that latent width.

    The encoder output conv is the only entry in it, because that is all
    `detect_latent_width` reads; the codec itself is stubbed by whoever points here.
    """
    return write_safetensors(path, {
        "encoder.head.2.weight": torch.zeros(2 * channels, 1, 1, 1, 1),
    })


def stub_wan_components(channels: int = 48):
    """Stand-ins for the two components a Wan adapter loads and a policy test never touches.

    An adapter loads what it is pointed at as it is built, like every image adapter, so
    even a test that only asks it about requests or LoRA names has to answer the loads.
    The transformer is not stubbed: its header says which variant this is, which is what
    most of these tests are really asking.
    """
    from thenoise.vae.wan_video import TEMPORAL_COMPRESSION

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.z_dim = channels
            self.temporal_compression = TEMPORAL_COMPRESSION
            # The 2.1 family halves the spatial factor of its still-image codec.
            self.spatial_compression = 16 if channels == 48 else 8

        @property
        def dtype(self):
            return self.weight.dtype

    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class Tokenizer:
        def encode(self, prompt):
            return (torch.ones(1, 4, dtype=torch.long),
                    torch.ones(1, 4, dtype=torch.bool))

    return (lambda _path, _device: VAE(),
            lambda _path, _device: (Text(), Tokenizer()))


def wan_adapter(variant: str = "ti2v_5b"):
    """A Wan adapter built for real on a throwaway tiny DiT of that variant.

    Enough to ask it about requests, capabilities and schedules: construction reads the
    DiT header to pick the variant and the codec header to check it pairs, which is what
    no fake path can do. An adapter loads its components as it is built, so the codec
    and the text encoder are stood in for here — the transformer is the real one, since
    its header is what the adapter is being asked about.
    """
    import tempfile
    from unittest.mock import patch

    from thenoise.dit.wan_video.model import CHANNELS
    from thenoise.models.config import ModelConfig
    from thenoise.models.wan_video import Wan22Model

    channels = CHANNELS[variant]
    vae, text_encoder = stub_wan_components(channels)
    with tempfile.TemporaryDirectory() as work:
        dit = write_wan_dit(Path(work) / "dit.safetensors", channels=channels)
        config = ModelConfig(
            # A two-expert variant is two files; one tiny transformer serves as both.
            dit_paths=({"": dit} if variant != "t2v_14b" else {"": dit, "low": dit}),
            vae_path=write_wan_vae(Path(work) / "vae.safetensors", channels=channels),
            text_encoder_path="", device="cpu", offload_device="cpu")
        with patch("thenoise.models.wan_video.load_wan_video_vae", vae), \
             patch("thenoise.models.wan_video.load_wan_text_encoder", text_encoder):
            return Wan22Model(config=config)


def wan_prefs(variant: str = "ti2v_5b") -> dict:
    """The complete request defaults the given Wan variant resolves against."""
    from thenoise.models.wan_video import VARIANTS, Wan22Model

    return {**Wan22Model.DEFAULT_PREFS, **VARIANTS[variant].prefs}


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


def int8_lora_state_dict(module: str = "q", rank: int = 8, gain: float = 1.0) -> dict:
    """A LoRA state dict targeting ``module`` (sd-scripts style factor names).

    ``gain`` scales the down factor, and so the whole delta: the default is orders
    of magnitude wider than any quantization step (a bakeable LoRA), a small gain
    makes it finer than one (the degenerate LoRAs the runtime path exists for).
    """
    out_f, in_f = TINY_DIM
    return {
        f"{module}.lora_down.weight": torch.randn(rank, in_f, dtype=torch.bfloat16) * gain,
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


def int8_realistic_pair(shape=TINY_DIM, sigma: float = 0.02):
    """An int8 weight whose per-row scale actually matches its codes.

    :func:`int8_pair` draws the scale independently of the codes, so the
    dequantized weight is absurdly large. Harmless for layout tests, but a test
    that needs a layer output and a small LoRA add-on to be comparable in BF16
    needs weights of a plausible magnitude.
    """
    weight = torch.randn(*shape) * sigma
    scale = (weight.abs().amax(dim=1, keepdim=True) / 127).to(torch.float32)
    qweight = (weight / scale).round().clamp_(-127, 127).to(torch.int8)
    return qweight, scale


def wrapped_int8_tensor(
    convrot: bool = True, groupsize: int = 256, realistic: bool = False
):
    """A ready-made ``TensorWiseINT8Layout`` QuantizedTensor matching :data:`TINY_DIM`."""
    qweight, scale = int8_realistic_pair() if realistic else int8_pair()
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
    # The two S3-DiTs are the same transformer, so the learned pad tokens Z-Image
    # ships are the whole separator: without them this key-set is a Ming-Image file.
    "zimage": (
        ZImageModel,
        [
            "x_embedder.weight",
            "x_pad_token",
            "cap_pad_token",
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
            "model.diffusion_model.x_pad_token",
            "model.diffusion_model.cap_embedder.1.weight",
            "model.diffusion_model.context_refiner.0.attention_norm1.weight",
            "model.diffusion_model.layers.0.attention_norm1.weight",
        ],
    ),
    # Ming-Image, bf16 release: legacy Lumina naming (``all_x_embedder["2-1"]``,
    # separate ``to_q/to_k/to_v``) and NO pad tokens.
    "ming_image": (
        MingImageModel,
        [
            "all_x_embedder.2-1.weight",
            "all_final_layer.2-1.linear.weight",
            "cap_embedder.1.weight",
            "context_refiner.0.attention.to_q.weight",
            "layers.0.attention.to_q.weight",
            "t_embedder.mlp.0.weight",
        ],
    ),
    "ming_image_wrapped": (
        MingImageModel,
        [
            "model.diffusion_model.all_x_embedder.2-1.weight",
            "model.diffusion_model.cap_embedder.1.weight",
            "model.diffusion_model.context_refiner.0.attention.to_q.weight",
            "model.diffusion_model.layers.0.attention.to_q.weight",
        ],
    ),
    # Ming-Image, int8-convrot release: already fused, and named EXACTLY like
    # Z-Image apart from the missing pad tokens — the detection trap of this family.
    "ming_image_int8": (
        MingImageModel,
        [
            "x_embedder.weight",
            "cap_embedder.1.weight",
            "context_refiner.0.attention.qkv.weight",
            "layers.0.attention.qkv.weight",
            "layers.0.attention.q_norm.weight",
            "final_layer.linear.weight",
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
    # Flux.3: the per-mode blocks and modulations are its own, and the flow-map
    # ``ref_time_in`` / split q/k/v variants change nothing about that fingerprint.
    "flux3": (
        Flux3Model,
        [
            "content_mode_blocks.image.0.attn_out.weight",
            "content_mode_blocks.image.0.qkv.weight",
            "content_mode_blocks.image_cond.0.qkv.weight",
            "txt_mode_blocks.0.mlp_in.weight",
            "single_blocks.0.qkv.weight",
            "early_stream_modulations.image_cond.lin.weight",
            "single_stream_modulations.txt.lin.weight",
            "emb_in.image.weight",
            "emb_in.image_cond.weight",
            "time_in.in_layer.weight",
            "ref_time_in.in_layer.weight",
            "txt_in.weight",
            "final_layer.image.linear.weight",
        ],
    ),
    "flux3_wrapped": (
        Flux3Model,
        [
            "model.diffusion_model.content_mode_blocks.image.0.attn_out.weight",
            "model.diffusion_model.single_blocks.0.qkv.weight",
            "model.diffusion_model.single_stream_modulations.txt.lin.weight",
            "model.diffusion_model.emb_in.image.weight",
            "model.diffusion_model.final_layer.image.linear.weight",
        ],
    ),
    # MOSS-Image runs one joint stream, so its shared block modulation and its fused
    # up/gate FFN are the whole fingerprint.
    "moss_image": (
        MossImageModel,
        [
            "video_embedder.weight",
            "text_embedder.proj.weight",
            "text_embedder.blocks.1.qkv.weight",
            "time_embedding.0.weight",
            "shared_block_modulation.weight",
            "layers.0.proj.qkv.weight",
            "layers.0.proj.ffn_up_gate.weight",
            "layers.41.proj.ffn_down.weight",
            "final_norm_video.weight",
            "final_linear_video.weight",
            "video_head_modulation.weight",
        ],
    ),
    "moss_image_wrapped": (
        MossImageModel,
        [
            "model.diffusion_model.video_embedder.weight",
            "model.diffusion_model.shared_block_modulation.weight",
            "model.diffusion_model.layers.0.proj.ffn_up_gate.weight",
            "model.diffusion_model.layers.41.proj.qkv.weight",
            "model.diffusion_model.final_linear_video.weight",
        ],
    ),
    "qwen_image": (
        QwenImageModel,
        [
            "img_in.weight",
            "txt_in.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.add_q_proj.weight",
            "transformer_blocks.0.img_mlp.net.2.weight",
            "transformer_blocks.59.attn.to_q.weight",
            "norm_out.linear.weight",
            "proj_out.weight",
        ],
    ),
    "qwen_image_wrapped": (
        QwenImageModel,
        [
            "model.diffusion_model.img_in.weight",
            "model.diffusion_model.txt_in.weight",
            "model.diffusion_model.time_text_embed.timestep_embedder.linear_1.weight",
            "model.diffusion_model.transformer_blocks.0.attn.to_q.weight",
            "model.diffusion_model.transformer_blocks.0.attn.add_q_proj.weight",
            "model.diffusion_model.transformer_blocks.0.img_mlp.net.2.weight",
            "model.diffusion_model.transformer_blocks.59.attn.to_q.weight",
            "model.diffusion_model.norm_out.linear.weight",
            "model.diffusion_model.proj_out.weight",
        ],
    ),
    # Mage-Flow IS the Qwen-Image block, at 12 layers instead of 60: the depth in the
    # key-set is the only thing that separates the two (see dit/mage_flow/keys.py).
    "mage_flow": (
        MageFlowModel,
        [
            "img_in.weight",
            "txt_in.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "txt_norm.weight",
            "transformer_blocks.0.attn.add_q_proj.weight",
            "transformer_blocks.0.img_mlp.net.2.weight",
            "transformer_blocks.11.attn.to_q.weight",
            "norm_out.linear.weight",
            "proj_out.weight",
        ],
    ),
    "mage_flow_wrapped": (
        MageFlowModel,
        [
            "model.diffusion_model.img_in.weight",
            "model.diffusion_model.txt_in.weight",
            "model.diffusion_model.time_text_embed.timestep_embedder.linear_1.weight",
            "model.diffusion_model.txt_norm.weight",
            "model.diffusion_model.transformer_blocks.0.attn.add_q_proj.weight",
            "model.diffusion_model.transformer_blocks.0.img_mlp.net.2.weight",
            "model.diffusion_model.transformer_blocks.11.attn.to_q.weight",
            "model.diffusion_model.norm_out.linear.weight",
            "model.diffusion_model.proj_out.weight",
        ],
    ),
    "qwen_image21": (
        QwenImage21Model,
        [
            "img_in.weight",
            "proj_out.weight",
            "modulation.1.weight",
            "txt_in.text_norm.weight",
            "txt_in.in_layer.weight",
            "time_text_embed.timestep_embedder.linear_1.weight",
            "transformer_blocks.0.attn.norm_q.weight",
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.img_mlp.gate_up.weight",
        ],
    ),
    "qwen_image21_wrapped": (
        QwenImage21Model,
        [
            "model.diffusion_model.img_in.weight",
            "model.diffusion_model.proj_out.weight",
            "model.diffusion_model.modulation.1.weight",
            "model.diffusion_model.txt_in.text_norm.weight",
            "model.diffusion_model.time_text_embed.timestep_embedder.linear_1.weight",
            "model.diffusion_model.transformer_blocks.0.attn.norm_q.weight",
            "model.diffusion_model.transformer_blocks.0.img_mlp.gate_up.weight",
        ],
    ),
    # The two Qwen-Images share the img_in / txt_in / time_text_embed prefixes: the
    # detection matrix below pins down that neither claims the other's key-set.
    # The same matrix is what keeps Mage-Flow (12 blocks) out of Qwen-Image's 60.
    # Fibo 2.0 shares the fused self-attention and the bare head with the Lumina-Next
    # models; the Perceiver's queries are the whole fingerprint.
    "fibo2": (
        Fibo2Model,
        [
            "perceiver.latent_queries",
            "perceiver.layers.0.cross_attention.to_kv.weight",
            "noise_refiner.0.attention.qkv.weight",
            "layers.0.attention.qkv.weight",
            "layers.0.attention.q_norm.weight",
            "layers.0.adaLN_modulation.0.weight",
            "layers.6.cross_attn_gate",
            "layers.0.feed_forward.w13.weight",
            "t_embedder.mlp.0.weight",
            "final_layer.linear.weight",
        ],
    ),
    "fibo2_wrapped": (
        Fibo2Model,
        [
            "model.diffusion_model.perceiver.latent_queries",
            "model.diffusion_model.layers.0.attention.qkv.weight",
            "model.diffusion_model.layers.0.adaLN_modulation.0.weight",
            "model.diffusion_model.layers.0.feed_forward.w13.weight",
            "model.diffusion_model.perceiver.layers.0.cross_attention.norm_q.weight",
            "model.diffusion_model.t_embedder.mlp.0.weight",
            "model.diffusion_model.final_layer.linear.weight",
        ],
    ),
    # The un-fused SwiGLU spelling is the same model: an export that stores w1/w3
    # apart must still resolve to it.
    "fibo2_split_ffn": (
        Fibo2Model,
        [
            "perceiver.latent_queries",
            "layers.0.attention.qkv.weight",
            "layers.0.adaLN_modulation.0.weight",
            "layers.0.feed_forward.w1.weight",
            "layers.0.feed_forward.w3.weight",
            "t_embedder.mlp.0.weight",
            "final_layer.linear.weight",
        ],
    ),
    # The packed H3 DiT without an audio stream. The video sibling shares every key but
    # ``audio_patch_proj``, and is claimed by nobody here rather than by this adapter.
    "h3_image": (
        H3ImageModel,
        [
            "video_patch_proj.weight",
            "condition_proj.weight",
            "blocks.0.attn.qkv_proj.weight",
            "final_layer.video_out.weight",
            "adaln_t_table",
        ],
    ),
    "h3_image_wrapped": (
        H3ImageModel,
        [
            "model.diffusion_model.video_patch_proj.weight",
            "model.diffusion_model.condition_proj.weight",
            "model.diffusion_model.blocks.0.attn.qkv_proj.weight",
        ],
    ),
    "h3_video": (
        None,
        [
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "condition_proj.weight",
            "final_layer.audio_out.weight",
        ],
    ),
    "wan22": (
        Wan22Model,
        [
            "patch_embedding.weight",
            "blocks.0.self_attn.q.weight",
            "text_embedding.0.weight",
        ],
    ),
    "unknown": (None, ["some.random.key", "blocks.0.attn.gate.weight"]),
}

KEYSET_IDS = sorted(MODEL_KEYSETS)
CATALOG_IDS = [cls.name for cls in MODEL_CATALOG]
# The contracts only the image medium answers to: a latent upscaler, and a sampler in
# the class defaults (a video variant carries its own recipe, layered at resolve time).
IMAGE_CATALOG = [cls for cls in MODEL_CATALOG if not issubclass(cls, VideoModel)]
IMAGE_CATALOG_IDS = [cls.name for cls in IMAGE_CATALOG]


# ------------------------------------------------------------------------ stub models


class StubLatentUpscaler(LatentUpscaler):
    """A mock latent upscaler: the interface with no strategy, weights or device.

    Pipeline tests must not care *how* the latent gets bigger, so this stands in
    for any ``LatentUpscaler``: it grows the latent by ``scale`` and returns a
    constant value, which lets a test reproduce the refine's input exactly.
    ``calls`` counts invocations (i.e. that the pipeline asked at all, and once).
    """

    def __init__(self, scale: int, value: float = 0.5):
        self.scale = scale
        self.value = value
        self.calls = 0

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        h, w = latents.shape[-2:]
        return torch.full(
            (latents.shape[0], latents.shape[1], self.scale * h, self.scale * w),
            self.value,
            dtype=latents.dtype,
            device=latents.device,
        )


class StubVideo(VideoModel):
    """A weight-free video adapter: enough of a codec and a DiT to drive a controller.

    Its latent is the canonical 5D form, ``[1, 4, T, H/8, W/8]``, at a temporal
    compression of 4 and one frame decoded per latent frame. Kernels answer zero, so a
    test sees orchestration — staging, progress, cancellation, teardown — without maths.
    """

    name = "stub-video"
    DEFAULT_PREFS = {**VideoModel.DEFAULT_PREFS, "width": 64, "height": 64,
                     "frames": 5, "fps": 8, "steps": 2, "guidance_scale": 1.0,
                     "sampler": "euler"}
    Z_DIM = 4
    SCALE = 8

    def __init__(self, *, config):
        super().__init__(config=config)
        self.dit = torch.nn.Linear(1, 1)
        self.calls = Counter()

    @staticmethod
    def detect(_checkpoint):
        return False

    def _create_upscaler(self):
        raise NotImplementedError(f"{self.name} has no latent upscaler")

    def _clear_weights(self):
        self.dit = None

    def encode_prompt(self, args):
        self.calls["encode_prompt"] += 1
        return Conditioning(cond=torch.zeros(1, 2, 4),
                            null=torch.zeros(1, 2, 4) if args.guidance_scale > 1 else None)

    def init_latents(self, params):
        self.calls["init_latents"] += 1
        return torch.zeros(1, self.Z_DIM, (params.frames - 1) // 4 + 1,
                           params.height // self.SCALE, params.width // self.SCALE)

    def schedule(self, params):
        grid = torch.linspace(1.0, 0.0, params.steps + 1)
        return [Step(t=grid[i], delta=grid[i] - grid[i + 1]) for i in range(params.steps)]

    def denoise_step(self, latents, t, cond, guidance_scale, i):
        self.calls["denoise_step"] += 1
        return torch.zeros_like(latents)

    def encode_reference(self, pixels):
        self.calls["encode_reference"] += 1
        return torch.zeros(1, self.Z_DIM, 1, pixels.shape[-2] // self.SCALE,
                           pixels.shape[-1] // self.SCALE)

    def decode_frames(self, latents):
        for _ in range(latents.shape[2]):
            yield torch.zeros(3, latents.shape[3] * self.SCALE,
                              latents.shape[4] * self.SCALE)


class StubModel(DiffusionModel):
    """A fully functional weight-free adapter for pipeline/sampler tests.

    Every kernel is real enough to drive the pipeline end to end on CPU fp32 and
    every one of them is counted in ``self.calls`` so cache behaviour, call
    counts and the refine schedule are directly observable.
    """

    name = "stub"
    DEFAULT_PREFS = {
        **DiffusionModel.DEFAULT_PREFS,
        "width": 64,
        "height": 64,
        "steps": 2,
        "guidance_scale": 1.0,
        "sampler": "euler",
    }
    # Stub-local latent geometry: a real adapter reads these off its VAE
    # (``vae.z_dim`` / ``vae.spatial_compression``), which this stub has none of.
    LATENT_CHANNELS = 4
    _VAE_SCALE = 8
    UPSCALE_SCALE = 2
    REFINE_STEPS = 1
    REFINE_DENOISE = 0.1

    def __init__(
        self,
        *,
        config: Optional[ModelConfig] = None,
        capabilities: Optional[dict] = None,
        lora_dir: Optional[str] = None,
    ):
        super().__init__(
            config=config
            or ModelConfig(
                dit_paths={"": "dit"}, vae_path="vae", text_encoder_path="te",
                offload_device="cpu", device="cpu", dtype=torch.float32, lora_dir=lora_dir,
            )
        )
        if capabilities is not None:
            self.CAPABILITIES = {**self.CAPABILITIES, **capabilities}
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

    def _create_upscaler(self) -> LatentUpscaler:
        return StubLatentUpscaler(self.UPSCALE_SCALE)


class EditingStubModel(StubModel):
    """A stub that declares the ``edit`` capability (reference-latent path)."""

    CAPABILITIES = {**StubModel.CAPABILITIES, "edit": True}


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

        def _create_upscaler(self):
            return StubLatentUpscaler(self.UPSCALE_SCALE)

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
            # A catalog entry answers for its own DiT files and releases its own state;
            # these stand-ins have no opinions, and the adapter tests cover the real hooks.
            "validate_paths": classmethod(lambda cls, dit_paths: None),
            "unload": lambda self: None,
            # A model that declares no capability; tests override it via ``**attrs``.
            "CAPABILITIES": {},
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
    """A fresh :class:`StubModel` with the ``edit`` capability enabled."""
    return EditingStubModel()


@pytest.fixture(scope="module")
def flux2_vae():
    """One random-init Flux.2 VAE per module (building it costs ~0.2s)."""
    return AutoencoderKLFlux2().eval().requires_grad_(False)


# For a test that asks a real card, not the figures ``stub_room`` supplies: a CPU-only
# torch raises here.
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA device")


def stub_room(monkeypatch, free, *, total=None, reserved: int = 0, live: int = 0) -> None:
    """The room a CUDA device reports: driver ``free``, allocator ``reserved``/``live``.
    ``free`` and ``total`` both None for a device that quotes nothing.
    """
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr("thenoise.memory.get_free_memory", lambda _dev: free)
    monkeypatch.setattr("thenoise.memory.get_device_memory", lambda _dev: total)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _dev: reserved)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _dev: live)


class TrackedModule(torch.nn.Module):
    """Weights parked on ``meta`` at a stated size, plus a log of every device move.

    ``meta`` makes an offload legal where there is no device, and a meta tensor still
    reports the byte size a placement decision reads. ``device`` is what the manager is
    *told* the module sits on, spelled as the manager spells its own.
    """

    def __init__(self, nbytes: int, device: str = "meta"):
        super().__init__()
        self.p = torch.nn.Parameter(torch.empty(max(nbytes // 4, 1), device="meta"))
        self._at = torch.device(device)
        self.moves: list[str] = []

    @property
    def device(self):
        return self._at

    def to(self, device=None, *args, **kwargs):  # noqa: D102
        self._at = torch.device(device)
        self.moves.append(self._at.type)
        return self
