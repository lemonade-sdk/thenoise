from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Tuple

from thenoise.utils.qk_norm import qk_norm_key_map
from thenoise.utils.safetensors import MemoryEfficientSafeOpen, unwrap_key

#: One Hunyuan3D shape file carries the DiT, the image conditioner and the shape codec.
DIT_PREFIX = "model."
CONDITIONER_PREFIX = "conditioner.main_image_encoder.model."
CODEC_PREFIX = "vae."

#: Experts the gate picks; no shape says, and no released checkpoint differs.
MOE_TOP_K = 2

#: The 2.1 fingerprint: a two-layer timestep MLP, and QK-norm inside the first block.
_FINGERPRINT = (
    DIT_PREFIX + "t_embedder.mlp.2.weight",
    DIT_PREFIX + "blocks.0.attn1.k_norm.weight",
)


@dataclass(frozen=True)
class Hunyuan3DShapeParams:
    """The DiT's geometry, as read out of the checkpoint header."""

    hidden_size: int
    depth: int
    num_heads: int
    head_dim: int
    in_channels: int
    out_channels: int
    context_dim: int
    mlp_ratio: int
    moe_layers: int
    num_experts: int
    top_k: int = MOE_TOP_K
    qkv_bias: bool = False
    norm_eps: float = 1e-6
    qk_norm_eps: float = 1e-6

    @property
    def skip_layers(self) -> int:
        """Blocks that fold in an earlier block's output; the deep half, minus one."""
        return self.depth - self.depth // 2 - 1

    def uses_moe(self, layer: int) -> bool:
        return layer >= self.depth - self.moe_layers

    def uses_skip(self, layer: int) -> bool:
        return layer > self.depth // 2


#: The 2.1 latent scale. No tensor in the file carries it, so it is a per-release fact.
SHAPE_SCALE_FACTOR = 1.0039506158752403


@dataclass(frozen=True)
class Hunyuan3DCodecParams:
    """The shape codec's decode half, as read out of the checkpoint header."""

    width: int
    embed_dim: int
    num_heads: int
    head_dim: int
    layers: int
    mlp_ratio: int
    geo_mlp_ratio: int
    num_freqs: int
    qkv_bias: bool = True
    qk_norm: bool = True
    geo_ln_post: bool = True
    norm_eps: float = 1e-6
    scale_factor: float = SHAPE_SCALE_FACTOR


def is_hunyuan3d_shape(keys: Iterable[str]) -> bool:
    """True for a Hunyuan3D shape checkpoint; the whole file, not just its DiT."""
    names = set(keys)
    return all(key in names for key in _FINGERPRINT)


def shape_params(shapes: Mapping[str, Tuple[int, ...]]) -> Hunyuan3DShapeParams:
    """The DiT geometry out of a name -> shape mapping (wrapper prefixes already off)."""
    p = DIT_PREFIX
    hidden = shapes[p + "x_embedder.weight"][0]
    head_dim = shapes[p + "blocks.0.attn1.q_norm.weight"][0]
    if hidden % head_dim:
        raise ValueError(f"{hidden} is not a whole number of {head_dim}-wide heads")
    depth = _block_count(shapes, p)
    experts = _expert_count(shapes, p)
    moe_layers = sum(1 for i in range(depth) if p + f"blocks.{i}.moe.gate.weight" in shapes)
    if moe_layers and not experts:
        raise ValueError("MoE blocks without a gate wide enough to count their experts")
    return Hunyuan3DShapeParams(
        hidden_size=hidden,
        depth=depth,
        num_heads=hidden // head_dim,
        head_dim=head_dim,
        in_channels=shapes[p + "x_embedder.weight"][1],
        out_channels=shapes[p + "final_layer.linear.weight"][0],
        context_dim=shapes[p + "blocks.0.attn2.to_k.weight"][1],
        mlp_ratio=_ffn_ratio(shapes, p, hidden, depth),
        moe_layers=moe_layers,
        num_experts=experts,
        qkv_bias=p + "blocks.0.attn1.to_q.bias" in shapes,
    )


def detect_params(dit_path: str) -> Hunyuan3DShapeParams:
    """Read the DiT geometry from a checkpoint's header, without loading tensors."""
    return shape_params(checkpoint_shapes(dit_path))


def codec_params(shapes: Mapping[str, Tuple[int, ...]]) -> Hunyuan3DCodecParams:
    """The decode-half geometry out of a name -> shape mapping."""
    p = CODEC_PREFIX
    width = shapes[p + "post_kl.weight"][0]
    resblock = p + "transformer.resblocks.0."
    head_dim = shapes[resblock + "attn.attention.q_norm.weight"][0]
    if width % head_dim:
        raise ValueError(f"{width} is not a whole number of {head_dim}-wide heads")
    query_in = shapes[p + "geo_decoder.query_proj.weight"][1]
    if (query_in - 3) % 6:
        raise ValueError(f"{query_in} query features are not xyz at any frequency count")
    return Hunyuan3DCodecParams(
        width=width,
        embed_dim=shapes[p + "post_kl.weight"][1],
        num_heads=width // head_dim,
        head_dim=head_dim,
        layers=_resblock_count(shapes, p + "transformer.resblocks."),
        mlp_ratio=shapes[resblock + "mlp.c_fc.weight"][0] // width,
        geo_mlp_ratio=shapes[p + "geo_decoder.cross_attn_decoder.mlp.c_fc.weight"][0] // width,
        num_freqs=(query_in - 3) // 6,
        qkv_bias=resblock + "attn.c_qkv.bias" in shapes,
        qk_norm=resblock + "attn.attention.q_norm.weight" in shapes,
        geo_ln_post=p + "geo_decoder.ln_post.weight" in shapes,
    )


def detect_codec_params(dit_path: str) -> Hunyuan3DCodecParams:
    """Read the codec geometry from a checkpoint's header, without loading tensors."""
    return codec_params(checkpoint_shapes(dit_path))


def codec_key_map(key: str) -> str:
    """``c_fc``/``c_proj`` onto the shared feed-forward, ``q_norm``/``k_norm`` onto QKNorm."""
    key = key.replace(".mlp.c_fc.", ".mlp.net.0.proj.")
    key = key.replace(".mlp.c_proj.", ".mlp.net.2.")
    key = key.replace(".attention.q_norm.", ".attention.query_norm.")
    return key.replace(".attention.k_norm.", ".attention.key_norm.")


def take_prefix(prefix: str) -> Callable[[dict], dict]:
    """A ``load_dit`` ``state_map`` keeping one submodule of a shared file."""
    def fold(state_dict: dict) -> dict:
        return {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
    return fold


def shape_key_map(key: str) -> str:
    """Checkpoint ``fc1``/``fc2`` and ``q_norm``/``k_norm`` onto the shared slots."""
    if ".moe." not in key:  # the experts are saved in net slots already
        key = key.replace(".mlp.fc1.", ".mlp.net.0.proj.")
        key = key.replace(".mlp.fc2.", ".mlp.net.2.")
    return qk_norm_key_map(key)


def checkpoint_shapes(path: str) -> Mapping[str, Tuple[int, ...]]:
    """Every tensor name in a file, wrapper-normalized, with its shape."""
    with MemoryEfficientSafeOpen(path) as f:
        return {unwrap_key(k): tuple(f.header[k]["shape"]) for k in f.keys()}


def _block_count(shapes: Mapping[str, Tuple[int, ...]], prefix: str) -> int:
    indices = set()
    for key in shapes:
        if not key.startswith(prefix + "blocks."):
            continue
        index = key[len(prefix) + len("blocks."):].split(".", 1)[0]
        if index.isdigit():
            indices.add(int(index))
    if not indices:
        raise ValueError("no transformer blocks under the DiT prefix")
    if indices != set(range(len(indices))):
        raise ValueError(f"block indices are not contiguous: 0..{max(indices)}")
    return len(indices)


def _expert_count(shapes: Mapping[str, Tuple[int, ...]], prefix: str) -> int:
    for key, shape in shapes.items():
        if key.startswith(prefix + "blocks.") and key.endswith(".moe.gate.weight"):
            return shape[0]
    return 0


def _resblock_count(shapes: Mapping[str, Tuple[int, ...]], prefix: str) -> int:
    indices = set()
    for key in shapes:
        if not key.startswith(prefix):
            continue
        index = key[len(prefix):].split(".", 1)[0]
        if index.isdigit():
            indices.add(int(index))
    if indices != set(range(len(indices))):
        raise ValueError(f"resblocks are not contiguous under {prefix}")
    return len(indices)


def _ffn_ratio(shapes: Mapping[str, Tuple[int, ...]], prefix: str, hidden: int,
               depth: int) -> int:
    """The FFN multiplier, from whichever kind of FFN the shallowest block has."""
    for i in range(depth):
        fc1 = prefix + f"blocks.{i}.mlp.fc1.weight"
        if fc1 in shapes:
            return shapes[fc1][0] // hidden
        expert = prefix + f"blocks.{i}.moe.experts.0.net.0.proj.weight"
        if expert in shapes:
            return shapes[expert][0] // hidden
    raise ValueError("no feed-forward under the DiT prefix")


__all__ = [
    "CODEC_PREFIX",
    "CONDITIONER_PREFIX",
    "DIT_PREFIX",
    "SHAPE_SCALE_FACTOR",
    "Hunyuan3DCodecParams",
    "Hunyuan3DShapeParams",
    "checkpoint_shapes",
    "codec_key_map",
    "codec_params",
    "detect_codec_params",
    "detect_params",
    "is_hunyuan3d_shape",
    "shape_key_map",
    "shape_params",
    "take_prefix",
]
