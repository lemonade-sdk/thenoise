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
    with MemoryEfficientSafeOpen(dit_path) as f:
        shapes = {unwrap_key(k): tuple(f.header[k]["shape"]) for k in f.keys()}
    return shape_params(shapes)


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
    "Hunyuan3DShapeParams",
    "detect_params",
    "is_hunyuan3d_shape",
    "shape_key_map",
    "shape_params",
    "take_prefix",
]
