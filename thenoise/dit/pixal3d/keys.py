from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from thenoise.utils.loader import DIT_PREFIX
from thenoise.utils.safetensors import Header, Signature

#: The cascade in the order it runs, as each stage is filed inside the shared file.
STAGE_PREFIXES = {
    "structure": "structure_model.",
    "shape512": "img2shape_512.",
    "shape": "img2shape.",
    "texture": "shape2txt.",
}

#: Two stages sharing one file is a key set no other model answers to.
PIXAL_SIGNATURE = Signature(
    "Pixal3D",
    keys=(DIT_PREFIX + "structure_model.blocks.0.self_attn.to_qkv.weight",
          DIT_PREFIX + "shape2txt.input_layer.weight"),
)

#: RoPE over the integer voxel grid: three axes share the head, under one theta.
ROPE_AXES = 3
ROPE_THETA = 10000.0

#: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
ADALN_CHUNKS = 6


@dataclass(frozen=True)
class Pixal3DStageParams:
    """One stage's geometry, every field read out of the checkpoint header."""

    hidden_size: int
    depth: int
    heads: int
    head_dim: int
    mlp_dim: int
    in_channels: int
    out_channels: int
    cond_channels: int
    proj_channels: int
    timestep_features: int
    shared_modulation: bool


def stage_prefix(stage: str) -> str:
    """The key prefix one stage lives under in a shared file."""
    try:
        return DIT_PREFIX + STAGE_PREFIXES[stage]
    except KeyError as error:
        raise ValueError(f"unknown Pixal3D stage {stage!r}") from error


def stages_in(checkpoint: Header | Mapping[str, Tuple[int, ...]]) -> list[str]:
    """The stages a file carries, in cascade order."""
    shapes = _shapes(checkpoint)
    return [stage for stage in STAGE_PREFIXES
            if stage_prefix(stage) + "input_layer.weight" in shapes]


def stage_params(
    checkpoint: Header | Mapping[str, Tuple[int, ...]], stage: str) -> Pixal3DStageParams:
    """One stage's geometry out of a name -> shape mapping (wrapper prefixes already off)."""
    shapes = _shapes(checkpoint)
    p = stage_prefix(stage)
    hidden = _shape(shapes, p + "input_layer.weight")[0]
    head_dim = _shape(shapes, p + "blocks.0.self_attn.q_rms_norm.gamma")[1]
    heads = _shape(shapes, p + "blocks.0.self_attn.q_rms_norm.gamma")[0]
    if hidden != heads * head_dim:
        raise ValueError(f"{hidden} is not {heads} heads of {head_dim}")
    mlp = _shape(shapes, p + "blocks.0.mlp.mlp.0.weight")
    cond = _shape(shapes, p + "blocks.0.cross_attn.cross_attn_block.to_kv.weight")
    proj = _shape(shapes, p + "blocks.0.cross_attn.proj_linear.weight")
    timestep = _shape(shapes, p + "t_embedder.mlp.0.weight")
    return Pixal3DStageParams(
        hidden_size=hidden,
        depth=_block_count(shapes, p),
        heads=heads,
        head_dim=head_dim,
        mlp_dim=mlp[0],
        in_channels=_shape(shapes, p + "input_layer.weight")[1],
        out_channels=_shape(shapes, p + "out_layer.weight")[0],
        cond_channels=cond[1],
        proj_channels=proj[1],
        timestep_features=timestep[1],
        shared_modulation=p + "adaLN_modulation.1.weight" in shapes,
    )


def pixal3d_key_map(key: str) -> str:
    """The per-head QK gains onto the shared QK-norm slots; every other name already fits."""
    for legacy, shared in (("q_rms_norm", "query_norm"), ("k_rms_norm", "key_norm")):
        key = key.replace(f".{legacy}.gamma", f".qk_norm.{shared}.weight")
    return key


def _shapes(checkpoint: Header | Mapping[str, Tuple[int, ...]]) -> Mapping[str, Tuple[int, ...]]:
    return checkpoint.shapes if isinstance(checkpoint, Header) else checkpoint


def _shape(shapes: Mapping[str, Tuple[int, ...]], key: str) -> Tuple[int, ...]:
    try:
        return shapes[key]
    except KeyError as error:
        raise ValueError(f"no {key} in the checkpoint") from error


def _block_count(shapes: Mapping[str, Tuple[int, ...]], prefix: str) -> int:
    head = prefix + "blocks."
    indices = {key[len(head):].split(".", 1)[0] for key in shapes if key.startswith(head)}
    numbers = {int(i) for i in indices if i.isdigit()}
    if numbers != set(range(len(numbers))):
        raise ValueError(f"block indices are not contiguous: 0..{max(numbers or {0})}")
    return len(numbers)


__all__ = [
    "ADALN_CHUNKS",
    "DIT_PREFIX",
    "PIXAL_SIGNATURE",
    "ROPE_AXES",
    "ROPE_THETA",
    "STAGE_PREFIXES",
    "Pixal3DStageParams",
    "pixal3d_key_map",
    "stage_params",
    "stage_prefix",
    "stages_in",
]
