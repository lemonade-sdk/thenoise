"""ComfyUI Wan 2.2 checkpoint key layouts used by Comfy-Org."""
from __future__ import annotations

import re

from thenoise.utils.checkpoint_weights import validated_key_map


def wan_dit_key(raw: str) -> str:
    """Map ComfyUI WanModel weights to TheNoise's TI2V-5B DiT structure."""
    name = raw.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.")
    roots = {
        "text_embedding.0.": "condition_embedder.text_embedder.linear_1.",
        "text_embedding.2.": "condition_embedder.text_embedder.linear_2.",
        "time_embedding.0.": "condition_embedder.time_embedder.linear_1.",
        "time_embedding.2.": "condition_embedder.time_embedder.linear_2.",
        "time_projection.1.": "condition_embedder.time_proj.",
        "head.norm.": "norm_out.",
        "head.head.": "proj_out.",
        "head.modulation": "scale_shift_table",
    }
    for prefix, replacement in roots.items():
        if name.startswith(prefix):
            return replacement + name[len(prefix):]

    match = re.match(r"^(blocks\.\d+)\.(.*)$", name)
    if not match:
        return name
    block, suffix = match.groups()
    if suffix == "modulation":
        return f"{block}.scale_shift_table"
    # ComfyUI norm2 is FFN, norm3 is cross-attention; Diffusers used the reverse.
    if suffix.startswith("norm2."):
        suffix = "norm3." + suffix[len("norm2."):]
    elif suffix.startswith("norm3."):
        suffix = "norm2." + suffix[len("norm3."):]
    for src, dst in (
        ("self_attn.q.", "attn1.to_q."),
        ("self_attn.k.", "attn1.to_k."),
        ("self_attn.v.", "attn1.to_v."),
        ("self_attn.o.", "attn1.to_out.0."),
        ("self_attn.norm_q.", "attn1.norm_q."),
        ("self_attn.norm_k.", "attn1.norm_k."),
        ("cross_attn.q.", "attn2.to_q."),
        ("cross_attn.k.", "attn2.to_k."),
        ("cross_attn.v.", "attn2.to_v."),
        ("cross_attn.o.", "attn2.to_out.0."),
        ("cross_attn.norm_q.", "attn2.norm_q."),
        ("cross_attn.norm_k.", "attn2.norm_k."),
        ("ffn.0.", "ffn.net.0.proj."),
        ("ffn.2.", "ffn.net.2."),
    ):
        if suffix.startswith(src):
            return f"{block}.{dst}{suffix[len(src):]}"
    return f"{block}.{suffix}"


def wan_vae_key(raw: str, *, num_res_blocks: int = 2) -> str:
    """Map ComfyUI WanVAE (vae2_2.py) to TheNoise's causal 3D VAE."""
    name = raw.removeprefix("vae.")
    if name.startswith("conv1."):
        return "quant_conv." + name[len("conv1."):]
    if name.startswith("conv2."):
        return "post_quant_conv." + name[len("conv2."):]
    for part in ("encoder", "decoder"):
        replacements = {
            f"{part}.conv1.": f"{part}.conv_in.",
            f"{part}.head.0.": f"{part}.norm_out.",
            f"{part}.head.2.": f"{part}.conv_out.",
            f"{part}.middle.0.": f"{part}.mid_block.resnets.0.",
            f"{part}.middle.1.": f"{part}.mid_block.attentions.0.",
            f"{part}.middle.2.": f"{part}.mid_block.resnets.1.",
        }
        for prefix, replacement in replacements.items():
            if name.startswith(prefix):
                name = replacement + name[len(prefix):]
                break

    match = re.match(r"^(encoder|decoder)\.(downsamples|upsamples)\.(\d+)\.\2\.(\d+)\.(.*)$", name)
    if match:
        part, _, block, index, rest = match.groups()
        idx = int(index)
        if part == "encoder":
            if idx == num_res_blocks:
                name = f"{part}.down_blocks.{block}.downsampler.{rest}"
            elif idx < num_res_blocks:
                name = f"{part}.down_blocks.{block}.resnets.{idx}.{rest}"
        else:
            if idx == num_res_blocks + 1:
                name = f"{part}.up_blocks.{block}.upsampler.{rest}"
            elif idx <= num_res_blocks:
                name = f"{part}.up_blocks.{block}.resnets.{idx}.{rest}"

    name = name.replace(".residual.0.", ".norm1.")
    name = name.replace(".residual.2.", ".conv1.")
    name = name.replace(".residual.3.", ".norm2.")
    name = name.replace(".residual.6.", ".conv2.")
    name = name.replace(".shortcut.", ".conv_shortcut.")
    return name
