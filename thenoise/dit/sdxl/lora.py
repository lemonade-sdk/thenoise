# SDXL Hyper-SD step-reduction LoRA support.
#
# Hyper-SD (https://huggingface.co/ByteDance/Hyper-SD) ships SDXL LoRAs that
# reduce the required denoising steps to 2/4/8 (or 8/12 CFG-preserved) while
# keeping quality. The LoRAs are trained on the *diffusers* SDXL UNet layout
# (keys like ``lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight``), but
# our UNet uses the LDM layout (``input_blocks.*``, ``middle_block.*``,
# ``output_blocks.*``). This module converts a diffusers-keyed UNet LoRA to the
# LDM key names so ``apply_lora_to_model`` can match it against our UNet.

from typing import Dict, Iterable, Tuple

import torch

#: diffusers submodule -> LDM submodule for a resnet (weight/bias-suffixed).
_UNET_MAP_RESNET = {
    "in_layers.2.weight": "conv1.weight",
    "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight",
    "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight",
    "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight",
    "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight",
    "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight",
    "out_layers.0.bias": "norm2.bias",
}

#: diffusers attention submodules that map to the LDM spatial transformer.
_UNET_MAP_ATTENTIONS = (
    "norm.weight", "norm.bias", "proj_in.weight", "proj_in.bias",
    "proj_out.weight", "proj_out.bias",
)

#: diffusers transformer-block submodules.
_TRANSFORMER_BLOCKS = (
    "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
    "norm3.weight", "norm3.bias",
    "attn1.to_q.weight", "attn1.to_k.weight", "attn1.to_v.weight",
    "attn1.to_out.0.weight", "attn1.to_out.0.bias",
    "attn2.to_q.weight", "attn2.to_k.weight", "attn2.to_v.weight",
    "attn2.to_out.0.weight", "attn2.to_out.0.bias",
    "ff.net.0.proj.weight", "ff.net.0.proj.bias",
    "ff.net.2.weight", "ff.net.2.bias",
)

#: top-level diffusers <-> LDM keys (LDM key, diffusers key) for SDXL.
_UNET_MAP_BASIC = (
    ("input_blocks.0.0.weight", "conv_in.weight"),
    ("input_blocks.0.0.bias", "conv_in.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"),
    ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"),
    ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
    ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
    ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("out.0.weight", "conv_norm_out.weight"),
    ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"),
    ("out.2.bias", "conv_out.bias"),
)


def _unet_to_diffusers_sdxl() -> Dict[str, str]:
    """Return ``diffusers_key -> ldm_key`` for the SDXL UNet.

    Mirrors ComfyUI's ``unet_to_diffusers`` for the SDXL config
    (num_res_blocks=[2,2,2], channel_mult=[1,2,4], transformer_depth
    [0,0,2,2,10,10], transformer_depth_middle=10,
    transformer_depth_output=[0,0,0,2,2,2,10,10,10]).
    """
    num_res_blocks = [2, 2, 2]
    channel_mult = [1, 2, 4]
    transformer_depth = [0, 0, 2, 2, 10, 10][:]
    transformer_depth_output = [0, 0, 0, 2, 2, 2, 10, 10, 10][:]
    num_blocks = len(channel_mult)
    transformers_mid = 10

    m = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in _UNET_MAP_RESNET:
                m[f"down_blocks.{x}.resnets.{i}.{_UNET_MAP_RESNET[b]}"] = f"input_blocks.{n}.0.{b}"
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in _UNET_MAP_ATTENTIONS:
                    m[f"down_blocks.{x}.attentions.{i}.{b}"] = f"input_blocks.{n}.1.{b}"
                for t in range(num_transformers):
                    for b in _TRANSFORMER_BLOCKS:
                        m[f"down_blocks.{x}.attentions.{i}.transformer_blocks.{t}.{b}"] = (
                            f"input_blocks.{n}.1.transformer_blocks.{t}.{b}"
                        )
            n += 1
        for k in ("weight", "bias"):
            m[f"down_blocks.{x}.downsamplers.0.conv.{k}"] = f"input_blocks.{n}.0.op.{k}"

    i = 0
    for b in _UNET_MAP_ATTENTIONS:
        m[f"mid_block.attentions.{i}.{b}"] = f"middle_block.1.{b}"
    for t in range(transformers_mid):
        for b in _TRANSFORMER_BLOCKS:
            m[f"mid_block.attentions.{i}.transformer_blocks.{t}.{b}"] = (
                f"middle_block.1.transformer_blocks.{t}.{b}"
            )
    for i, n in enumerate([0, 2]):
        for b in _UNET_MAP_RESNET:
            m[f"mid_block.resnets.{i}.{_UNET_MAP_RESNET[b]}"] = f"middle_block.{n}.{b}"

    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        l = num_res_blocks[x] + 1
        for i in range(l):
            c = 0
            for b in _UNET_MAP_RESNET:
                m[f"up_blocks.{x}.resnets.{i}.{_UNET_MAP_RESNET[b]}"] = f"output_blocks.{n}.0.{b}"
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in _UNET_MAP_ATTENTIONS:
                    m[f"up_blocks.{x}.attentions.{i}.{b}"] = f"output_blocks.{n}.1.{b}"
                for t in range(num_transformers):
                    for b in _TRANSFORMER_BLOCKS:
                        m[f"up_blocks.{x}.attentions.{i}.transformer_blocks.{t}.{b}"] = (
                            f"output_blocks.{n}.1.transformer_blocks.{t}.{b}"
                        )
            if i == l - 1:
                for k in ("weight", "bias"):
                    m[f"up_blocks.{x}.upsamplers.0.conv.{k}"] = f"output_blocks.{n}.{c}.conv.{k}"
            n += 1

    for ldm_key, diffusers_key in _UNET_MAP_BASIC:
        m[diffusers_key] = ldm_key

    return m


def _ldm_for_diffusers_key(
    mapping: Dict[str, str], diffusers_weight_key: str
) -> str:
    """Return the LDM key for a diffusers key, or '' if unmapped."""
    return mapping.get(diffusers_weight_key, "")


def convert_hyper_sd_lora(
    lora_sd: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Convert a Hyper-SD SDXL UNet LoRA to our LDM key naming (in place-free).

    ``lora_sd`` keys are sd-scripts style::
        lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight
        lora_unet_down_blocks_0_resnets_0_conv1.lora_up.weight
        lora_unet_down_blocks_0_resnets_0_conv1.alpha
    Returns a new dict with the diffusers module path remapped to the LDM path,
    e.g. ``lora_unet_input_blocks_1_0_in_layers_0.lora_down.weight``.
    """
    mapping = _unet_to_diffusers_sdxl()
    # map: underscore diffusers weight key -> ldm weight key
    weight_map = {
        dkey.replace(".", "_"): ldm
        for dkey, ldm in mapping.items()
    }
    out: Dict[str, torch.Tensor] = {}
    for key, value in lora_sd.items():
        if not key.startswith("lora_unet_"):
            out[key] = value
            continue
        rest = key[len("lora_unet_"):]
        # find suffix
        for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha"):
            if rest.endswith(suffix):
                path = rest[: -len(suffix)]
                ldm_w = weight_map.get(path + "_weight")
                if ldm_w is None:
                    out[key] = value  # keep unmapped (e.g. unexpected)
                    break
                module = ldm_w[: -len(".weight")]
                out["lora_unet_" + module.replace(".", "_") + suffix] = value
                break
        else:
            out[key] = value
    return out


def lora_uses_diffusers_unet_keys(lora_sd: Dict[str, torch.Tensor]) -> bool:
    """True if the LoRA targets the diffusers SDXL UNet (down_blocks_/up_blocks_)."""
    return any(
        k.startswith("lora_unet_down_blocks_") or k.startswith("lora_unet_up_blocks_")
        for k in lora_sd
    )
