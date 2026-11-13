"""Comfy-Org/Wan-2.2 key-layout checks; no model download needed."""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from thenoise.utils.wan_weights import wan_dit_key, wan_vae_key, validated_key_map
from thenoise.vae.wan_video import _comfy_vae_config, WanNativeVideoVAE, load_wan_native_video_vae


def _original_vae_key(key: str, depth: int = 1) -> str:
    k = key
    if k.startswith("quant_conv."):
        return "conv1." + k[11:]
    if k.startswith("post_quant_conv."):
        return "conv2." + k[16:]
    for part in ("encoder", "decoder"):
        swaps = {
            f"{part}.conv_in.": f"{part}.conv1.",
            f"{part}.norm_out.": f"{part}.head.0.",
            f"{part}.conv_out.": f"{part}.head.2.",
            f"{part}.mid_block.resnets.0.": f"{part}.middle.0.",
            f"{part}.mid_block.attentions.0.": f"{part}.middle.1.",
            f"{part}.mid_block.resnets.1.": f"{part}.middle.2.",
        }
        for prefix, dest in swaps.items():
            if k.startswith(prefix):
                k = dest + k[len(prefix):]
                break
    match = re.match(r"^(encoder)\.down_blocks\.(\d+)\.(resnets|downsampler)\.?([\d]*)(?:\.)?(.*)$", k)
    if match:
        part, i, typ, j, rest = match.groups()
        index = int(j) if typ == "resnets" else depth
        k = f"encoder.downsamples.{i}.downsamples.{index}.{rest}"
    match = re.match(r"^(decoder)\.up_blocks\.(\d+)\.(resnets|upsampler)\.?([\d]*)(?:\.)?(.*)$", k)
    if match:
        part, i, typ, j, rest = match.groups()
        index = int(j) if typ == "resnets" else depth + 1
        k = f"decoder.upsamples.{i}.upsamples.{index}.{rest}"
    k = k.replace(".norm1.", ".residual.0.")
    k = k.replace(".norm2.", ".residual.3.")
    k = k.replace(".conv1.", ".residual.2.") if ".downsamples." in k or ".upsamples." in k or ".middle." in k else k
    k = k.replace(".conv2.", ".residual.6.")
    k = k.replace(".conv_shortcut.", ".shortcut.")
    return k


def test_vae_comfy_key_map_full_topology():
    cfg = _comfy_vae_config(dict(base_dim=4, decoder_base_dim=4, num_res_blocks=1))
    with torch.device("meta"):
        module = WanNativeVideoVAE(cfg)
    names = set(module.state_dict())
    raw = {_original_vae_key(k) for k in names}
    mapped = {wan_vae_key(k, num_res_blocks=1) for k in raw}
    if mapped != names:
        print("missing", sorted(names - mapped)[:25]); print("extra", sorted(mapped-names)[:25])
    assert len(raw) == len(names)
    assert mapped == names
    assert len(validated_key_map(raw, lambda x: wan_vae_key(x, num_res_blocks=1), names)) == len(names)


def test_no_unexpected_or_alias_keys():
    with pytest.raises(ValueError, match="duplicates"):
        validated_key_map({"one", "two"}, lambda x: "same", {"same"})
    with pytest.raises(ValueError, match="missing"):
        validated_key_map({"one"}, lambda x: x, {"one", "two"})


def test_dit_comfy_names():
    expected = {
        "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
        "time_projection.1.bias": "condition_embedder.time_proj.bias",
        "head.modulation": "scale_shift_table",
        "blocks.0.norm2.weight": "blocks.0.norm3.weight",
        "blocks.0.norm3.weight": "blocks.0.norm2.weight",
        "blocks.10.self_attn.q.weight": "blocks.10.attn1.to_q.weight",
        "blocks.5.cross_attn.o.bias": "blocks.5.attn2.to_out.0.bias",
        "blocks.2.ffn.0.weight": "blocks.2.ffn.net.0.proj.weight",
        "blocks.2.ffn.2.weight": "blocks.2.ffn.net.2.weight",
    }
    assert {k: wan_dit_key(k) for k in expected} == expected
