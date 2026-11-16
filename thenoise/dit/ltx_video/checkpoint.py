"""Original LTX-2.3/2.5 Safetensors to modular PyTorch components.

TheNoise owns the I/O and residency. This loader never materializes a full
44-GiB state dictionary or imports the Lightricks Python packages. Numerical
modules come from the optional, *PyPI-released*, Apache-2 diffusers package.
No Diffusers pipeline is instantiated. ComfyUI is a layout reference only.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

import torch
from thenoise.utils.safetensors import MemoryEfficientSafeOpen
from thenoise.utils.loader import load_streamed_components

log = logging.getLogger(__name__)

VERSION_23 = "2.3"
VERSION_25 = "2.5"


def version_from_checkpoint(path: str | Path) -> str | None:
    """Read headers, never weight data; don't detect from a guessed basename alone."""
    path = Path(path).expanduser()
    if not path.is_file():
        return None
    with MemoryEfficientSafeOpen(str(path)) as f:
        keys = set(f.keys())
        meta = f.metadata() or {}
    cfg = meta.get("config")
    try:
        cfg = json.loads(cfg) if isinstance(cfg, str) else (cfg or {})
    except (TypeError, ValueError):
        cfg = {}
    version = str(meta.get("model_version", "") or cfg.get("model_version", ""))
    if version.startswith("2.5"):
        return VERSION_25
    if version.startswith("2.3"):
        return VERSION_23
    # The absolute-position embedding is LTX-2.5-specific. The diffusion
    # checkpoint may be split or use the Comfy wrapper prefix.
    if any(k.endswith("keyframes_abs_pos_embedding") for k in keys):
        return VERSION_25
    if any(k.endswith("patchify_proj.weight") for k in keys) and any(
        k.endswith("audio_patchify_proj.weight") for k in keys
    ):
        # Version 2.5 carries the keyframe embedding; its absence is 2.3.
        # Never infer an architecture from user-controlled path names.
        return VERSION_23
    return None


# Explicit transformations derived from the Apache-2 reference converter.
_COMMON_TRANSFORMER_RENAMES = (
    ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
    ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
    ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
    ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
    ("scale_shift_table_a2v_ca_video", "video_a2v_cross_attn_scale_shift_table"),
    ("scale_shift_table_a2v_ca_audio", "audio_a2v_cross_attn_scale_shift_table"),
    ("audio_patchify_proj", "audio_proj_in"),
    ("patchify_proj", "proj_in"),
    ("audio_prompt_adaln_single", "audio_prompt_adaln"),
    ("prompt_adaln_single", "prompt_adaln"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
)
_CONNECTOR_PREFIXES = (
    "connectors.", "video_embeddings_connector.",
    "audio_embeddings_connector.", "text_embedding_projection.",
    "video_connector.", "audio_connector.",
)


def _strip_dit_prefix(k: str) -> str:
    for prefix in ("model.diffusion_model.", "diffusion_model.", "net."):
        if k.startswith(prefix):
            return k[len(prefix):]
    return k


def split_ltx_key(raw: str) -> tuple[str, str] | None:
    """Return (component, target parameter) for an original checkpoint tensor."""
    k = _strip_dit_prefix(raw)
    if k.startswith(_CONNECTOR_PREFIXES):
        if k.startswith("connectors."):
            k = k.removeprefix("connectors.")
        for a, b in (
            ("video_embeddings_connector", "video_connector"),
            ("audio_embeddings_connector", "audio_connector"),
            ("transformer_1d_blocks", "transformer_blocks"),
            ("text_embedding_projection.audio_aggregate_embed", "audio_text_proj_in"),
            ("text_embedding_projection.video_aggregate_embed", "video_text_proj_in"),
            ("q_norm", "norm_q"), ("k_norm", "norm_k"),
        ):
            k = k.replace(a, b)
        return "connectors", k
    if k.startswith("vae.") or k.startswith("audio_vae.") or k.startswith("vocoder."):
        prefix, rest = k.split(".", 1)
        return prefix, rest
    if k.startswith("duration_head."):
        return None  # Optional duration prediction isn't needed for explicit frames.
    if k.startswith("adaln_single."):
        return "dit", "time_embed." + k[len("adaln_single."):]
    if k.startswith("audio_adaln_single."):
        return "dit", "audio_time_embed." + k[len("audio_adaln_single."):]
    for a, b in _COMMON_TRANSFORMER_RENAMES:
        k = k.replace(a, b)
    return "dit", k


def _remap_vae(k: str) -> str | None:
    if k.startswith("per_channel_statistics.channel") or k.startswith("per_channel_statistics.mean-of-stds"):
        return None
    replacements = (
        ("down_blocks.8.", "mid_block."),
        ("down_blocks.7.", "down_blocks.3.downsamplers.0."),
        ("down_blocks.6.", "down_blocks.3."),
        ("down_blocks.5.", "down_blocks.2.downsamplers.0."),
        ("down_blocks.4.", "down_blocks.2."),
        ("down_blocks.3.", "down_blocks.1.downsamplers.0."),
        ("down_blocks.2.", "down_blocks.1."),
        ("down_blocks.1.", "down_blocks.0.downsamplers.0."),
        ("up_blocks.8.", "up_blocks.3."),
        ("up_blocks.7.", "up_blocks.3.upsamplers.0."),
        ("up_blocks.6.", "up_blocks.2."),
        ("up_blocks.5.", "up_blocks.2.upsamplers.0."),
        ("up_blocks.4.", "up_blocks.1."),
        ("up_blocks.3.", "up_blocks.1.upsamplers.0."),
        ("up_blocks.2.", "up_blocks.0."),
        ("up_blocks.1.", "up_blocks.0.upsamplers.0."),
        ("up_blocks.0.", "mid_block."),
    )
    # Only a single block-index transformation. Replacing every occurrence in
    # order would corrupt e.g. down_blocks.7 -> down_blocks.3 -> down_blocks.1.
    prefix = ""
    if k.startswith(("encoder.", "decoder.")):
        prefix, k = k.split(".", 1)
        prefix += "."
    for src, dst in replacements:
        if k.startswith(src):
            k = dst + k[len(src):]
            break
    k = prefix + k
    return (k.replace("res_blocks", "resnets")
             .replace("per_channel_statistics.mean-of-means", "latents_mean")
             .replace("per_channel_statistics.std-of-means", "latents_std"))


def _remap_vocoder(k: str) -> str:
    for src, dst in (
        ("resblocks", "resnets"), ("conv_pre", "conv_in"),
        ("conv_post", "conv_out"), ("act_post", "act_out"),
        ("downsample.lowpass", "downsample"), (".ups.", ".upsamplers."),
    ):
        k = k.replace(src, dst)
    return k


def _map_component(comp: str, k: str) -> str | None:
    if comp == "vae":
        return _remap_vae(k)
    if comp == "audio_vae":
        return (k.replace("per_channel_statistics.mean-of-means", "latents_mean")
                 .replace("per_channel_statistics.std-of-means", "latents_std"))
    if comp == "vocoder":
        return _remap_vocoder(k)
    return k


def make_transformer(version: str):
    from diffusers import LTX2VideoTransformer3DModel
    cfg = dict(
        in_channels=128, out_channels=128, patch_size=1, patch_size_t=1,
        num_attention_heads=32, attention_head_dim=128, cross_attention_dim=4096,
        vae_scale_factors=(8, 32, 32), pos_embed_max_pos=20,
        base_height=2048, base_width=2048,
        gated_attn=True, cross_attn_mod=True,
        audio_in_channels=128, audio_out_channels=128,
        audio_patch_size=1, audio_patch_size_t=1,
        audio_num_attention_heads=32, audio_attention_head_dim=64,
        audio_cross_attention_dim=2048, audio_scale_factor=4,
        audio_pos_embed_max_pos=20, audio_sampling_rate=16000,
        audio_hop_length=160, audio_gated_attn=True, audio_cross_attn_mod=True,
        num_layers=48, activation_fn="gelu-approximate",
        qk_norm="rms_norm_across_heads", norm_elementwise_affine=False,
        norm_eps=1e-6, caption_channels=3840, attention_bias=True,
        attention_out_bias=True, rope_theta=10000., rope_double_precision=True,
        causal_offset=1, timestep_scale_multiplier=1000,
        cross_attn_timestep_scale_multiplier=1000, rope_type="split",
        use_prompt_embeddings=False, perturbed_attn=True,
        ff_bias=version != VERSION_25, use_keyframes_abs_pos_embedding=version == VERSION_25,
    )
    from thenoise.dit.quantized import replace_linears
    with torch.device("meta"):
        model = LTX2VideoTransformer3DModel(**cfg)
        replace_linears(model)
        return model


def make_video_vae():
    from diffusers import AutoencoderKLLTX2Video
    cfg = dict(
        in_channels=3, out_channels=3, latent_channels=128,
        block_out_channels=(256,512,1024,1024),
        down_block_types=("LTX2VideoDownBlock3D",)*4,
        decoder_block_out_channels=(256,512,512,1024),
        layers_per_block=(4,6,4,2,2), decoder_layers_per_block=(4,6,4,2,2),
        spatio_temporal_scaling=(True,)*4,
        decoder_spatio_temporal_scaling=(True,)*4,
        decoder_inject_noise=(False,)*5,
        downsample_type=("spatial","temporal","spatiotemporal","spatiotemporal"),
        upsample_type=("spatiotemporal","spatiotemporal","temporal","spatial"),
        upsample_residual=(False,)*4, upsample_factor=(2,2,1,2),
        timestep_conditioning=False, patch_size=4, patch_size_t=1,
        resnet_norm_eps=1e-6, encoder_causal=True, decoder_causal=False,
        encoder_spatial_padding_mode="zeros", decoder_spatial_padding_mode="zeros",
        spatial_compression_ratio=32, temporal_compression_ratio=8,
    )
    with torch.device("meta"):
        return AutoencoderKLLTX2Video(**cfg)


def make_audio_vae():
    from diffusers import AutoencoderKLLTX2Audio
    cfg = dict(
        base_channels=128, output_channels=2, ch_mult=(1,2,4),
        num_res_blocks=2, attn_resolutions=None, in_channels=2,
        resolution=256, latent_channels=8, norm_type="pixel",
        causality_axis="height", dropout=0.0, mid_block_add_attention=False,
        sample_rate=16000, mel_hop_length=160, is_causal=True,
        mel_bins=64, double_z=True,
    )
    with torch.device("meta"):
        return AutoencoderKLLTX2Audio(**cfg)


def make_vocoder():
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE
    cfg = dict(
        in_channels=128, hidden_channels=1536, out_channels=2,
        upsample_kernel_sizes=[11,4,4,4,4,4],
        upsample_factors=[5,2,2,2,2,2],
        resnet_kernel_sizes=[3,7,11],
        resnet_dilations=[[1,3,5]]*3, act_fn="snakebeta",
        leaky_relu_negative_slope=0.1, antialias=True,
        antialias_ratio=2, antialias_kernel_size=12,
        final_act_fn=None, final_bias=False,
        bwe_in_channels=128, bwe_hidden_channels=512,
        bwe_out_channels=2,
        bwe_upsample_kernel_sizes=[12,11,4,4,4],
        bwe_upsample_factors=[6,5,2,2,2],
        bwe_resnet_kernel_sizes=[3,7,11],
        bwe_resnet_dilations=[[1,3,5]]*3,
        bwe_act_fn="snakebeta", bwe_leaky_relu_negative_slope=0.1,
        bwe_antialias=True, bwe_antialias_ratio=2,
        bwe_antialias_kernel_size=12, bwe_final_act_fn=None,
        bwe_final_bias=False, filter_length=512, hop_length=80,
        window_length=512, num_mel_channels=64,
        input_sampling_rate=16000, output_sampling_rate=48000,
    )
    with torch.device("meta"):
        model = LTX2VocoderWithBWE(**cfg)

    # Diffusers' LTX2VocoderWithBWE.resampler is constructed with
    # persistent=False: its deterministic Hann filter is absent from the
    # checkpoint/state_dict. Building the vocoder on meta therefore leaves
    # resampler.filter uninitialized even after *all* checkpoint weights load.
    # Recreate this small buffer on CPU, without materializing any weights or
    # changing the checkpoint-key validation performed by _assign().
    from diffusers.pipelines.ltx2.vocoder import UpSample1d
    with torch.device("cpu"):
        resampler = UpSample1d(
            ratio=cfg["output_sampling_rate"] // cfg["input_sampling_rate"],
            window_type="hann", persistent=False,
        )
    model.resampler.filter = resampler.filter
    return model


def _upscaler_key(raw: str) -> tuple[str, str]:
    """Keep ``upsampler.``: it is a real module name, not a file wrapper.

    LTX2LatentUpsamplerModel.state_dict() contains ``upsampler.0.weight`` and
    ``upsampler.0.bias``, matching the original LTX-2.5 checkpoint verbatim.
    The strict assignment path verifies any other keys against model state.
    """
    return "upscaler", raw


def make_upsampler():
    from diffusers.models.latent_upscaler.latent_upscaler_ltx2 import LTX2LatentUpsamplerModel
    with torch.device("meta"):
        return LTX2LatentUpsamplerModel(
            in_channels=128, mid_channels=1024, num_blocks_per_stage=4,
            dims=3, spatial_upsample=True, temporal_upsample=False,
            rational_spatial_scale=2.0, use_rational_resampler=False,
        )


def make_connectors(version: str, gemma_text_config):
    from diffusers.models.condition_embedders.condition_embedder_ltx2 import LTX2TextConnectors
    if version == VERSION_23:
        caption_channels, factor = 3840, 49
    else:
        caption_channels, factor = (
            gemma_text_config.hidden_size, gemma_text_config.num_hidden_layers + 1
        )
    cfg = dict(
        caption_channels=caption_channels, text_proj_in_factor=factor,
        video_connector_num_attention_heads=32,
        video_connector_attention_head_dim=128,
        video_connector_num_layers=8, video_connector_num_learnable_registers=128,
        video_gated_attn=True, audio_connector_num_attention_heads=32,
        audio_connector_attention_head_dim=64,
        audio_connector_num_layers=8, audio_connector_num_learnable_registers=128,
        audio_gated_attn=True, connector_rope_base_seq_len=4096,
        rope_theta=10000., rope_double_precision=True,
        causal_temporal_positioning=False, rope_type="split",
        per_modality_projections=True, video_hidden_dim=4096,
        audio_hidden_dim=2048, proj_bias=True,
    )
    from thenoise.dit.quantized import replace_linears
    with torch.device("meta"):
        model = LTX2TextConnectors(**cfg)
        replace_linears(model)
        return model


def load_original_components(
    dit_path: str | Path, vae_path: str | Path,
    audio_vae_path: str | Path, upscaler_path: str | Path,
    *, version: str, cancel: Callable[[], None], dtype=torch.bfloat16,
    gemma_text_config=None, text_encoder_path: str | Path | None = None,
    text_projection_path: str | Path | None = None,
) -> dict:
    """Load only original checkpoint tensors required by these LTX-2.x models."""
    from diffusers import __version__ as diffusers_version
    if tuple(map(int, diffusers_version.split(".")[:2])) < (0, 43):
        raise RuntimeError("LTX requires diffusers >=0.43 from PyPI (no git dependency)")

    modules = dict(
        dit=make_transformer(version), vae=make_video_vae(),
        audio_vae=make_audio_vae(), vocoder=make_vocoder(),
        upscaler=make_upsampler(),
    )
    if gemma_text_config is not None:
        modules["connectors"] = make_connectors(version, gemma_text_config)
    sources = {
        "dit": Path(dit_path), "vae": Path(vae_path),
        "audio_vae": Path(audio_vae_path), "upscaler": Path(upscaler_path),
    }
    groups: dict[Path, set[str]] = {}
    for comp, path in sources.items():
        groups.setdefault(path.resolve(), set()).add(comp)
    # Both releases store connector attention weights in the DiT. 2.5's
    # Gemma-4 "with-proj" file additionally stores the text aggregation
    # projections, not the audio/video attention blocks.
    if version == VERSION_25 and text_encoder_path is None:
        raise ValueError("LTX 2.5 requires the Gemma-4 with-proj text checkpoint")
    if "connectors" in modules:
        groups.setdefault(Path(dit_path).resolve(), set()).add("connectors")
        if version == VERSION_25:
            groups.setdefault(Path(text_encoder_path).resolve(), set()).add("connectors")
        elif text_projection_path is not None:
            # The LTX 2.3 community DiT omits the four learned Gemma projections.
            # Feed the standalone projection through the existing strict split loader.
            groups.setdefault(Path(text_projection_path).resolve(), set()).add("connectors")
    groups[Path(audio_vae_path).resolve()].add("vocoder")
    assigned: set[tuple[str, str]] = set()
    for path, names in groups.items():
        cancel()
        with MemoryEfficientSafeOpen(str(path)) as f:
            items = {k: m for k,m in modules.items() if k in names}
            if "dit" in items or "connectors" in items:
                selector = split_ltx_key
            elif "vae" in items:
                selector = lambda raw: ("vae", raw.removeprefix("vae."))
            elif "audio_vae" in items:
                def selector(raw):
                    if raw.startswith("vocoder."):
                        return "vocoder", raw.removeprefix("vocoder.")
                    return "audio_vae", raw.removeprefix("audio_vae.")
            elif "upscaler" in items:
                selector = _upscaler_key
            else:
                raise RuntimeError(f"Unhandled LTX checkpoint roles: {sorted(names)}")
            def mapped_selector(raw):
                selected = selector(raw)
                if selected is None:
                    return None
                component, key = selected
                mapped = _map_component(component, key)
                return (component, mapped) if mapped is not None else None

            loaded = load_streamed_components(
                f, items, cancel=cancel, selector=mapped_selector, dtype=dtype,
                already_loaded=assigned,
                defer_complete={"connectors"} if "connectors" in items else None,
            )
            assigned.update((name, key) for name, keys in loaded.items() for key in keys)

    # A split component is only valid once every source has contributed.
    # Never accept a partial connector or silently replace a missing weight.
    missing = sorted((name, key) for name, module in modules.items()
                     for key in module.state_dict() if (name, key) not in assigned)
    if missing:
        raise ValueError(
            f"LTX checkpoint mismatch across component files: missing={missing[:8]} "
            "(never continue with random/meta weights)"
        )
    for name, module in modules.items():
        leftover = [key for key, value in module.named_parameters() if value.is_meta]
        leftover += [key for key, value in module.named_buffers() if value.is_meta]
        if leftover:
            raise RuntimeError(
                f"LTX component {name} has uninitialized meta tensors: {leftover[:8]}"
            )
    return {name: m.eval() for name, m in modules.items()}
