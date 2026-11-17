"""Original LTX-2.3/2.5 Safetensors to modular PyTorch components.

TheNoise owns the I/O and residency. This loader never materializes a full
44-GiB state dictionary or imports the Lightricks Python packages. Numerical
modules are native PyTorch; neither Diffusers nor ComfyUI is a runtime
dependency. ComfyUI is used only as a checkpoint-layout reference.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

import torch
from thenoise.utils.safetensors import MemoryEfficientSafeOpen

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
    from thenoise.dit.ltx_video.transformer import LTX2VideoTransformer3DModel
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
    from thenoise.vae.ltx_video import LTXVideoVAE
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
        return LTXVideoVAE(**cfg)


def make_audio_vae():
    from thenoise.vae.ltx_audio import LTXAudioVAE
    cfg = dict(
        base_channels=128, output_channels=2, ch_mult=(1,2,4),
        num_res_blocks=2, attn_resolutions=None, in_channels=2,
        resolution=256, latent_channels=8, norm_type="pixel",
        causality_axis="height", dropout=0.0, mid_block_add_attention=False,
        sample_rate=16000, mel_hop_length=160, is_causal=True,
        mel_bins=64, double_z=True,
    )
    with torch.device("meta"):
        return LTXAudioVAE(**cfg)


def make_vocoder():
    from thenoise.dit.ltx_video.vocoder import LTXVocoderWithBWE
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
        return LTXVocoderWithBWE(**cfg)

def _upscaler_key(raw: str) -> tuple[str, str]:
    """Keep ``upsampler.``: it is a real module name, not a file wrapper.

    LTX2LatentUpsamplerModel.state_dict() contains ``upsampler.0.weight`` and
    ``upsampler.0.bias``, matching the original LTX-2.5 checkpoint verbatim.
    The strict assignment path verifies any other keys against model state.
    """
    return "upscaler", raw


def make_upsampler():
    # Only the released spatial x2 model, using TheNoise's existing module seam.
    from thenoise.dit.ltx_video.upscaler import LTXSpatialUpscaler
    with torch.device("meta"):
        return LTXSpatialUpscaler(in_channels=128, mid_channels=1024,
                                 num_blocks_per_stage=4)


def make_connectors(version: str, gemma_text_config):
    from thenoise.dit.ltx_video.connectors import LTX2TextConnectors
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


def load_ltx_component(
    name: str, module: torch.nn.Module, groups: dict[Path, set[str]],
    *, cancel: Callable[[], None], dtype: torch.dtype,
) -> torch.nn.Module:
    """Assemble one LTX component's split checkpoint files, then use TheNoise's loader.

    The component-specific *key mapping* belongs here. Quantization, ConvRot and
    assignment are handled by the existing shared load_quantized_state_dict.
    In particular this does not add a new generic loader to thenoise.utils.
    """
    from thenoise.utils.checkpoint import CHECKPOINT_MARKER_KEYS
    from thenoise.utils.loader import RUNTIME_STATE_SUFFIXES, load_quantized_state_dict
    from thenoise.utils.safetensors import unwrap_key

    expected = {key: tuple(value.shape) for key, value in module.state_dict().items()}
    weights: dict[str, torch.Tensor] = {}
    quant_suffixes = (".weight_scale", ".scale_weight", ".scale_input",
                      ".input_scale", ".comfy_quant")
    quant_dtypes = {torch.int8}
    quant_dtypes.update(getattr(torch, attr) for attr in ("float8_e4m3fn", "float8_e5m2")
                        if hasattr(torch, attr))

    for path, roles in groups.items():
        if name not in roles:
            continue
        cancel()
        with MemoryEfficientSafeOpen(str(path)) as source:
            for index, raw in enumerate(source.keys()):
                if index % 16 == 0:
                    cancel()
                if raw.endswith(RUNTIME_STATE_SUFFIXES) or unwrap_key(raw) in CHECKPOINT_MARKER_KEYS:
                    continue
                if "dit" in roles or "connectors" in roles:
                    mapped = split_ltx_key(raw)
                elif "vae" in roles:
                    mapped = ("vae", raw.removeprefix("vae."))
                elif "audio_vae" in roles:
                    mapped = (("vocoder", raw.removeprefix("vocoder."))
                              if raw.startswith("vocoder.") else
                              ("audio_vae", raw.removeprefix("audio_vae.")))
                elif "upscaler" in roles:
                    mapped = _upscaler_key(raw)
                else:
                    raise RuntimeError(f"unknown LTX checkpoint group: {roles}")
                if mapped is None or mapped[0] != name:
                    continue
                key = _map_component(name, mapped[1])
                if key is None:
                    continue
                # A scale/marker belongs to a weight, not the module's state_dict.
                suffix = next((s for s in quant_suffixes if key.endswith(s)), None)
                if suffix is not None:
                    parent = key[:-len(suffix)]
                    if parent + ".weight" not in expected:
                        raise ValueError(f"{path}: unexpected LTX quantization metadata {raw!r}")
                elif key not in expected:
                    # Checkpoint-wide metadata is not a learned parameter.
                    if key == "scaled_fp8" or key.endswith(".scaled_fp8"):
                        continue
                    raise ValueError(f"{path}: unexpected {name} tensor {raw!r} -> {key!r}")
                elif tuple(source.header[raw]["shape"]) != expected[key]:
                    raise ValueError(f"{path}: {raw!r} shape "
                                     f"{source.header[raw]['shape']} != {expected[key]}")
                if key in weights:
                    raise ValueError(f"{path}: duplicate LTX {name} weight {key!r}")
                weights[key] = source.get_tensor(raw)

    missing = set(expected) - set(weights)
    if missing:
        raise ValueError(f"LTX {name} checkpoint missing={sorted(missing)[:8]} "
                         "(no random/meta weight fallback)")
    is_quantized = any(key in expected and tensor.dtype in quant_dtypes
                       for key, tensor in weights.items())
    if is_quantized:
        # Shared quant loader checks marker structure and restores the native
        # QuantizedTensor layout. Do not implement a second quantized assigner.
        for key, value in weights.items():
            if key not in expected or value.dtype != torch.int8:
                continue
            if not key.endswith(".weight"):
                raise ValueError(f"LTX {name} unexpected INT8 parameter {key!r}")
            stem = key[:-len(".weight")]
            scales = [stem + s for s in (".weight_scale", ".scale_weight")
                      if stem + s in weights]
            if len(scales) != 1 or stem + ".comfy_quant" not in weights:
                raise ValueError(f"LTX {name}: INT8 {key} requires a single "
                                 "weight scale and .comfy_quant marker")
        load_quantized_state_dict(module, weights, dtype=dtype)
    else:
        stray = sorted(set(weights) - set(expected))
        if stray:
            raise ValueError(f"LTX {name} checkpoint has unused quant metadata: {stray[:8]}")
        module.load_state_dict(
            {key: value.to(dtype=dtype) if value.is_floating_point() else value
             for key, value in weights.items()},
            strict=True, assign=True,
        )
    leftover = [key for key, weight in module.named_parameters() if weight.is_meta]
    leftover += [key for key, weight in module.named_buffers() if weight.is_meta]
    if leftover:
        raise RuntimeError(f"LTX {name} contains uninitialized tensors: {leftover[:8]}")
    return module.eval()


def load_original_components(
    dit_path: str | Path, vae_path: str | Path,
    audio_vae_path: str | Path, upscaler_path: str | Path,
    *, version: str, cancel: Callable[[], None], dtype=torch.bfloat16,
    gemma_text_config=None, text_encoder_path: str | Path | None = None,
    text_projection_path: str | Path | None = None,
) -> dict:
    """Original LTX file roles and mappings, using TheNoise's quantized loader.

    Multiple explicitly provided checkpoint paths may be identical (LTX 2.3
    monolith). The LTX 2.5 Gemma *with-proj* file contains its projections.
    """
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
    for component, path in sources.items():
        groups.setdefault(path.expanduser().resolve(), set()).add(component)
    groups[Path(audio_vae_path).expanduser().resolve()].add("vocoder")
    if "connectors" in modules:
        groups[Path(dit_path).expanduser().resolve()].add("connectors")
        if version == VERSION_25:
            if text_encoder_path is None:
                raise ValueError("LTX 2.5 needs a Gemma with-proj text checkpoint")
            groups.setdefault(Path(text_encoder_path).expanduser().resolve(), set()).add("connectors")
        elif text_projection_path is not None:
            groups.setdefault(Path(text_projection_path).expanduser().resolve(), set()).add("connectors")

    for component, module in modules.items():
        load_ltx_component(component, module, groups, cancel=cancel, dtype=dtype)
    return modules
