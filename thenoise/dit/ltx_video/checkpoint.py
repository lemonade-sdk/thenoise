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
from collections.abc import Iterable
from pathlib import Path
from typing import Callable

import torch

log = logging.getLogger(__name__)

VERSION_23 = "2.3"
VERSION_25 = "2.5"


def version_from_checkpoint(path: str | Path) -> str | None:
    """Identify a released LTX file using the shared header representation."""
    from thenoise.utils.safetensors import read_header

    path = Path(path).expanduser()
    if not path.is_file():
        return None
    header = read_header(str(path))
    cfg = header.metadata_json("config")
    version = str(header.metadata.get("model_version", "") or cfg.get("model_version", ""))
    if version.startswith("2.5"):
        return VERSION_25
    if version.startswith("2.3"):
        return VERSION_23
    keys = header.names
    if "keyframes_abs_pos_embedding" in keys:
        return VERSION_25
    if "patchify_proj.weight" in keys and "audio_patchify_proj.weight" in keys:
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
    with torch.device("meta"):
        model = LTX2VideoTransformer3DModel(**cfg)
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
    with torch.device("meta"):
        model = LTX2TextConnectors(**cfg)
        return model


#: The quantization metadata a repacked checkpoint carries beside its weights.
QUANT_SUFFIXES = (".weight_scale", ".scale_weight", ".scale_input",
                  ".input_scale", ".comfy_quant")

#: The parts a file can name by prefix, alone or inside an all-in-one checkpoint.
COMPONENT_PREFIXES = ("vae", "audio_vae", "vocoder")

#: Which request role names a component's own file.
ROLE_OF_COMPONENT = {"dit": "", "vae": "vae", "audio_vae": "audio_vae",
                     "vocoder": "audio_vae", "upscaler": "upscaler",
                     "connectors": "text_projection"}


def _flag_for(name: str, version: str) -> str:
    """The flag that supplies a missing component."""
    if name == "connectors":
        return ("--text-encoder, which for LTX 2.5 is a Gemma checkpoint with its "
                "projections" if version == VERSION_25 else "--text-projection")
    return {"dit": "--dit", "vae": "--vae", "audio_vae": "--audio-vae",
            "vocoder": "--audio-vae", "upscaler": "--spatial-upscaler"}.get(name, "")


def _mapped(name: str, raw: str, *, transformer: bool) -> tuple[str, str | None]:
    """Which component a checkpoint tensor spells, and the parameter it fills there.

    A file holding the transformer spells every key as :func:`split_ltx_key` reads it; a
    file holding one part spells it bare, so a prefix names the part instead.
    """
    from thenoise.utils.safetensors import unwrap_key

    key = unwrap_key(raw)
    if transformer or name in ("dit", "connectors"):
        return split_ltx_key(key) or ("", None)
    head, _, rest = key.partition(".")
    if head in COMPONENT_PREFIXES:
        return head, _map_component(head, rest)
    return name, _map_component(name, key)


def select_ltx_keys(name: str, header, expected: dict[str, tuple[int, ...]],
                    *, strict: bool = True, transformer: bool = False) -> dict[str, str]:
    """The tensors of one file that fill component ``name``: raw key -> module key.

    Lenient skips what the component has no parameter for instead of refusing it, so a
    file can be asked whether it carries the part at all.
    """
    from thenoise.utils.checkpoint import CHECKPOINT_MARKER_KEYS
    from thenoise.utils.loader import RUNTIME_STATE_SUFFIXES
    from thenoise.utils.safetensors import unwrap_key

    def refuse(reason):
        if strict:
            raise ValueError(reason)

    selected: dict[str, str] = {}
    for raw, entry in header.raw_entries.items():
        if unwrap_key(raw) in CHECKPOINT_MARKER_KEYS or raw.endswith(RUNTIME_STATE_SUFFIXES):
            continue
        component, target = _mapped(name, raw, transformer=transformer)
        if component != name or target is None:
            continue
        suffix = next((v for v in QUANT_SUFFIXES if target.endswith(v)), None)
        if suffix:
            if target[:-len(suffix)] + ".weight" not in expected:
                refuse(f'{header.path}: unexpected quantization metadata {raw!r}')
                continue
        elif target not in expected:
            # A multi-role file may carry an FP8 marker, not a model tensor.
            if target == "scaled_fp8" or target.endswith(".scaled_fp8"):
                continue
            refuse(f'{header.path}: unexpected {name} tensor {raw!r} -> {target!r}')
            continue
        elif tuple(entry["shape"]) != expected[target]:
            refuse(f'{header.path}: {raw!r} shape {entry["shape"]} != {expected[target]}')
            continue
        if target in selected.values():
            refuse(f'{header.path}: duplicate {name} weight {target!r}')
            continue
        selected[raw] = target
    return selected


def _header_of(path: Path, cache: dict[Path, object] | None):
    """A file's header, parsed once per load however many components live in it."""
    from thenoise.utils.safetensors import read_header

    if cache is None:
        return read_header(str(path))
    if path not in cache:
        cache[path] = read_header(str(path))
    return cache[path]


def load_ltx_component(
    name: str, module: torch.nn.Module, paths: Iterable[Path],
    *, cancel: Callable[[], None], dtype: torch.dtype,
    header_cache: dict[Path, object] | None = None, transformer: Path | None = None,
) -> torch.nn.Module:
    """Fill one component from the files that carry it, materializing only the selected
    keys. ``transformer`` is the file holding the transformer, which spells its keys its
    own way. Strict assignment and quantization stay in ``thenoise.utils.loader``.
    """
    from thenoise.utils.loader import load_module_state, load_quantized_state_dict
    from thenoise.utils.safetensors import load_safetensors

    expected = {key: tuple(t.shape) for key, t in module.state_dict().items()}
    weights: dict[str, torch.Tensor] = {}
    used: list[str] = []

    for path in paths:
        cancel()
        selected = select_ltx_keys(name, _header_of(path, header_cache), expected,
                              transformer=(path == transformer))
        if not selected:
            continue
        if any(target in weights for target in selected.values()):
            raise ValueError(f"{path}: duplicate {name} weight")
        data = load_safetensors(str(path), device="cpu", keys=set(selected))
        for raw, tensor in data.items():
            weights[selected[raw]] = tensor
        used.append(str(path))

    missing = set(expected) - set(weights)
    if missing:
        raise ValueError(f'LTX {name} checkpoint missing={sorted(missing)[:8]} '
                         '(no random/meta weight fallback)')

    quantized = (any(key.endswith((".weight_scale", ".scale_weight")) for key in weights)
                 or any(t.dtype == torch.int8 or str(t.dtype).startswith("torch.float8")
                        for t in weights.values()))
    if quantized:
        load_quantized_state_dict(module, weights, dtype=dtype)
    else:
        load_module_state(module, {k: v.to(dtype=dtype) if v.is_floating_point() else v
                                   for k, v in weights.items()},
                          ', '.join(used), what=f'LTX {name}')
    leftover = [key for key, p in module.named_parameters() if p.is_meta]
    leftover += [key for key, b in module.named_buffers() if b.is_meta]
    if leftover:
        raise RuntimeError(f'LTX {name} uninitialized tensors: {leftover[:8]}')
    return module.eval()


def load_original_components(
    files: dict[str, str | Path], *, version: str,
    cancel: Callable[[], None], dtype=torch.bfloat16, gemma_text_config=None,
    header_cache: dict[Path, object] | None = None,
) -> tuple[dict, dict[str, Path]]:
    """Build every LTX component from the files a request named by role (``""`` is the
    transformer), and answer where each one came from.

    A part comes from the file named for it, else from whichever named file carries its
    tensors; the connector pair is read from every file that carries it.
    """
    modules = dict(
        dit=make_transformer(version), vae=make_video_vae(),
        audio_vae=make_audio_vae(), vocoder=make_vocoder(),
        upscaler=make_upsampler(),
    )
    if gemma_text_config is not None:
        modules["connectors"] = make_connectors(version, gemma_text_config)
    if header_cache is None:
        header_cache = {}

    given = {role: Path(path).expanduser().resolve()
             for role, path in files.items() if path}
    primary = given.get("")
    order = [p for p in given.values()]
    sources: dict[str, list[Path]] = {}
    for name, module in modules.items():
        expected = {key: tuple(t.shape) for key, t in module.state_dict().items()}
        named = given.get(ROLE_OF_COMPONENT[name])
        carriers = [p for p in ([named] if named else []) + order
                    if select_ltx_keys(name, _header_of(p, header_cache), expected,
                                       strict=False, transformer=(p == primary))]
        chosen = carriers if name == "connectors" else carriers[:1]
        if not chosen:
            flag = _flag_for(name, version)
            raise ValueError(
                f"LTX {name} weights are in none of the checkpoints given"
                + (f"; pass {flag}" if flag else ""))
        sources[name] = chosen
    log.info("LTX %s components: %s", version,
             ", ".join(f"{name} from {', '.join(p.name for p in paths)}"
                       for name, paths in sorted(sources.items())))

    for name, module in modules.items():
        load_ltx_component(name, module, sources[name], cancel=cancel, dtype=dtype,
                           header_cache=header_cache, transformer=primary)
    return modules, {name: paths[0] for name, paths in sources.items()}
