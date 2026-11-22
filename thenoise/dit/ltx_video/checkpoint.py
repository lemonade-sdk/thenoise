"""Original LTX-2.3/2.5 Safetensors to modular PyTorch components.

TheNoise owns the I/O and residency. This loader never materializes a full
44-GiB state dictionary or imports the Lightricks Python packages. Numerical
modules are native PyTorch; neither Diffusers nor ComfyUI is a runtime
dependency. ComfyUI is used only as a checkpoint-layout reference.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Mapping

import torch

from thenoise.dit.ltx_video.keys import component_for, split_ltx_key
from thenoise.dit.ltx_video.transformer import (
    AUDIO_HEAD_DIM, LTXParams, VIDEO_HEAD_DIM,
)
from thenoise.utils.loader import load_selected_state, partition_keys
from thenoise.utils.safetensors import as_header, require_divisible

log = logging.getLogger(__name__)

VERSION_23 = "2.3"
VERSION_25 = "2.5"

#: Prompt modulation adds three rows to an AdaLN table; six is a table without it.
_MOD_WITH_PROMPT = 9


def detect_params(checkpoint) -> LTXParams:
    """Read a transformer's geometry out of the checkpoint header.

    The shapes are indexed by *module* parameter through :func:`split_ltx_key`, so a key
    the checkpoint spells differently is renamed once, here and at load, rather than in
    two places that can drift. A file that is not an LTX transformer fails on the first
    name :meth:`Header.shape` cannot find.
    """
    header = as_header(checkpoint)
    shapes: dict[str, tuple[int, ...]] = {}
    for raw in header.names:
        part, target = split_ltx_key(raw)
        if part == "dit" and target is not None:
            shapes[target] = header.shape(raw)

    dim, in_channels = shapes["proj_in.weight"]
    audio_dim, audio_in = shapes["audio_proj_in.weight"]
    require_divisible(dim, VIDEO_HEAD_DIM, "video trunk", "head width")
    require_divisible(audio_dim, AUDIO_HEAD_DIM, "audio trunk", "head width")
    block = "transformer_blocks.0."
    return LTXParams(
        in_channels=in_channels,
        out_channels=shapes["proj_out.weight"][0],
        dim=dim,
        num_layers=1 + max(int(key.split(".")[1]) for key in shapes
                           if key.startswith("transformer_blocks.")),
        cross_attention_dim=shapes[block + "attn2.to_k.weight"][1],
        audio_in_channels=audio_in,
        audio_out_channels=shapes["audio_proj_out.weight"][0],
        audio_dim=audio_dim,
        audio_cross_attention_dim=shapes[block + "audio_attn2.to_k.weight"][1],
        ff_bias=block + "ff.net.0.proj.bias" in shapes,
        audio_ff_bias=block + "audio_ff.net.0.proj.bias" in shapes,
        gated_attn=block + "attn1.to_gate_logits.weight" in shapes,
        audio_gated_attn=block + "audio_attn1.to_gate_logits.weight" in shapes,
        cross_attn_mod=shapes["time_embed.linear.weight"][0] // dim == _MOD_WITH_PROMPT,
        audio_cross_attn_mod=(
            shapes["audio_time_embed.linear.weight"][0] // audio_dim == _MOD_WITH_PROMPT),
        keyframes="keyframes_abs_pos_embedding" in shapes,
    )


def make_transformer(params: LTXParams):
    """Build the transformer of these geometry numbers, on meta.

    What a checkpoint cannot state stays here: the RoPE base and causal offset, the
    timestep multipliers, and the grid the positional embedding was fitted to.
    """
    from thenoise.dit.ltx_video.transformer import LTX2VideoTransformer3DModel

    with torch.device("meta"):
        return LTX2VideoTransformer3DModel(
            in_channels=params.in_channels, out_channels=params.out_channels,
            num_attention_heads=params.heads, attention_head_dim=VIDEO_HEAD_DIM,
            cross_attention_dim=params.cross_attention_dim,
            audio_in_channels=params.audio_in_channels,
            audio_out_channels=params.audio_out_channels,
            audio_num_attention_heads=params.audio_heads,
            audio_attention_head_dim=AUDIO_HEAD_DIM,
            audio_cross_attention_dim=params.audio_cross_attention_dim,
            num_layers=params.num_layers, ff_bias=params.ff_bias,
            audio_ff_bias=params.audio_ff_bias, gated_attn=params.gated_attn,
            audio_gated_attn=params.audio_gated_attn,
            cross_attn_mod=params.cross_attn_mod,
            audio_cross_attn_mod=params.audio_cross_attn_mod,
            use_keyframes_abs_pos_embedding=params.keyframes,
        )


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

def make_upsampler():
    # Only the released spatial x2 model, using TheNoise's existing module seam.
    from thenoise.dit.ltx_video.upscaler import LTXSpatialUpscaler
    with torch.device("meta"):
        return LTXSpatialUpscaler(in_channels=128, mid_channels=1024,
                                 num_blocks_per_stage=4)


def make_connectors(gemma_text_config, params: LTXParams):
    """The text-projection pair, on meta.

    The Gemma decides the caption width and how many of its layers are read out -- both
    released Gemmas are 3840 wide and 48 deep, which is why no version belonged in that
    number. The connectors' depth and register count stay literals: those tensors can
    arrive from a file of their own, and no released pair differs on them.
    """
    from thenoise.dit.ltx_video.connectors import LTX2TextConnectors

    cfg = dict(
        caption_channels=gemma_text_config.hidden_size,
        text_proj_in_factor=gemma_text_config.num_hidden_layers + 1,
        video_connector_num_attention_heads=32,
        video_connector_attention_head_dim=128,
        video_connector_num_layers=8, video_connector_num_learnable_registers=128,
        video_gated_attn=True, audio_connector_num_attention_heads=32,
        audio_connector_attention_head_dim=64,
        audio_connector_num_layers=8, audio_connector_num_learnable_registers=128,
        audio_gated_attn=True, connector_rope_base_seq_len=4096,
        rope_theta=10000., rope_double_precision=True,
        causal_temporal_positioning=False, rope_type="split",
        per_modality_projections=True, video_hidden_dim=params.dim,
        audio_hidden_dim=params.audio_dim, proj_bias=True,
    )
    with torch.device("meta"):
        model = LTX2TextConnectors(**cfg)
        return model


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


def load_original_components(
    files: dict[str, str | Path], *, version: str,
    device_for: Callable[[list[Path]], object],
    dtype=torch.bfloat16, gemma_text_config=None, headers: Mapping[str, object] | None = None,
) -> tuple[dict, dict[str, Path]]:
    """Build every LTX component from the files a request named by role (``""`` is the
    transformer), and answer where each one came from.

    A part comes from the file named for it, else from whichever named file carries its
    tensors; a part two files both carry (the connector pair) is read from the file it
    belongs to, the others filling only what it leaves out. ``headers`` is
    the runtime's own parsed headers by role, so a checkpoint is opened once however many
    parts ask about it.

    ``device_for`` answers where the part arriving from the given files should live -- the
    memory manager's decision, made on the bytes of those files -- and the tensors are read
    there, so a 22 GB transformer does not pass through system RAM on its way to the card.
    """
    given = {role: Path(path).expanduser().resolve()
             for role, path in files.items() if path}
    primary = given.get("")
    if primary is None:
        raise ValueError("LTX needs its transformer: pass --dit")
    params = detect_params(_header("", primary, headers))
    modules = dict(
        dit=make_transformer(params), vae=make_video_vae(),
        audio_vae=make_audio_vae(), vocoder=make_vocoder(),
        upscaler=make_upsampler(),
    )
    if gemma_text_config is not None:
        modules["connectors"] = make_connectors(gemma_text_config, params)
    expected = {name: {key: tuple(t.shape) for key, t in module.state_dict().items()}
                for name, module in modules.items()}

    named_parts: dict[Path, tuple[str, ...]] = {}
    for role, path in given.items():
        named_parts[path] = named_parts.get(path, ()) + tuple(
            name for name in ROLE_OF_COMPONENT if ROLE_OF_COMPONENT[name] == role)

    picks: dict[str, dict[Path, dict[str, str]]] = {name: {} for name in modules}
    misses: dict[str, list[tuple[Path, str]]] = {name: [] for name in modules}
    for role, path in given.items():
        named = named_parts[path]
        chosen, rejected = partition_keys(
            _header(role, path, headers), expected,
            part_of=lambda raw: component_for(raw, transformer=(path == primary), named=named),
            strict=frozenset(named))
        for name in modules:
            if chosen[name]:
                picks[name][path] = chosen[name]
            misses[name] += [(path, raw) for raw in rejected[name]]

    sources: dict[str, list[Path]] = {}
    claims: dict[str, dict[Path, dict[str, str]]] = {}
    for name in modules:
        taken: set[str] = set()
        per_file: dict[Path, dict[str, str]] = {}
        for path in _carriers_for(name, given, list(picks[name])):
            selected = {raw: target for raw, target in picks[name][path].items()
                        if target not in taken}
            taken |= set(selected.values())
            if selected:
                per_file[path] = selected
        if not per_file:
            raise ValueError(
                f"LTX {name} weights are in none of the checkpoints given"
                + (f"; pass {_flag_for(name, version)}" if _flag_for(name, version) else "")
                + _near_miss(name, misses[name]))
        sources[name] = list(per_file)
        claims[name] = per_file
    log.info("LTX %s components: %s", version,
             ", ".join(f"{name} from {', '.join(p.name for p in paths)}"
                       for name, paths in sorted(sources.items())))

    for name, module in modules.items():
        load_selected_state(
            module, [(str(path), selected) for path, selected in claims[name].items()],
            dtype=dtype, device=device_for(sources[name]),
            what=f"LTX {name}")
        module.eval()
    return modules, {name: paths[0] for name, paths in sources.items()}


def _carriers_for(name: str, given: Mapping[str, Path],
                  carriers: list[Path]) -> list[Path]:
    """The files to read one part from, most authoritative first.

    The connector pair is the one part two released files both carry: 2.5 ships it with
    the text encoder, 2.3 as its own file, and a repackaged transformer holds a copy of
    whichever it was made from. Two files holding the same pair is not a mistake, so the
    file the request named for the part decides and the rest only supply what it left
    out -- which is also how a pair legitimately split across two files still completes.

    Every other part is answered by exactly one file.
    """
    wanted = given.get(ROLE_OF_COMPONENT[name])
    if name != "connectors":
        return [wanted] if wanted in carriers else carriers[:1]
    own = [given[role] for role in ("text_projection", "text_encoder")
           if given.get(role) in carriers]
    return own + [path for path in carriers if path not in own]


def _header(role: str, path: Path, headers: Mapping[str, object] | None):
    """The parsed header for a file the request named under ``role``."""
    from thenoise.utils.safetensors import read_header

    given = (headers or {}).get(role)
    return given if given is not None else read_header(str(path))


def _near_miss(name: str, misses: list[tuple[Path, str]]) -> str:
    """What to add when a part was not found: files that spelled it without fitting it.

    A checkpoint of another generation answers to the same prefixes, and "pass --vae"
    alone sends a request looking for a file it already has.
    """
    if not misses:
        return ""
    path, raw = misses[0]
    return (f" ({path.name} spells {len(misses)} tensors as {name} that this release has "
            f"no parameter for, e.g. {raw!r})")
