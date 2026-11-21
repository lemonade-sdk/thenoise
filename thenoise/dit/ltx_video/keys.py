"""Checkpoint names for LTX 2.x: what each released spelling calls each module.

The same networks arrive under at least three layouts -- the all-in-one that packs the
transformer beside both codecs and the vocoder, the split files that name nothing because
they hold one part each, and ComfyUI's own names for the transformer, which the Apache-2
reference converter translates. So the answer to "which part does this tensor spell, and
which parameter does it fill there" lives here rather than in a branch of the loader, and a
fourth spelling belongs in these tables rather than in another one.

Names are checkpoint spellings on the way in and module parameters on the way out, so a
rename is written once here and everything downstream -- geometry read off the shapes, the
strict assignment of the weights -- agrees on the result.
"""
from __future__ import annotations

from thenoise.utils.qk_norm import qk_norm_key_map
from thenoise.utils.safetensors import Signature, unwrap_key

#: Both streams' patch projections are what makes a released file an LTX 2.x transformer.
#: 2.5 adds one embedding that 2.3 cannot have, and each version asks for the other's
#: answer, so no file is claimed by the version that would build it wrong. Checkpoint
#: names, not module names: see :func:`split_ltx_key`.
_AV_KEYS = ("patchify_proj.weight", "audio_patchify_proj.weight")
_KEYFRAME_EMBEDDING = "keyframes_abs_pos_embedding"
SIGNATURE_23 = Signature("LTX-2.3 transformer", keys=_AV_KEYS,
                         absent=(_KEYFRAME_EMBEDDING,))
SIGNATURE_25 = Signature("LTX-2.5 transformer",
                         keys=(*_AV_KEYS, _KEYFRAME_EMBEDDING))

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


def _qk_layout(k: str) -> str:
    """The two spellings the releases use for a QK norm, onto the shared module's names."""
    for legacy in (("q_norm", "k_norm"), ("norm_q", "norm_k")):
        k = qk_norm_key_map(k, *legacy)
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
        ):
            k = k.replace(a, b)
        return "connectors", _qk_layout(k)
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
    return "dit", _qk_layout(k)

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

def map_component(comp: str, k: str) -> str | None:
    if comp == "vae":
        return _remap_vae(k)
    if comp == "audio_vae":
        return (k.replace("per_channel_statistics.mean-of-means", "latents_mean")
                 .replace("per_channel_statistics.std-of-means", "latents_std"))
    if comp == "vocoder":
        return _remap_vocoder(k)
    return k

#: The parts a file can name by prefix, alone or inside an all-in-one checkpoint.
COMPONENT_PREFIXES = ("vae", "audio_vae", "vocoder")


def component_for(raw: str, *, transformer: bool = False,
            named: tuple[str, ...] = ()) -> tuple[str | None, str | None]:
    """Which component a checkpoint tensor spells, and the parameter it fills there.

    A file holding the transformer spells every key as :func:`split_ltx_key` reads it,
    prefixes included, and so does the connector pair wherever it is packed. A file
    holding parts of its own names them by prefix, and a key with no prefix belongs to
    ``named`` -- the parts this file was given for, whose only multi-part case (the audio
    codec beside its vocoder) spells both by prefix. A file named for nothing, a text
    encoder, answers for nothing.
    """
    key = unwrap_key(raw)
    if transformer or _strip_dit_prefix(key).startswith(_CONNECTOR_PREFIXES):
        component, target = split_ltx_key(key) or ("", None)
        if target is not None and component in COMPONENT_PREFIXES:
            # A codec packed with the transformer is spelled the original way: the prefix
            # comes off here, the codec's own map renames it.
            target = map_component(component, target)
        return component, target
    head, _, rest = key.partition(".")
    if head in COMPONENT_PREFIXES:
        return head, map_component(head, rest)
    if not named:
        return None, None
    return named[0], map_component(named[0], key)

