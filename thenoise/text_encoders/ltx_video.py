"""Gemma 3/4 prompt features for LTX 2.x; no LTX Python package imports.

Both Gemma 3 and Gemma 4 encoders are single-file safetensors. Gemma 3
uses vendored minimal tokenizer/config assets; Gemma 4 embeds those assets
with its model-specific LTX projection weights.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path

import torch
from thenoise.utils.safetensors import MemoryEfficientSafeOpen
from torch import nn

from thenoise.utils.loader import load_text_encoder_weights





def _use_math_gemma_attention(device: torch.device) -> bool:
    """Avoid broken masked Gemma SDPA on confirmed ROCm GPU architectures.

    Selection is scoped to Gemma inference, never global or DiT-wide.
    `THENOISE_LTX_GEMMA_SDPA=math|default` permits A/B validation of an
    affected driver/PyTorch combination without changing installed packages.
    """
    mode = os.environ.get("THENOISE_LTX_GEMMA_SDPA", "auto").strip().lower()
    if mode not in ("auto", "math", "default"):
        raise ValueError("THENOISE_LTX_GEMMA_SDPA must be auto, math or default")
    if mode != "auto":
        return mode == "math"
    if not torch.version.hip or device.type != "cuda":
        return False
    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
    except (AttributeError, RuntimeError):
        return False
    return arch in ("gfx1100", "gfx1151")


class GemmaFeatures(nn.Module):
    """Module wrapper so the standard TheNoise MemoryManager owns residency."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Calling the inner model avoids the 250K-token LM-head logits, exactly
        # as the official Lightricks text-encoder implementation does.
        inner = getattr(self.model, "model", self.model)
        if _use_math_gemma_attention(ids.device):
            from torch.nn.attention import SDPBackend, sdpa_kernel
            attention_context = sdpa_kernel([SDPBackend.MATH])
        else:
            attention_context = nullcontext()
        with attention_context:
            features = inner(
                input_ids=ids, attention_mask=mask, output_hidden_states=True,
            ).hidden_states
        if features is None:
            raise RuntimeError("Gemma text model returned no hidden states")
        return torch.stack(features, dim=-1).flatten(2, 3)

def _read_asset(source, meta: dict, name: str) -> bytes:
    key = f"hf_asset__{name}"
    if key in source.keys():
        tensor = source.get_tensor(key).to(torch.uint8).cpu().numpy()
        return tensor.tobytes()
    if name in meta:
        return meta[name].encode()
    raise ValueError(f"LTX packed Gemma checkpoint lacks required {name!r}")


def _gemma4_key(raw: str, expected: set[str] | None = None) -> tuple[str, str] | None:
    """Translate ComfyUI's Gemma-4 names to the installed HF model layout.

    The official Gemma-4 model has used two internal multimodal layouts:
    the unified embed_vision module (with a nested multimodal_embedder) and
    separate vision_tower/embed_vision modules. The LTX checkpoint carries
    ComfyUI names, so select only a destination present in the constructed
    model. Never silently discard an unmatched learned tensor.
    """
    if raw == "tokenizer_json" or raw.startswith("hf_asset__"):
        return None
    if raw.startswith(("video_embeddings_connector.", "audio_embeddings_connector.",
                       "text_embedding_projection.", "connectors.")):
        return None  # The LTX text connectors are a separate component.
    if raw.startswith(("lm_head.", "model.lm_head.")):
        return None  # Generation logits are unused; the LM head is not built.

    def candidates(prefixes: tuple[str, ...], tail: str) -> list[str]:
        return ["model.model." + prefix + tail for prefix in prefixes]

    if raw.startswith("model.language_model."):
        names = candidates(("language_model.",), raw[len("model.language_model."):])
    elif raw.startswith("language_model."):
        names = candidates(("language_model.",), raw[len("language_model."):])
    elif raw.startswith("model.vision_embedder."):
        names = candidates(("embed_vision.", "vision_tower.", "vision_embedder."),
                           raw[len("model.vision_embedder."):])
    elif raw.startswith("vision_model."):
        names = candidates(("embed_vision.", "vision_tower.", "vision_embedder."),
                           raw[len("vision_model."):])
    elif raw.startswith("vision_tower."):
        names = candidates(("vision_tower.", "embed_vision."), raw[len("vision_tower."):])
    elif raw.startswith("model.embed_vision."):
        tail = raw[len("model.embed_vision."):]
        names = candidates(("embed_vision.multimodal_embedder.", "embed_vision.",
                            "multi_modal_projector."), tail)
    elif raw.startswith("multi_modal_projector."):
        tail = raw[len("multi_modal_projector."):]
        names = candidates(("embed_vision.multimodal_embedder.", "embed_vision.",
                            "multi_modal_projector."), tail)
    elif raw.startswith("audio_model."):
        names = candidates(("audio_tower.", "embed_audio.", "audio_model."),
                           raw[len("audio_model."):])
    elif raw.startswith("model.audio_embedder."):
        names = candidates(("audio_tower.", "embed_audio.", "audio_model."),
                           raw[len("model.audio_embedder."):])
    elif raw.startswith("model.embed_audio."):
        names = candidates(("embed_audio.multimodal_embedder.", "embed_audio.",
                            "audio_projector."), raw[len("model.embed_audio."):])
    elif raw.startswith("audio_projector."):
        names = candidates(("embed_audio.multimodal_embedder.", "embed_audio.",
                            "audio_projector."), raw[len("audio_projector."):])
    elif raw.startswith("model."):
        names = candidates(("language_model.",), raw[len("model."):])
    else:
        return None

    if expected is not None:
        matches = [name for name in names if name in expected]
        if len(matches) > 1:
            raise ValueError(f"ambiguous Gemma-4 checkpoint key {raw!r}: {matches}")
        if matches:
            return "gemma", matches[0]
    # Returning the primary candidate retains strict _assign validation: an
    # unknown parameter becomes an explicit 'unexpected', not a silently skipped key.
    return "gemma", names[0]


def _drop_gemma_lm_head(model: nn.Module) -> None:
    """LTX reads hidden states, never vocab logits (~2 GiB for Gemma-4).

    Comfy's LTX text checkpoint deliberately contains no lm_head; HF's
    conditional-generation wrapper creates one. Drop it before strict loading,
    without suppressing any required text/vision/audio weights.
    """
    if "lm_head" in model._modules:
        del model.lm_head


def _materialize_gemma4_constants(model: nn.Module) -> None:
    """Rebuild Gemma-4's deterministic, non-persistent buffers outside meta.

    LTX checkpoints never contain the scaled embedding constant or the RoPE
    inverse frequencies. HF creates them on meta together with the huge learned
    weights; unlike the learned weights, the loader cannot assign these from the
    checkpoint. Only reconstruct the original constants from HF's own config and
    constructors. Never replace learned parameters or relax the strict load.
    """
    language = model.model.language_model
    for attr in ("embed_tokens", "embed_tokens_per_layer"):
        embedding = getattr(language, attr, None)
        if embedding is None:
            continue
        scale = embedding._buffers.get("embed_scale")
        if scale is None or not scale.is_meta:
            continue
        if "embed_scale" not in embedding._non_persistent_buffers_set:
            raise RuntimeError(f"Gemma-4 {attr}.embed_scale is unexpectedly persistent")
        original = getattr(embedding, "scalar_embed_scale", None)
        if original is None or scale.numel() != 1:
            raise RuntimeError(f"Gemma-4 {attr}.embed_scale cannot be reconstructed")
        embedding.register_buffer(
            "embed_scale", torch.tensor(original, device="cpu", dtype=scale.dtype).reshape(scale.shape),
            persistent=False,
        )

    rotary = language.rotary_emb
    if not any(buffer.is_meta for buffer in rotary.buffers()):
        return
    # RoPE has no learned state. Its constructor alone recreates its frequency
    # vectors, including the separate sliding/full attention tables and any
    # version-specific rope scaling. Instantiating it costs only small CPU buffers.
    if list(rotary.parameters()) or rotary.state_dict():
        raise RuntimeError("Gemma-4 rotary_emb unexpectedly has checkpoint state")
    with torch.device("cpu"):
        ready = type(rotary)(rotary.config, device=torch.device("cpu"))
    if (set(ready._buffers) != set(rotary._buffers)
            or ready._non_persistent_buffers_set != rotary._non_persistent_buffers_set
            or list(ready.parameters()) or ready.state_dict()
            or any(buffer.is_meta for buffer in ready.buffers())):
        raise RuntimeError("Gemma-4 RoPE runtime buffers could not be reconstructed")
    language.rotary_emb = ready


def _materialize_gemma3_constants(model: nn.Module) -> None:
    """Initialize only deterministic Gemma-3 runtime buffers created on meta.

    The INT8 checkpoint supplies learned parameters, not the nonpersistent
    embedding scale or RoPE inverse frequencies. Keep _assign() strict for
    all learned weights instead of tolerating uninitialized tensors.
    """
    embedding = model.embed_tokens
    scale = embedding._buffers.get("embed_scale")
    if scale is not None and scale.is_meta:
        if "embed_scale" not in embedding._non_persistent_buffers_set:
            raise RuntimeError("Gemma-3 embed_scale is unexpectedly persistent")
        value = getattr(embedding, "scalar_embed_scale", None)
        if value is None or scale.numel() != 1:
            raise RuntimeError("Gemma-3 embed_scale cannot be reconstructed")
        embedding.register_buffer(
            "embed_scale", torch.tensor(value, dtype=scale.dtype, device="cpu").reshape(scale.shape),
            persistent=False,
        )

    rotary = model.rotary_emb
    if not any(buffer.is_meta for buffer in rotary.buffers()):
        return
    if list(rotary.parameters()) or rotary.state_dict():
        raise RuntimeError("Gemma-3 rotary_emb unexpectedly contains checkpoint state")
    with torch.device("cpu"):
        ready = type(rotary)(rotary.config)
    if (set(ready._buffers) != set(rotary._buffers)
            or ready._non_persistent_buffers_set != rotary._non_persistent_buffers_set
            or list(ready.parameters()) or ready.state_dict()
            or any(buffer.is_meta for buffer in ready.buffers())):
        raise RuntimeError("Gemma-3 RoPE runtime buffers could not be reconstructed")
    model.rotary_emb = ready


def _gemma3_key(raw: str) -> tuple[str, str] | None:
    """Map Comfy Gemma-3 text-only layers to HF Gemma3TextModel keys."""
    if raw in ("spiece_model", "tokenizer_json") or raw.startswith("hf_asset__"):
        return None
    key = raw
    # Community Comfy checkpoints vary in whether the outer CLIP key and
    # Gemma's own `model.` wrapper are kept. Strip both, not just the first.
    prefixes = ("gemma3_12b.", "text_encoder.", "model.language_model.",
                "language_model.", "model.")
    for _ in range(4):
        matched = next((prefix for prefix in prefixes if key.startswith(prefix)), None)
        if matched is None:
            break
        key = key[len(matched):]
    if not key.startswith(("layers.", "embed_tokens.", "norm.", "rotary_emb.")):
        return None
    return "gemma", "model." + key


def _load_ltx_text_weights(module, path: Path, key_map, drop_keys: tuple[str, ...]):
    """Use TheNoise's shared Gemma/quantized text-encoder loader.

    Only checkpoint-specific name mapping lives here. Tokenizer/config blobs are
    assets, not model tensors. Learned unknown keys must fail, not be dropped.
    """
    def mapped(raw):
        result = key_map(raw)
        if result is None:
            raise ValueError(f"unmapped Gemma weight in {path}: {raw!r}")
        return result[1]

    return load_text_encoder_weights(
        module, str(path), device="cpu", dtype=torch.bfloat16,
        key_map=mapped, drop_keys=drop_keys,
    )





def load_ltx_gemma(path: str | Path, *, version: str):
    """Return `(GemmaFeatures, tokenizer, text_config)` on CPU.

    Model weights come from one safetensors file. Gemma3 uses local
    tokenizer.json and a vendored config; Gemma4 embeds its assets.
    """
    try:
        from transformers import AutoModelForImageTextToText, PreTrainedTokenizerFast
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except ImportError as e:
        raise RuntimeError("LTX text encoding requires transformers>=5.4") from e

    path = Path(path).expanduser()
    if not path.is_file() or path.suffix != ".safetensors":
        raise ValueError(f"LTX {version} text encoder requires one existing .safetensors file: {path}")
    if version == "2.3":
        from transformers import Gemma3TextConfig, Gemma3TextModel, GemmaTokenizer
        from thenoise.dit.quantized import replace_linears
        from thenoise.utils.text_encoder import GEMMA3_CONFIG_DIR

        vendor = Path(GEMMA3_CONFIG_DIR)
        config = Gemma3TextConfig(**json.loads(
            (vendor / "text_config.json").read_text(encoding="utf-8")
        ))
        tokenizer_path = vendor / "tokenizer.json"
        # Use the bundled full fast tokenizer JSON, not a five-token fallback:
        tokenizer = GemmaTokenizer(
            tokenizer_file=str(tokenizer_path),
            add_bos_token=True, add_eos_token=False,
        )
        if tokenizer.vocab_size < 250_000 or len(
            tokenizer("A red fox runs through a snowy forest.").input_ids
        ) < 6:
            raise RuntimeError(
                f"LTX Gemma-3 tokenizer is incomplete: {tokenizer_path}; "
                "expected a full Gemma-3 tokenizer.json"
            )
        with torch.device("meta"):
            model = Gemma3TextModel(config)
            replace_linears(model)
        _materialize_gemma3_constants(model)
        wrapper = GemmaFeatures(model)
        _load_ltx_text_weights(
            wrapper, path, _gemma3_key,
            ("spiece_model", "gemma3_12b.spiece_model", "tokenizer_json",
             "gemma3_12b.tokenizer_json", "hf_asset__", "multi_modal_projector.",
             "model.multi_modal_projector.", "gemma3_12b.multi_modal_projector.",
             "vision_tower.", "model.vision_tower.", "gemma3_12b.vision_tower.",
             "vision_model.", "model.vision_model.", "gemma3_12b.vision_model."),
        )
        return wrapper.eval(), tokenizer, config

    if version != "2.5":
        raise ValueError(f"unsupported LTX text encoder version: {version!r}")
    if path.suffix != ".safetensors":
        raise ValueError("LTX-2.5 Gemma-4 must be a packed .safetensors file")
    from tokenizers import Tokenizer

    with MemoryEfficientSafeOpen(str(path)) as f:
        meta = f.metadata() or {}
        raw_config = meta.get("gemma_config")
        if not raw_config:
            raise ValueError("LTX-2.5 Gemma file lacks gemma_config metadata")
        config_data = json.loads(raw_config)
        if config_data.get("model_type") != "gemma4_unified":
            raise ValueError("LTX-2.5 requires gemma4_unified, not stock Gemma-3")
        hf_config = CONFIG_MAPPING["gemma4_unified"].from_dict(config_data)
        tokenizer_data = f.get_tensor("tokenizer_json").to(torch.uint8).numpy().tobytes()
        tokenizer_config = json.loads(_read_asset(f, meta, "tokenizer_config.json"))
        tokenizer_config = {k:v for k,v in tokenizer_config.items()
                            if k not in ("tokenizer_class", "auto_map", "backend", "is_local",
                                         "model_max_length", "processor_class", "added_tokens_decoder")}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_buffer(tokenizer_data),
            model_max_length=1024, **tokenizer_config,
        )
        # Construct without a CPU-initialized 24-GiB copy.
        from thenoise.dit.quantized import replace_linears
        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(hf_config)
            _drop_gemma_lm_head(model)
            replace_linears(model)
        _materialize_gemma4_constants(model)
        wrapper = GemmaFeatures(model)
        # The expected names are the actual loaded Transformers version's
        # layout, not guesses based on an HF release. Every other learned
        # tensor must match exactly before any heavy data is read.
        expected = set(wrapper.state_dict())
        # The model file is closed before the common loader opens it for
        # actual weight assignment. Only tokenizer/config parsing uses the
        # model-specific asset reader above.
    _load_ltx_text_weights(
        wrapper, path, lambda raw: _gemma4_key(raw, expected),
        ("tokenizer_json", "hf_asset__", "text_embedding_projection.",
         "connectors.", "video_embeddings_connector.", "audio_embeddings_connector.",
         "video_connector.", "audio_connector.", "lm_head.", "model.lm_head."),
    )
    return wrapper.eval(), tokenizer, hf_config.text_config
