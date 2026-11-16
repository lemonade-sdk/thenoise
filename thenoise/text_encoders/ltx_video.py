"""Gemma 3/4 prompt features for LTX 2.x; no LTX Python package imports.

The single-file LTX-2.5 encoder stores its HF config, tokenizer and
processor assets alongside the weights. Do not silently substitute an
ordinary Gemma-4 checkpoint: the LTX projection weights are model-specific.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from thenoise.utils.safetensors import MemoryEfficientSafeOpen
from torch import nn

from thenoise.utils.loader import load_streamed_components


class GemmaFeatures(nn.Module):
    """Module wrapper so the standard TheNoise MemoryManager owns residency."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Calling the inner model avoids the 250K-token LM-head logits, exactly
        # as the official Lightricks text-encoder implementation does.
        inner = getattr(self.model, "model", self.model)
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


def load_ltx_gemma(path: str | Path, *, version: str, cancel):
    """Return `(GemmaFeatures, tokenizer, text_config)` on CPU.

    All construction of a single-file Gemma-4 is on meta and weights are read
    one at a time. A Gemma-3 HF directory uses Transformers' indexed shard
    loader; no HF network access, git URL, or second LTX SDK is required.
    """
    try:
        from transformers import (AutoModelForImageTextToText, AutoTokenizer,
                                  PreTrainedTokenizerFast)
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except ImportError as e:
        raise RuntimeError("LTX text encoding requires transformers>=5.4") from e

    path = Path(path).expanduser()
    cancel()
    if path.is_dir():
        config_file = path / "config.json"
        if not config_file.is_file():
            raise FileNotFoundError(
                f"Missing LTX {version} text-encoder config: {config_file}. "
                "For LTX 2.3 INT8 run: python scripts/download.py --model ltx23 "
                "(downloads only missing assets; already present weights are skipped)."
            )
        config = json.loads(config_file.read_text())
        if (version == "2.3" and config.get("model_type") != "gemma3") or (
            version == "2.5" and config.get("model_type") != "gemma4_unified"
        ):
            raise ValueError(f"Wrong Gemma model type for LTX {version}: {config.get('model_type')!r}")
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        quant_path = next(
            (path / name for name in (
                "gemma_3_12B_it-int8-convrot-simple.safetensors",
                "gemma_3_12B_it_int8_convrot.safetensors",
            ) if (path / name).is_file()),
            None,
        )
        if version == "2.3" and quant_path is not None:
            # The Comfy pack contains the Gemma text model only. The tokenizer
            # and the config live next to it; we never download BF16 shards.
            from transformers import AutoConfig, Gemma3TextModel
            from thenoise.dit.quantized import replace_linears
            hf_config = AutoConfig.from_pretrained(str(path), local_files_only=True)
            with torch.device("meta"):
                text = Gemma3TextModel(hf_config.text_config)
                replace_linears(text)
            _materialize_gemma3_constants(text)
            wrapper = GemmaFeatures(text)
            with MemoryEfficientSafeOpen(str(quant_path)) as f:
                load_streamed_components(f, {"gemma": wrapper}, cancel=cancel,
                        selector=_gemma3_key, dtype=torch.bfloat16)
            return wrapper.eval(), tokenizer, hf_config.text_config
        if not (path / "model.safetensors.index.json").exists() and not (
                path / "model.safetensors").exists():
            raise FileNotFoundError(
                f"No Gemma {version} weights found in {path}: expected INT8 ConvRot "
                "file or a complete BF16 HF model directory")
        # BF16 remains supported explicitly; no `device_map=auto` bypass.
        model = AutoModelForImageTextToText.from_pretrained(
            str(path), dtype=torch.bfloat16, low_cpu_mem_usage=True,
            local_files_only=True,
        )
        return GemmaFeatures(model.eval()), tokenizer, model.config.text_config

    if version != "2.5":
        raise ValueError("LTX-2.3 expects an original Gemma-3 HF directory")
    if path.suffix != ".safetensors":
        raise ValueError("LTX-2.5 Gemma-4 must be a packed .safetensors file or a HF directory")
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
        load_streamed_components(
            f, {"gemma": wrapper}, cancel=cancel,
            selector=lambda raw: _gemma4_key(raw, expected),
            dtype=torch.bfloat16,
        )
    return wrapper.eval(), tokenizer, hf_config.text_config
