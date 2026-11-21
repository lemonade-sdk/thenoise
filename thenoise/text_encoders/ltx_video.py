"""Gemma 3/4 prompt features for LTX 2.x, without the LTX Python package.

Each encoder is one safetensors file, given to `--text-encoder`: the 22B transformer
carries no Gemma weights, no tokenizer and no text config. Gemma 3 keeps its
SentencePiece proto beside its weights and its architecture is spelled out below,
because the ComfyUI single-file Gemma 3 checkpoints have no metadata at all. Gemma 4
packs a tokenizer JSON, a tokenizer config and a `gemma_config` metadata entry.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import sentencepiece as spm
import torch
from accelerate import init_empty_weights
from torch import nn

from thenoise.utils.loader import load_text_encoder_weights, spiece_model
from thenoise.utils.safetensors import MemoryEfficientSafeOpen

#: The prompt context the LTX text connectors were released with: eight blocks of their
#: 128 learned registers, every position of which becomes DiT cross-attention context.
#: Truncation is against it, since the width the ids finally occupy is this and not the
#: prompt's length -- but Gemma itself runs on the prompt's length alone.
PROMPT_CONTEXT = 1024

#: Fields of Gemma 4's packed tokenizer config that describe the file rather than the
#: tokens: the class and processor the packer used, where it was loaded from, the
#: sentinel length Transformers records when nobody chose one, and the special-token
#: table the tokenizer buffer beside it already carries. What is left -- `bos_token`,
#: `eos_token`, the chat markers -- is what a tokenizer is actually asked for.
_TOKENIZER_FILE_FIELDS = ("auto_map", "backend", "is_local", "model_max_length",
                          "processor_class", "tokenizer_class", "added_tokens_decoder")

#: Gemma 3 12B as the LTX 2.3 text encoder, reproducing the `config.json` of the
#: upstream model: the released single-file checkpoints ship weights and a tokenizer
#: and no architecture description, so this is what the encoder is built from.
#: `vocab_size` exceeds the tokenizer's 262144 pieces because the embedding table is
#: padded to a round 262208 rows.
GEMMA3_TEXT_CONFIG = {
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_logit_softcapping": None,
    "cache_implementation": "hybrid",
    "final_logit_softcapping": None,
    "head_dim": 256,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 3840,
    "initializer_range": 0.02,
    "intermediate_size": 15360,
    "max_position_embeddings": 131072,
    "model_type": "gemma3_text",
    "num_attention_heads": 16,
    "num_hidden_layers": 48,
    "num_key_value_heads": 8,
    "query_pre_attn_scalar": 256,
    "rms_norm_eps": 1e-06,
    "rope_local_base_freq": 10000,
    "rope_scaling": {"factor": 8.0, "rope_type": "linear"},
    "rope_theta": 1000000,
    "sliding_window": 1024,
    "sliding_window_pattern": 6,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 262208,
}





class GemmaFeatures(nn.Module):
    """Module wrapper so the standard TheNoise MemoryManager owns residency."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Prompt ids at their own length -> every layer's features, concatenated.

        The inner model is called directly, so the vocab logits are never formed. No
        attention mask goes in: a prompt arrives alone, so there is nothing to mask out,
        and a dense row is both cheaper and free of the masked kernels that the padded
        form needed.
        """
        inner = getattr(self.model, "model", self.model)
        features = inner(input_ids=ids, output_hidden_states=True).hidden_states
        if features is None:
            raise RuntimeError("Gemma text model returned no hidden states")
        return torch.stack(features, dim=-1).flatten(2, 3)


class GemmaPrompter:
    """Prompt text to ids, at the prompt's own length, BOS first.

    The two generations tokenize with different libraries -- Gemma 3 with the
    SentencePiece proto in its file, Gemma 4 with the packed tokenizer JSON -- and both
    end up here, since what the connectors need is the same: the prompt's ids, truncated
    to the context they will occupy, and a leading BOS. Neither tokenizer supplies one
    on its own, and a prompt without it conditions on the wrong token.
    """

    def __init__(self, ids_of: Callable[[str], Sequence[int]], bos: Optional[int]):
        # SentencePiece spells a tokenizer without one as -1 rather than None.
        if bos is None or int(bos) < 0:
            raise ValueError("LTX Gemma tokenizer has no BOS token")
        self._ids_of = ids_of
        self.bos_token_id = int(bos)

    @classmethod
    def from_sentencepiece(cls, processor) -> "GemmaPrompter":
        return cls(lambda text: processor.encode(text, out_type=int), processor.bos_id())

    @classmethod
    def from_transformers(cls, tokenizer) -> "GemmaPrompter":
        return cls(lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
                   tokenizer.bos_token_id)

    def encode(self, text: str) -> torch.Tensor:
        ids = list(self._ids_of(text.strip()))[:PROMPT_CONTEXT]
        if not ids or ids[0] != self.bos_token_id:
            ids = [self.bos_token_id, *ids][:PROMPT_CONTEXT]
        return torch.tensor([ids], dtype=torch.long)

def _read_asset(source, meta: dict, name: str) -> bytes:
    """An asset packed into a Gemma file: JSON the packer stored as bytes, or, in files
    that carry no `hf_asset__` prefix, as metadata text.

    The packer stores the tokenizer bare and the smaller configs prefixed, which is why
    both spellings are tried before anything is called missing.
    """
    for key in (f"hf_asset__{name}", name):
        if key in source.keys():
            return source.get_tensor(key).numpy().tobytes()
    if name in meta:
        return meta[name].encode()
    raise ValueError(f"LTX packed Gemma checkpoint lacks required {name!r}")


#: The 2.5 text encoder's names for its own modules, and where the model built from that
#: file's `gemma_config` metadata keeps them. ComfyUI leaves the language stack at
#: `model.`, the image tower at `vision_model.`/`multi_modal_projector.` and audio at
#: `audio_projector.`; Gemma 4 nests those one level deeper.
#:
#: All three released 2.5 text-encoder files spell their keys identically -- the official
#: BF16 one and the BF16 and INT8 ConvRot builds `scripts/download.py` installs -- and
#: these destinations are what each file's own metadata builds: all 677 parameters of the
#: wrapped encoder filled, with nothing left over but the packed assets and the LTX
#: projection weights, which are a separate component. A Transformers rename therefore
#: refuses the load, which is when a name is added here.
_GEMMA4_LAYOUT = {
    "model.": "language_model.",
    "vision_model.": "embed_vision.",
    "multi_modal_projector.": "embed_vision.multimodal_embedder.",
    "audio_projector.": "embed_audio.",
}


def _gemma4_key(raw: str) -> tuple[str, str] | None:
    """Translate a 2.5 text-encoder key into the installed model's names.

    None means the key is not this module's: the packed tokenizer assets and the LTX
    text connectors. A *learned* key that matches none of these prefixes also comes back
    None, and the loader turns that into an error -- an unmapped tensor is refused, never
    skipped.
    """
    if raw == "tokenizer_json" or raw.startswith("hf_asset__"):
        return None
    if raw.startswith(("video_embeddings_connector.", "audio_embeddings_connector.",
                       "text_embedding_projection.", "connectors.")):
        return None  # The LTX text connectors are a separate component.
    if raw.startswith(("lm_head.", "model.lm_head.")):
        return None  # Generation logits are unused; the LM head is not built.
    for source, destination in _GEMMA4_LAYOUT.items():
        if raw.startswith(source):
            return "gemma", "model.model." + destination + raw[len(source):]
    return None


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


def _load_ltx_text_weights(module, path: Path, key_map,
                           drop_keys: tuple[str, ...], device: str):
    """Use TheNoise's shared Gemma/quantized text-encoder loader.

    Only checkpoint-specific name mapping lives here. Tokenizer/config blobs are
    assets, not model tensors. Learned unknown keys must fail, not be dropped.

    ``device`` is the caller's because it is the residency manager's: this file is
    15-26 GB, and reading it onto CPU to be carried back afterwards holds all of it in
    host RAM while the transformer is being read too.
    """
    def mapped(raw):
        result = key_map(raw)
        if result is None:
            raise ValueError(f"unmapped Gemma weight in {path}: {raw!r}")
        return result[1]

    return load_text_encoder_weights(
        module, str(path), device=device, dtype=torch.bfloat16,
        key_map=mapped, drop_keys=drop_keys,
    )





#: Gemma 4 widens every sixth attention to ``head_dim`` 512, and as of ROCm 10.1.0rc2 the
#: fused SDPA backends answer that width with NaN -- over different tokens, layers and
#: sequence lengths on every run, so the encoder loses a few rows of a short prompt or
#: all of it. 256 measures clean on the same kernels, which is why Gemma 3 keeps them.
#: The reference implementation costs nothing here: one call per prompt, over at most the
#: connectors' 1024 rows.
GEMMA4_ATTENTION = "eager"


def build_gemma4(hf_config) -> "GemmaFeatures":
    """The Gemma 4 body of an LTX-2.5 encoder, empty and ready for quantized weights.

    Built through ``AutoModelForImageTextToText`` because that is how the unified
    checkpoint is registered. The vocab logits are removed: the connectors read hidden
    states, and the projection costs about 2 GiB to never be run.
    """
    from transformers import AutoModelForImageTextToText

    from thenoise.dit.quantized import replace_linears

    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(
            hf_config, attn_implementation=GEMMA4_ATTENTION)
        del model.lm_head
        replace_linears(model)
    return GemmaFeatures(model)


def load_ltx_gemma(path: str | Path, *, version: str, device: str):
    """Return `(GemmaFeatures, prompter, text_config)` built on ``device``.

    Weights come from one safetensors file: Gemma 3 from a vendored tokenizer and
    config, Gemma 4 from assets embedded in that same file. They are written straight
    to ``device``, and the finished module is moved there as well: these builds blank
    the parameters and leave ``transformers``' own buffers — the rotary frequencies,
    ``embed_scale`` — behind on CPU, and a moved module sweeps them without touching a
    parameter that is already where it belongs.
    """
    try:
        from transformers import PreTrainedTokenizerFast
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    except ImportError as e:
        raise RuntimeError("LTX text encoding requires transformers>=5.4") from e

    path = Path(path).expanduser()
    if not path.is_file() or path.suffix != ".safetensors":
        raise ValueError(f"LTX {version} text encoder requires one existing .safetensors file: {path}")
    if version == "2.3":
        from transformers import Gemma3TextConfig, Gemma3TextModel
        from thenoise.dit.quantized import replace_linears

        config = Gemma3TextConfig(**GEMMA3_TEXT_CONFIG)
        prompter = GemmaPrompter.from_sentencepiece(
            spm.SentencePieceProcessor(model_proto=spiece_model(path)))
        with init_empty_weights():
            model = Gemma3TextModel(config)
            replace_linears(model)
        wrapper = GemmaFeatures(model)
        _load_ltx_text_weights(
            wrapper, path, _gemma3_key,
            ("spiece_model", "gemma3_12b.spiece_model", "tokenizer_json",
             "gemma3_12b.tokenizer_json", "hf_asset__", "multi_modal_projector.",
             "model.multi_modal_projector.", "gemma3_12b.multi_modal_projector.",
             "vision_tower.", "model.vision_tower.", "gemma3_12b.vision_tower.",
             "vision_model.", "model.vision_model.", "gemma3_12b.vision_model."),
            device,
        )
        return wrapper.to(device).eval(), prompter, config

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
        # The packed tokenizer's post-processor adds nothing at all, so the BOS the
        # connectors were trained with comes from the tokenizer config beside it.
        hf_config = CONFIG_MAPPING["gemma4_unified"].from_dict(config_data)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_buffer(_read_asset(f, meta, "tokenizer_json")),
            **{k: v for k, v in json.loads(_read_asset(f, meta, "tokenizer_config.json")).items()
               if k not in _TOKENIZER_FILE_FIELDS},
        )
    wrapper = build_gemma4(hf_config)
    _load_ltx_text_weights(
        wrapper, path, _gemma4_key,
        ("tokenizer_json", "hf_asset__", "text_embedding_projection.",
         "connectors.", "video_embeddings_connector.", "audio_embeddings_connector.",
         "video_connector.", "audio_connector.", "lm_head.", "model.lm_head."),
        device,
    )
    return (wrapper.to(device).eval(), GemmaPrompter.from_transformers(tokenizer),
            hf_config.text_config)
