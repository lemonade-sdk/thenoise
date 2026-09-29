"""The Ming-Image conditioner: the BailingMM2 thinker, its connector, and the two
tensors the DiT is conditioned on.

Ming-Image does not read its LLM's last hidden state as a caption. One causal
pass over the prompt produces **two** tensors of different shape and provenance:

* ``cap_feats`` ``[1, 256, 2560]`` — the prompt carries 256 learnable *query*
  tokens, spliced over the ``<imagePatch>`` token as a fake image block. Their
  thinker hidden states go through ``proj_in``, the 28-layer **bidirectional**
  Qwen2 connector, and ``proj_out``. This is the tensor the DiT's
  ``cap_embedder`` eats.
* ``direct_context`` ``[1, P, 3840]`` — ``proj_directvlm`` of the three captured
  thinker states (the inputs of layers 5 and 12, and the post-final-norm state)
  over the **prompt span only** (``<image>`` excluded). Already DiT-wide, so the
  Lumina core concatenates it *after* ``cap_embedder``.

Those two tensors, not the LLM, are what makes Ming-Image its own model. The
query block also decides *which experts* run: image positions route through
``mlp.image_gate`` (``router_type: MultiRouter``), i.e. entirely different
experts from the prompt text. Hence the block geometry (:data:`QUERY_BLOCK_GRID`)
is part of this module's contract, not an implementation detail.

Checkpoint facts this module is written against — measured from the released bf16
and int8-convrot text-encoder files, not from the paper:

``thinker.*`` (BailingMoeV2) ``connector.*`` ``query_tokens`` ``proj_in``
``proj_out`` ``proj_directvlm.{0,1}`` ``tokenizer_json`` (a 12 MB U8 payload)
``thinker.lm_head.*`` ``vision.*`` + ``linear_proj.*`` (the image tower, not
built here). The module tree mirrors the file's own names exactly, so the load
needs no key map; the four prefixes the conditioner does not build are dropped
(:data:`TEXT_ENCODER_DROP_KEYS`) and the load stays strict.

Ported from the vendor's ``modeling_bailingmm2.py`` (Ant Group `Ming-Image`,
Apache-2.0) and cross-checked line by line against ComfyUI's condensed
``comfy/text_encoders/ming_image.py``.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from tokenizers import Tokenizer

from thenoise.dit.quantized import QuantizedLinear
from thenoise.text_encoders.bailing_moe import (
    BailingMoeV2,
    BailingMoeV2Config,
    MLP,
    Block,
)
from thenoise.utils.attention import AttentionParams, attention
from thenoise.utils.loader import load_text_encoder_weights
from thenoise.utils.rms_norm import RMSNorm
from thenoise.utils.rope import apply_rope_split_half, split_half_rope_1d
from thenoise.utils.safetensors import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------- prompt side

#: ``tokenizer_json``'s own ids for the block the query tokens are spliced into.
IMAGE_TOKEN = 157158  # <image>
IMAGE_PATCH_TOKEN = 157157  # <imagePatch>: the row the query tokens replace
IMAGE_END_TOKEN = 157159  # </image>
#: The file's pad id. Batch is always 1, so it is only ever a name we know.
PAD_TOKEN = 156892

#: ``img_gen_scales`` is ``[16]``, so the learnable block is 16² = 256 tokens. The
#: reference registers it as ``image_grid_thw = [1, 2, 2 * 256]``, which the vision
#: merger's 2×2 fold turns into a ``1 x 256`` grid: one row, every token spread
#: along the width axis. A different grid is not an error, it is a silently worse
#: picture — the rope positions of the whole sequence follow this block.
QUERY_TOKENS = 256
QUERY_BLOCK_GRID: Tuple[int, int] = (1, QUERY_TOKENS)

#: The vendor generator's own chat template: its default Chinese system turn, the
#: ``detailed thinking off`` switch it always sends, and the query block appended
#: after the open assistant turn.
T2I_PROMPT_TEMPLATE = (
    "<role>SYSTEM</role>你是一个友好的AI助手。\n\n"
    "detailed thinking off<|role_end|>"
    "<role>HUMAN</role>{prompt}<|role_end|>"
    "<role>ASSISTANT</role>"
    "<image><imagePatch></image>"
)

#: ``mlp/config.json``'s ``selected_hidden_states_layers``. In the reference's
#: numbering, index ``k`` below the layer count is the INPUT of layer ``k``; the
#: last entry (20 == the layer count) is the post-final-norm state, which
#: :class:`~thenoise.text_encoders.bailing_moe.BailingMoeV2` always appends.
SELECTED_LAYERS: Tuple[int, ...] = (5, 12, 20)

#: ``mlp/config.json``: ``diffusion_c_input_dim`` / ``diffusion_inner_dim``.
CAP_FEAT_DIM = 2560
DIRECT_DIM = 3840

#: What the released file carries that this module tree does not build: the image
#: tower and its projection (the edit path), the LM head (never run), and the
#: tokenizer payload (read on its own, see :func:`load_ming_tokenizer`).
TEXT_ENCODER_DROP_KEYS = (
    "vision.",
    "linear_proj.",
    "thinker.lm_head.",
    "tokenizer_json",
)


class MingTokenizerError(RuntimeError):
    """Raised when neither the file nor a fallback directory has a usable tokenizer."""


# ------------------------------------------------------------------- the connector


@dataclass
class MingConnectorConfig:
    """The connector as ``connector/config.json`` and the 1.31 B of weights measure it."""

    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0


class ConnectorAttention(nn.Module):
    """GQA attention with separate, BIASED q/k/v and NO causal mask.

    Both halves of that sentence are load-bearing: the file ships
    ``self_attn.{q,k,v}_proj.bias`` (Qwen2, unlike the thinker), and the vendor
    sets ``is_causal = False`` on every connector layer while handing it an
    all-ones mask — the connector reads its query tokens as one bidirectional set,
    not as a sequence.
    """

    def __init__(self, config: MingConnectorConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = QuantizedLinear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = QuantizedLinear(config.hidden_size, kv_dim, bias=True)
        self.v_proj = QuantizedLinear(config.hidden_size, kv_dim, bias=True)
        self.o_proj = QuantizedLinear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        query = self.q_proj(x).unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        key = self.k_proj(x).unflatten(-1, (self.num_kv_heads, self.head_dim)).transpose(1, 2)
        value = self.v_proj(x).unflatten(-1, (self.num_kv_heads, self.head_dim)).transpose(1, 2)
        # Full-head split-half RoPE: no partial factor here, unlike the thinker.
        query = apply_rope_split_half(query, *freqs)
        key = apply_rope_split_half(key, *freqs)
        # The missing mask IS the bidirectionality: every query reads every key,
        # which is what an all-ones 4D mask means to the reference.
        return self.o_proj(attention([query, key, value], attn_params=AttentionParams(None)))


class ConnectorDecoderLayer(nn.Module):
    """Pre-norm Qwen2 block, named exactly as the checkpoint spells it."""

    def __init__(self, config: MingConnectorConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ConnectorAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # The connector's feed-forward is the thinker's SwiGLU at other widths.
        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor, freqs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), freqs)
        return x + self.mlp(self.post_attention_layernorm(x))


class MingConnector(nn.Module):
    """``connector.layers.*`` + ``connector.norm``: 28 bidirectional blocks, then RMSNorm.

    The trailing norm belongs to this module because the reference reads
    ``connector(...).hidden_states[-1]``, and transformers appends that state AFTER
    the model's final norm.
    """

    def __init__(self, config: Optional[MingConnectorConfig] = None) -> None:
        super().__init__()
        self.config = config or MingConnectorConfig()
        self.layers = nn.ModuleList(
            ConnectorDecoderLayer(self.config) for _ in range(self.config.num_hidden_layers)
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self._rope = split_half_rope_1d(self.config.head_dim, self.config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Positions are the plain ``0..L-1`` of the block handed to it; the thinker's
        # ``video_rope`` business does not reach in here.
        cos, sin = self._rope(x.shape[1], x.device)
        freqs = (cos.to(x.dtype), sin.to(x.dtype))
        for layer in self.layers:
            x = layer(x, freqs)
        return self.norm(x)


# ------------------------------------------------------------------- the conditioner


class MingImageConditioner(nn.Module):
    """The released conditioner end to end, under the checkpoint's own module names.

    Built on meta and filled by :func:`load_ming_text_encoder`. The tree has no
    ``vision``/``linear_proj``/``lm_head`` — those prefixes are dropped at load —
    so anything else the file cannot place is a hard error rather than a zero fill.
    """

    def __init__(
        self,
        *,
        thinker: Optional[Union[BailingMoeV2, BailingMoeV2Config]] = None,
        connector: Optional[Union[MingConnector, MingConnectorConfig]] = None,
        selected_layers: Sequence[int] = SELECTED_LAYERS,
        num_queries: int = QUERY_TOKENS,
        query_grid: Tuple[int, int] = QUERY_BLOCK_GRID,
        cap_feat_dim: int = CAP_FEAT_DIM,
        direct_dim: int = DIRECT_DIM,
        direct_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        # A config stands for "build it" — what lets a caller (the loader, a test)
        # resize either half without constructing it itself.
        if isinstance(thinker, BailingMoeV2Config):
            thinker = BailingMoeV2(thinker)
        if isinstance(connector, MingConnectorConfig):
            connector = MingConnector(connector)
        self.thinker = thinker or BailingMoeV2()
        self.connector = connector or MingConnector()
        hidden = self.thinker.config.hidden_size
        connector_hidden = self.connector.config.hidden_size

        layers = tuple(selected_layers)
        # Every entry but the last names a layer INPUT; the last must be the layer
        # count itself, i.e. the post-final-norm state index 20 stands for.
        if not layers or layers[-1] != self.thinker.config.num_hidden_layers:
            raise ValueError(
                f"selected_layers {layers} must end at the thinker's layer count "
                f"({self.thinker.config.num_hidden_layers}): that is the "
                "post-final-norm state the reference's last index captures"
            )
        self.selected_layers = layers
        self.capture_pre_layers = tuple(k for k in layers if k < layers[-1])

        if num_queries != query_grid[0] * query_grid[1]:
            raise ValueError(
                f"{num_queries} query tokens do not fill a {query_grid} block "
                f"({query_grid[0] * query_grid[1]} tokens)"
            )
        self.num_queries = num_queries
        self.query_grid = tuple(query_grid)
        # Drawn like the embedding rows these stand in for: the checkpoint overwrites
        # them, and an uninitialised ``torch.empty`` is garbage that bf16 then rounds
        # into a different garbage (so a hand-built conditioner is not even
        # reproducible through a save/load round-trip).
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden))

        self.proj_in = QuantizedLinear(hidden, connector_hidden, bias=True)
        self.proj_out = QuantizedLinear(connector_hidden, cap_feat_dim, bias=True)
        direct_in = hidden * len(layers)
        self.proj_directvlm = nn.Sequential(
            RMSNorm(direct_in, eps=direct_norm_eps),
            QuantizedLinear(direct_in, direct_dim, bias=True),
        )

    @property
    def device(self) -> torch.device:
        return self.query_tokens.device

    def query_block(self, embeddings: torch.Tensor, query_index: int) -> torch.Tensor:
        """Splice the learnable rows OVER the ``<imagePatch>`` embedding.

        The result is ``[prompt 0..q-1][query tokens][</image>]``: the token under
        ``q`` is REPLACED, not shifted. That is what makes the reference's ``:q-1``
        prompt span, and the rope counter that steps one position over the block,
        fall out of the same index.
        """
        if embeddings.shape[0] != 1:
            raise ValueError(
                f"the Ming-Image conditioner encodes one prompt at a time, got "
                f"{embeddings.shape[0]}"
            )
        queries = self.query_tokens.to(embeddings.dtype).expand(embeddings.shape[0], -1, -1)
        return torch.cat(
            [embeddings[:, :query_index], queries, embeddings[:, query_index + 1 :]], dim=1
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``input_ids`` ``[1, L]`` (from :func:`build_prompt_ids`) to the two tensors.

        Returns ``(cap_feats [1, num_queries, cap_feat_dim], direct_context
        [1, P, direct_dim])``.
        """
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"the Ming-Image conditioner encodes one prompt at a time, got "
                f"{input_ids.shape[0]}"
            )
        hits = (input_ids[0] == IMAGE_PATCH_TOKEN).nonzero()
        if hits.numel() != 1:
            raise ValueError(
                f"expected exactly one <imagePatch> token, found {hits.numel()}; "
                "build the ids with build_prompt_ids()"
            )
        q = int(hits[0, 0])

        embeddings = self.thinker.embed_tokens(input_ids)
        hidden, captured = self.thinker(
            self.query_block(embeddings, q),
            blocks=[Block((q, self.query_grid[0], self.query_grid[1]))],
            attention_mask=attention_mask,
            capture_pre_layers=self.capture_pre_layers,
        )
        if len(captured) != len(self.selected_layers):
            raise RuntimeError(
                f"the thinker captured {len(captured)} states, expected "
                f"{len(self.selected_layers)} for {self.selected_layers}"
            )

        block = hidden[:, q : q + self.num_queries]
        cap_feats = self.proj_out(self.connector(self.proj_in(block)))

        # ``:q-1`` is the prompt span with ``<image>`` excluded — the vendor's
        # ``labels < 0`` mask, i.e. exactly "the tokens the user sent".
        direct = torch.cat([state[:, : q - 1] for state in captured], dim=-1)
        return cap_feats, self.proj_directvlm(direct)


# ------------------------------------------------------------------- tokenizer


def build_prompt(prompt: str) -> str:
    """The exact text the released generator tokenizes."""
    return T2I_PROMPT_TEMPLATE.format(prompt=prompt)


def build_prompt_ids(tokenizer: Tokenizer, prompt: str) -> List[int]:
    """Tokenize :func:`build_prompt` with ``add_special_tokens=False``.

    Nothing is added because the template already spells out both delimiters —
    letting the tokenizer prepend one would shift every rope position.
    """
    return tokenizer.encode(build_prompt(prompt), add_special_tokens=False).ids


def check_query_block(ids: Sequence[int]) -> int:
    """The index of the ``<imagePatch>`` token, asserted against its own delimiters.

    Asserted rather than trusted: a template that stopped emitting the marker (or a
    tokenizer that split it) would otherwise encode a picture out of whatever the
    tokenizer made of the text, with no error anywhere.
    """
    hits = [i for i, token in enumerate(ids) if token == IMAGE_PATCH_TOKEN]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one <imagePatch> token, found {len(hits)}")
    q = hits[0]
    # Bounds-guarded rather than indexed: a marker at either end is a broken
    # template, which has to read as this error, not as an IndexError.
    before = ids[q - 1] if q >= 1 else None
    after = ids[q + 1] if q + 1 < len(ids) else None
    if before != IMAGE_TOKEN or after != IMAGE_END_TOKEN:
        raise ValueError(
            f"<imagePatch> at {q} is not surrounded by <image>/</image> "
            f"(found {before}/{after}); "
            "the prompt template and the tokenizer disagree about the query block"
        )
    return q


def load_ming_tokenizer(path: str, *, tokenizer_dir: Optional[str] = None) -> Tokenizer:
    """The tokenizer, which the text-encoder FILE carries as a U8 JSON payload.

    Nothing to vendor and nothing extra to download. ``tokenizer_dir`` (a directory
    holding ``tokenizer.json``) is the fallback for a file stripped of the payload,
    e.g. a repackaging that dropped the non-weight tensor.
    """
    raw = _read_tokenizer_json(path)
    if raw is None:
        candidate = os.path.join(tokenizer_dir, "tokenizer.json") if tokenizer_dir else None
        if not candidate or not os.path.isfile(candidate):
            raise MingTokenizerError(
                f"{path!r} carries no tokenizer_json tensor"
                + (f" and no tokenizer at {candidate!r}" if candidate else "")
                + "; pass a tokenizer_dir pointing at a directory with tokenizer.json"
            )
        with open(candidate, "rb") as handle:
            raw = handle.read()
        logger.info("Loaded the Ming-Image tokenizer from %s", candidate)
    return Tokenizer.from_str(raw.decode("utf-8"))


def _read_tokenizer_json(path: str) -> Optional[bytes]:
    """The ``tokenizer_json`` payload as bytes: header read plus one slice."""
    with MemoryEfficientSafeOpen(path) as f:
        if "tokenizer_json" not in f.keys():
            return None
        tensor = f.get_tensor("tokenizer_json")
    return tensor.reshape(-1).cpu().to(torch.uint8).numpy().tobytes()


# ------------------------------------------------------------------- load and encode


def load_ming_text_encoder(
    path: str,
    *,
    device: Union[str, torch.device],
    dtype: torch.dtype,
    tokenizer_dir: Optional[str] = None,
    with_tokenizer: bool = True,
    config: Optional[dict] = None,
) -> Tuple[MingImageConditioner, Optional[Tokenizer]]:
    """Build the conditioner on meta, load one safetensors file into it, attach the tokenizer.

    ``config`` overrides ``MingImageConditioner``'s constructor keywords (the same
    escape hatch ``load_ming_dit`` has): the released defaults are an 18 B
    conditioner, and a test that only wants to exercise the load path asks for a
    tiny one here.

    bf16 and int8-convrot are chosen from the file's own header by
    ``load_text_encoder_weights``, like every other adapter. Nothing is
    dequantized on the way in: the ``[experts, out, in]`` banks stay low-bit and
    run one row-sliced ``QuantizedTensor`` per expert (see
    :class:`~thenoise.text_encoders.bailing_moe.ExpertBank`), so an int8 text
    encoder really is about a third of the bf16 one.
    """
    logger.info("Loading Ming-Image text encoder from %s", path)

    with init_empty_weights():
        model = MingImageConditioner(**(config or {}))

    load_text_encoder_weights(
        model,
        path,
        device=device,
        dtype=dtype,
        drop_keys=TEXT_ENCODER_DROP_KEYS,
    )
    tokenizer = load_ming_tokenizer(path, tokenizer_dir=tokenizer_dir) if with_tokenizer else None
    return model.eval().requires_grad_(False), tokenizer


@torch.no_grad()
def encode_ming_prompt(
    conditioner: MingImageConditioner,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One prompt to the DiT's two conditioning tensors, both with a leading batch dim."""
    ids = build_prompt_ids(tokenizer, prompt)
    check_query_block(ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device=conditioner.device)
    cap_feats, direct_context = conditioner(input_ids)
    if dtype is not None:
        cap_feats, direct_context = cap_feats.to(dtype), direct_context.to(dtype)
    return cap_feats, direct_context


__all__ = [
    "CAP_FEAT_DIM",
    "DIRECT_DIM",
    "IMAGE_END_TOKEN",
    "IMAGE_PATCH_TOKEN",
    "IMAGE_TOKEN",
    "MingConnector",
    "MingConnectorConfig",
    "MingImageConditioner",
    "MingTokenizerError",
    "PAD_TOKEN",
    "QUERY_BLOCK_GRID",
    "QUERY_TOKENS",
    "SELECTED_LAYERS",
    "T2I_PROMPT_TEMPLATE",
    "TEXT_ENCODER_DROP_KEYS",
    "build_prompt",
    "build_prompt_ids",
    "check_query_block",
    "encode_ming_prompt",
    "load_ming_text_encoder",
    "load_ming_tokenizer",
]
