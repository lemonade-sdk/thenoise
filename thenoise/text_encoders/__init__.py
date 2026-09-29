"""Text encoders this repo has to own: architectures ``transformers`` cannot run.

The adapters' other encoders are stock ``transformers`` models loaded through
:mod:`thenoise.utils.text_encoder`. What lives here instead is a model's own
conditioner, vendored against this repo's primitives (``QuantizedLinear``, the
shared ``RMSNorm``/``attention``/RoPE helpers) because the upstream file depends
on private transformers APIs the installed version dropped:

* :mod:`~thenoise.text_encoders.bailing_moe` — BailingMoeV2, the Ling-2.0-mini MoE
  language core (group-limited top-k routing with a separate *image* router,
  fused ``[experts, out, in]`` banks, ``video_rope``).
* :mod:`~thenoise.text_encoders.ming_image` — Ming-Image's conditioner on top of
  it: the query-token block, the 28-layer bidirectional connector, and the two
  conditioning tensors its DiT reads.
"""
from thenoise.text_encoders.bailing_moe import (
    MROPE_SECTION,
    BailingAttention,
    BailingMoeV2,
    BailingMoeV2Config,
    Block,
    DecoderLayer,
    ExpertBank,
    Experts,
    Gate,
    MLP,
    SparseMoeBlock,
    expert_dispatch,
    video_rope,
)
from thenoise.text_encoders.ming_image import (
    CAP_FEAT_DIM,
    DIRECT_DIM,
    IMAGE_END_TOKEN,
    IMAGE_PATCH_TOKEN,
    IMAGE_TOKEN,
    PAD_TOKEN,
    QUERY_BLOCK_GRID,
    QUERY_TOKENS,
    SELECTED_LAYERS,
    T2I_PROMPT_TEMPLATE,
    TEXT_ENCODER_DROP_KEYS,
    MingConnector,
    MingConnectorConfig,
    MingImageConditioner,
    MingTokenizerError,
    build_prompt,
    build_prompt_ids,
    check_query_block,
    encode_ming_prompt,
    load_ming_text_encoder,
    load_ming_tokenizer,
)

__all__ = [
    "BailingAttention",
    "BailingMoeV2",
    "BailingMoeV2Config",
    "Block",
    "CAP_FEAT_DIM",
    "DecoderLayer",
    "DIRECT_DIM",
    "ExpertBank",
    "Experts",
    "Gate",
    "IMAGE_END_TOKEN",
    "IMAGE_PATCH_TOKEN",
    "IMAGE_TOKEN",
    "MLP",
    "MROPE_SECTION",
    "MingConnector",
    "MingConnectorConfig",
    "MingImageConditioner",
    "MingTokenizerError",
    "PAD_TOKEN",
    "QUERY_BLOCK_GRID",
    "QUERY_TOKENS",
    "SELECTED_LAYERS",
    "SparseMoeBlock",
    "T2I_PROMPT_TEMPLATE",
    "TEXT_ENCODER_DROP_KEYS",
    "build_prompt",
    "build_prompt_ids",
    "check_query_block",
    "encode_ming_prompt",
    "expert_dispatch",
    "load_ming_text_encoder",
    "load_ming_tokenizer",
    "video_rope",
]
