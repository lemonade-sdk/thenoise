# Unified attention function over a batch of sequences.
#
# The default method is the comfy-kitchen INT8 attention kernel: it keeps the
# projection weights at their native BF16/FP16 dtype while running the attention
# compute as INT8 on the GPU's matrix cores (available on RDNA3/RDNA4 via the
# HIP backend). PyTorch's scaled dot-product attention (SDPA) remains available as
# the other method.
#

import logging
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
import comfy_kitchen


logger = logging.getLogger(__name__)

# Attention method constants.
ATTENTION_SDPA = "sdpa"
ATTENTION_INT8 = "int8"
ATTENTION_METHODS = frozenset((ATTENTION_SDPA, ATTENTION_INT8))
# INT8 is the default for the DiT attention path; see the module docstring.
DEFAULT_ATTENTION_METHOD = ATTENTION_INT8


@dataclass
class AttentionParams:
    attention_mask: Optional[torch.Tensor] = None

    @staticmethod
    def create_attention_params() -> "AttentionParams":
        return AttentionParams()

    @staticmethod
    def create_attention_params_from_mask(
        img_len: Optional[int], attention_mask: Optional[torch.Tensor]
    ) -> "AttentionParams":
        if attention_mask is None:
            # No attention mask provided: assume all tokens are valid
            return AttentionParams()
        # Note: attention_mask is only for text tokens, not including image tokens.
        # Expand to include the (always valid) image tokens, then shape as an SDPA
        # key-padding mask: [B, 1, 1, img_len + L].
        attention_mask = F.pad(attention_mask, (img_len, 0), value=1)  # [B, img_len + L]
        attention_mask = attention_mask[:, None, None, :].to(torch.bool)  # [B, 1, 1, img_len + L]

        return AttentionParams(attention_mask)


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    drop_rate: float,
) -> torch.Tensor:
    """SDPA on ``[B, H, L, D]`` tensors -> ``[B, H, L, D]`` output.

    GQA: q may carry more heads than k/v (e.g. Krea 2 = 48 query / 12 kv heads).
    SDPA has no native fused GQA path, so we expand k/v to q's head count. We avoid
    enable_gqa=True because that forces SDPA onto the slow math kernel (~7x slower at K2
    scale); the repeat is numerically identical. (q/k/v here are [B, H, L, D].)
    """
    if q.shape[1] != k.shape[1]:  # heads at dim 1
        g = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=drop_rate
    )


def _int8_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """comfy-kitchen INT8 attention on ``[B, H, L, D]`` -> ``[B, H, L, D]``.

    Weights stay BF16/FP16; only the Q/K/V activations are quantized to INT8 inside
    the kernel. Grouped-query attention is supported natively, so no head expansion is
    needed here.
    """
    return comfy_kitchen.int8_attention(q, k, v, attn_mask=attn_mask)


def attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
    method: str = DEFAULT_ATTENTION_METHOD,
) -> torch.Tensor:
    """
    Compute attention over a batch of sequences.

    The whole batch is processed in a single call. Variable sequence lengths are handled
    with a key-padding ``attention_mask`` (padding positions excluded from the softmax).
    ``attn_params`` may carry no mask, meaning all tokens are valid.

    Args:
        qkv_or_q: Query tensor [B, L, H, D]. or list of such tensors.
        k: Key tensor [B, L, H, D].
        v: Value tensor [B, L, H, D].
        attn_params: Attention parameters including the optional key-padding mask.
        drop_rate: Attention dropout rate (SDPA method only; the INT8 kernel has no
            dropout, so it is ignored when ``method`` is ``ATTENTION_INT8``).
        method: One of ``ATTENTION_SDPA`` or ``ATTENTION_INT8``. Defaults to
            ``DEFAULT_ATTENTION_METHOD`` (``ATTENTION_INT8``).

    Returns:
        Attention output tensor [B, L, H*D].
    """
    if method not in ATTENTION_METHODS:
        raise ValueError(
            f"unsupported attention method {method!r}; expected one of "
            f"{sorted(ATTENTION_METHODS)}"
        )
    if isinstance(qkv_or_q, list):
        q, k, v = qkv_or_q
        q: torch.Tensor = q
        qkv_or_q.clear()
        del qkv_or_q
    else:
        q: torch.Tensor = qkv_or_q
        del qkv_or_q
        assert k is not None and v is not None, "k and v must be provided if qkv_or_q is a tensor"
    if attn_params is None:
        attn_params = AttentionParams.create_attention_params()

    # Both methods use the [B, H, L, D] layout, so transpose once from the
    # [B, L, H, D] input.
    transpose_fn = lambda x: x.transpose(1, 2)

    q = transpose_fn(q)
    k = transpose_fn(k)
    v = transpose_fn(v)

    if method == ATTENTION_INT8:
        x = _int8_attention(q, k, v, attn_params.attention_mask)
    else:
        x = _sdpa_attention(q, k, v, attn_params.attention_mask, drop_rate)

    x = transpose_fn(x)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]

    return x
