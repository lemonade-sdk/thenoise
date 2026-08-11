# Unified attention function using PyTorch's scaled dot-product attention (SDPA).

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from typing import Optional, Union


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


def attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention over a batch of sequences.

    The whole batch is processed in a single SDPA call. Variable sequence lengths are
    handled with a key-padding ``attention_mask`` (padding positions excluded from the
    softmax). ``attn_params`` may carry no mask, meaning all tokens are valid.

    Args:
        qkv_or_q: Query tensor [B, L, H, D]. or list of such tensors.
        k: Key tensor [B, L, H, D].
        v: Value tensor [B, L, H, D].
        attn_param: Attention parameters including the optional key-padding mask.
        drop_rate: Attention dropout rate.

    Returns:
        Attention output tensor [B, L, H*D].
    """
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

    # GQA: q may carry more heads than k/v (e.g. Krea 2 = 48 query / 12 kv heads). SDPA has no
    # native fused GQA path, so we expand k/v to q's head count. We avoid enable_gqa=True because
    # that forces SDPA onto the slow math kernel (~7x slower at K2 scale); the repeat is numerically
    # identical. (q/k/v here are [B, L, H, D].)
    enable_gqa = q.shape[-2] != k.shape[-2]

    # SDPA layout is [B, H, L, D], so transpose from the [B, L, H, D] input.
    transpose_fn = lambda x: x.transpose(1, 2)

    q = transpose_fn(q)
    k = transpose_fn(k)
    v = transpose_fn(v)

    if enable_gqa:  # expand k/v heads to avoid SDPA's slow enable_gqa math path
        g = q.shape[1] // k.shape[1]  # [B, H, L, D] -> heads at dim 1
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

    x = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_params.attention_mask, dropout_p=drop_rate
    )

    x = transpose_fn(x)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]

    return x
