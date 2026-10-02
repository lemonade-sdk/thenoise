# SDPA wrapper handling masking, GQA and layout.

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
            return AttentionParams()
        # The mask covers text tokens only; expand to include the (always valid)
        # image tokens and shape it as an SDPA key-padding mask.
        attention_mask = F.pad(attention_mask, (img_len, 0), value=1)  # [B, img_len + L]
        attention_mask = attention_mask[:, None, None, :].to(torch.bool)  # [B, 1, 1, img_len + L]

        return AttentionParams(attention_mask)


def l_major(t: torch.Tensor) -> bool:
    """True when ``L`` is the second-innermost dimension of a ``[B, H, L, D]`` tensor.

    This is the layout the fused SDPA kernels want; a transpose of a packed
    ``[B, L, H, D]`` projection is contiguous in the plain sense but not ``L``-major.
    """
    return t.dim() == 4 and t.stride(-1) == 1 and t.stride(-2) == t.size(-1)


def uniform_layout(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Put ``q``/``k``/``v`` in the ``L``-major layout SDPA's fast kernels require.

    The kernels pick their code path from the input strides and a straggler in
    packed order costs an order of magnitude on gfx1151 at 16k tokens (32.6 TFLOPS
    all ``L``-major, 5.4 with one packed tensor, 1.7 all packed). Blocks hand over
    ``v`` as a transpose of its packed projection and, when compiled, possibly q/k
    too, so the trio must be normalised on every call.

    Already-``L``-major tensors copy nothing, and a segment's offset ``q`` keeps the
    strides of the buffer it slices.
    """
    if not l_major(q):
        q = q.contiguous()
    if not l_major(k):
        k = k.contiguous()
    if not l_major(v):
        v = v.contiguous()
    return q, k, v

@torch._dynamo.disable()
def eager_attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """Run SDPA eagerly.

    ROCm shows a large performance drop at high token counts when this call is
    compiled in some scenarios; disabling compilation here costs nothing otherwise.
    """
    return attention(qkv_or_q, k=k, v=v, attn_params=attn_params, drop_rate=drop_rate)

def attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """Compute scaled dot-product attention over a batch of sequences.

    The whole batch goes through a single SDPA call; variable sequence lengths are
    handled by the key-padding mask in ``attn_params`` (no mask = all tokens valid).

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

    # GQA: expand k/v to q's head count. SDPA's enable_gqa flag forces the math
    # kernel (~7x slower at large scale); the repeat is numerically identical.
    enable_gqa = q.shape[1] != k.shape[1]

    if enable_gqa:
        g = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

    q, k, v = uniform_layout(q, k, v)

    x = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_params.attention_mask, dropout_p=drop_rate
    )

    # Token-major output, heads concatenated per token.
    x = x.transpose(1, 2)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]

    return x
