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


def l_major(t: torch.Tensor) -> bool:
    """True when ``L`` is the second-innermost dimension of a ``[B, H, L, D]`` tensor.

    That order — not plain ``is_contiguous()`` — is what the fused attention kernels
    want: an offset slice of a bigger buffer has it and runs at full speed, while a
    ``view(B, L, H, D).transpose(1, 2)`` of a packed projection does not.
    """
    return t.dim() == 4 and t.stride(-1) == 1 and t.stride(-2) == t.size(-1)


def uniform_layout(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Put ``q``/``k``/``v`` in the ``L``-major layout SDPA's fast kernels require.

    The fused attention kernels pick their code path from the input strides, and the
    blocks naturally hand them a trio that is not ``L``-major: ``v`` is the
    ``view(B, L, H, D).transpose(1, 2)`` of its packed projection, and inside a
    compiled block Inductor may keep q/k in that same packed order too rather than
    materialising ``[B, H, L, D]``. The measured cost on gfx1151, at 16k tokens:
    all three ``L``-major 32.6 TFLOPS, one packed straggler 5.4, all packed 1.7 — so a
    mixed trio is up to six times slower and a wholly packed one twenty, which reads as
    a resolution cliff because at 4k tokens the mixed case loses only ~8 %.

    Normalising *to ``L``-major* rather than to whichever layout ``k`` happens to have
    matters: normalising towards the packed order would make things worse, and eager
    and compiled blocks disagree about which layout that is (eager's norm and RoPE
    hand back ``[B, H, L, D]`` contiguous; a fused block may keep the packed order).

    Slices keep strides, so a segment's offset ``q`` is already ``L``-major and free,
    and a trio already in the right order copies nothing (``contiguous()`` returns its
    input when there is nothing to fix).
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
    """
    Executes sdpa in eager mode. 
    Both ROCm 7.14 and 10.1 show significant performance drop at higher token count
    in some scenarios. The underlying cause needs further investigation but this
    workaround doesn't harm performance.
    """
    return attention(qkv_or_q, k=k, v=v, attn_params=attn_params, drop_rate=drop_rate)

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
        qkv_or_q: Query tensor [B, H, L, D]. or list of such tensors.
        k: Key tensor [B, H, L, D].
        v: Value tensor [B, H, L, D].
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
    # identical. (q/k/v here are [B, H, L, D].)
    enable_gqa = q.shape[1] != k.shape[1]

    if enable_gqa:  # expand k/v heads to avoid SDPA's slow enable_gqa math path
        g = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(g, dim=1)
        v = v.repeat_interleave(g, dim=1)

    q, k, v = uniform_layout(q, k, v)

    x = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_params.attention_mask, dropout_p=drop_rate
    )

    # Token-major output [B, L, H*D] (heads concatenated per token), as callers expect.
    x = x.transpose(1, 2)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]

    return x
