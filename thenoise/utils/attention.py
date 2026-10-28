# SDPA wrapper for the DiT (masking, GQA, layout), and the manual attention the VAE
# decoders use instead of SDPA.

from dataclasses import dataclass
import math
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

SCORE_LIMIT_BYTES = 2 << 30   # at or under this a whole matrix is scored at once
SCORE_TILE_BYTES = 256 << 20  # target tile size above it
MIN_TILE_ROWS = 64            # thinner than this a tile is launch overhead, not a tile


def score_tile_rows(n: int) -> int:
    """Query rows to score at once for an ``N x N`` score matrix. ``n`` means: all of it.

    A row costs ``2N`` bytes since the probabilities overwrite the scores, so the
    ``4N`` divisor leaves every tile at half of ``SCORE_TILE_BYTES``, and tiling engages
    at half the token count the limit suggests. Halving rows back would spend the same
    bytes for half the launches; the footprints decided for the saving.
    """
    if 4 * n * n <= SCORE_LIMIT_BYTES:
        return n
    rows = -(-SCORE_TILE_BYTES // (4 * n))
    return max(MIN_TILE_ROWS, min(n, rows))


def single_head_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          rows: int | None = None) -> torch.Tensor:
    """Attention over the second-to-last axis as two matmuls and a softmax.

    A VAE attention block is one head as wide as the whole channel count, over every
    pixel of the latent. As of ROCm 10.1.0rc2, the fused SDPA backends do not serve 
    that shape honestly: they refuse it, or accept it and return wrong values, and the 
    math backend they fall back to costs several times these two matmuls.

    Both paths write the probabilities over the scores they came from: the same row
    operation, and a whole score matrix less — 0.5 GiB on an untiled 16384-token grid,
    one tile per call above that, bit-identical on the machines tried. ``out=`` aliasing
    is not an ATen promise; were it ever dropped, the answer stays correct and only the
    footprint returns.
    """
    n, embed = q.shape[-2], q.shape[-1]
    scale = math.sqrt(embed)
    if rows is None:
        rows = score_tile_rows(n)

    if rows >= n:
        scores = q @ k.transpose(-2, -1)
        scores.div_(scale)
        return torch.softmax(scores, -1, out=scores) @ v

    rows = max(MIN_TILE_ROWS, rows)
    out = torch.empty((*q.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device)
    scratch = torch.empty((*q.shape[:-2], rows, n), dtype=q.dtype, device=q.device)
    keys_t = k.transpose(-2, -1)
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        tile = scratch[..., : stop - start, :]
        torch.matmul(q[..., start:stop, :], keys_t, out=tile)
        tile.div_(scale)
        probs = torch.softmax(tile, -1, out=tile)
        torch.matmul(probs, v, out=out[..., start:stop, :])
    return out


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
