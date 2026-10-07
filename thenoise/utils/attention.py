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

# Score a whole N x N matrix while it fits in this. One live bf16 matrix, not two:
# the probabilities are written over the scores, so the buffer is no longer doubled.
#
# Measured on gfx1151, not derived. At N=49152 (4.3 GiB) scoring whole beat 1024-row
# tiles on every codec — it took the 1536x2048 rungs from +6.7% to unchanged. At
# N=65536 (8 GiB) it went the other way: +6.8% on qwen/ming encodes, +4.7% on their
# decodes, and only flux/flux2 improved (0.7-1.9%). So the allowance is one rung,
# not infinity: 5 GiB scores everything up to 51k positions whole and lets the rest
# tile exactly as wide as it used to.
SCORE_LIMIT_BYTES = 5 << 30
SCORE_TILE_BYTES = 256 << 20    # target tile size above the limit
MIN_TILE_ROWS = 64              # thinner than this a tile is launch overhead, not a tile


def score_tile_rows(n: int) -> int:
    """Query rows to score at once for an ``N x N`` score matrix. ``n`` means: all of it.

    A row of scores costs ``2N`` bytes — one live bf16 matrix ``N`` wide, since
    ``single_head_attention`` overwrites the scores with probabilities — so a tile is
    ``SCORE_TILE_BYTES / 2N`` rows, floored at ``MIN_TILE_ROWS``.
    """
    if 2 * n * n <= SCORE_LIMIT_BYTES:
        return n
    rows = -(-SCORE_TILE_BYTES // (2 * n))
    return max(MIN_TILE_ROWS, min(n, rows))


def single_head_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          rows: int | None = None) -> torch.Tensor:
    """Attention over the second-to-last axis as two matmuls and a softmax.

    A VAE attention block is one head as wide as the whole channel count, over every
    pixel of the latent. As of ROCm 10.1.0rc2 the fused SDPA backends do not serve
    that shape honestly: they refuse it, or accept it and return wrong values (the
    AOTriton kernel behind ``EFFICIENT_ATTENTION`` returns NaN above head_dim 256),
    and the math backend they fall back to costs several times these two matmuls.

    Nothing here is made contiguous. A GEMM wants a leading dimension, not an
    orientation, so the operands are consumed exactly as the 1x1 convs left them —
    a ``(C, L)`` matrix read as ``(L, C)`` tokens — and the result is written back in
    that same orientation, as ``v' @ p'`` rather than ``p @ v``, so the projection
    conv reads it without a transposing copy either. The orientation follows ``q``'s
    strides: hand over packed ``(L, C)`` tokens and packed ``(L, C)`` comes back.
    Copying instead is what used to cost a flux decode four full passes over the
    feature map and left it the slowest of the codecs that copy nothing.

    The ``1/sqrt(E)`` rides in the score GEMM's ``alpha``, which the kernel applies to
    its fp32 accumulator. Dividing the score matrix afterwards reads and writes N*N
    elements to do what a scalar costs; folding it into ``q`` instead would round a
    bf16 operand and measure worse than the divide, which this beats on both counts.
    """
    n, embed = q.shape[-2], q.shape[-1]
    if rows is None:
        rows = score_tile_rows(n)
    rows = n if rows >= n else max(MIN_TILE_ROWS, rows)

    # ``baddbmm`` is 3-D only, so the batch axes collapse first. A reshape that cannot
    # view would copy — at N*C, the size this function is careful about last, since
    # every operand here is smaller than the score matrix it is about to build.
    q3, k3, v3 = (t.reshape(-1, n, embed) for t in (q, k, v))
    keys_t = k3.transpose(-2, -1)
    # ``(L, C)`` tokens whose channel axis is the strided one: a channels-first conv,
    # which is every conv here. Its answer goes back the same way, into a (C, L) buffer
    # — which is free when the matrix is scored in one piece and costs a column-slice
    # store per tile when it is not: each tile then writes ``rows`` columns of a
    # ``(C, L)`` matrix instead of a contiguous block of rows. Untiled that store is
    # the whole point (the projection conv reads the buffer untouched); tiled it is the
    # one part of the old path that was not already slow, so tiled goes back to writing
    # token-major and the caller's existing ``transpose + reshape`` does one N*C copy
    # (0.6 ms at 2048x2048) instead of the attention doing it tile by tile.
    channel_major = q.stride(-1) != 1 and rows >= n
    batch = q.shape[:-2]
    if channel_major:
        out3 = torch.empty((q3.shape[0], embed, n), dtype=q.dtype, device=q.device)
        vals = v3.transpose(-2, -1)
    else:
        out3 = torch.empty((q3.shape[0], n, embed), dtype=q.dtype, device=q.device)
        vals = v3

    scratch = torch.empty((q3.shape[0], rows, n), dtype=q.dtype, device=q.device)
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        # A full tile is the whole scratch, hence contiguous, which is what lets the
        # probabilities land on the scores below. The ragged last tile is a strided
        # view and softmax over an aliased strided view reads what it has already
        # written, so that one gets a buffer of its own.
        tile = scratch if stop - start == rows else torch.empty(
            (q3.shape[0], stop - start, n), dtype=q.dtype, device=q.device)
        torch.baddbmm(tile, q3[:, start:stop], keys_t, beta=0.0,
                      alpha=1.0 / math.sqrt(embed), out=tile)
        # Probabilities land on the scores: they are read once, by the matmul below,
        # and nothing else wanted the buffer. Same kernel, and it halves the scratch.
        torch.ops.aten._softmax.out(tile, -1, False, out=tile)
        if channel_major:
            torch.matmul(vals, tile.transpose(-2, -1), out=out3[:, :, start:stop])
        else:
            torch.matmul(tile, vals, out=out3[:, start:stop])

    return out3.reshape((*batch, embed, n)).transpose(-1, -2) if channel_major \
        else out3.reshape((*batch, n, embed))


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
