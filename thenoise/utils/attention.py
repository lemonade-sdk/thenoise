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
    # A crop of the leading keys, in place of a mask. Compiled blocks guard on this
    # int by value and automatic dynamic promotes it; ``dynamic=False`` would not.
    key_prefix: Optional[int] = None

    @staticmethod
    def create_attention_params() -> "AttentionParams":
        return AttentionParams()

    @staticmethod
    def create_attention_params_from_mask(
        img_len: int, attention_mask: Optional[torch.Tensor]
    ) -> "AttentionParams":
        """Params for a key-padding mask over the text tail of a sequence.

        Reads the mask to choose, so it belongs to the eager side of a compiled block.
        """
        if attention_mask is None:
            return AttentionParams()
        # The mask covers text tokens only; expand to include the (always valid)
        # image tokens and shape it as an SDPA key-padding mask.
        attention_mask = F.pad(attention_mask, (img_len, 0), value=1)  # [B, img_len + L]
        attention_mask = attention_mask[:, None, None, :].to(torch.bool)  # [B, 1, 1, img_len + L]

        prefix = shared_key_prefix(attention_mask)
        if prefix is not None:
            if prefix == attention_mask.shape[-1]:
                # Nothing to crop, and the whole length as an int would be a guard
                # the block recompiles for.
                return AttentionParams()
            return AttentionParams(key_prefix=prefix)

        return AttentionParams(attention_mask)


def shared_key_prefix(attention_mask: torch.Tensor) -> Optional[int]:
    """The valid-key count when every query sees exactly the same keys, else None.

    Such a mask constrains keys and never queries, so the same attention is K and V
    cropped to that run with no mask. Rows that disagree, a hole inside the run, and
    per-head or per-query structure keep the mask: no single crop expresses them.
    """
    if (attention_mask.ndim != 4 or attention_mask.shape[1] != 1
            or attention_mask.shape[-2] != 1):
        return None
    keys = attention_mask.shape[-1]
    rows = attention_mask.reshape(-1, keys)
    counts = rows.sum(-1)
    if not bool((counts == counts[0]).all()):
        return None
    valid = int(counts[0])
    if valid == 0:
        return None                      # nothing valid is the mask's problem, not a crop
    ladder = torch.arange(keys, device=rows.device) < valid
    return valid if bool((rows == ladder).all()) else None


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
    handled by the key-padding mask in ``attn_params``, or by ``key_prefix``, which
    crops K and V to the keys every query shares and leaves the call unmasked: a
    mask is an operand, not an annotation.

    ``key_prefix`` must stay a view or end in a copy. The crop leaves K's row pitch
    that of the uncropped buffer and SDPA reads through it; a GQA fold that merges
    heads into the batch with ``as_strided`` cannot consume it, because Inductor
    re-derives the folded stride from that uncropped buffer and reads past the crop,
    silently in a compiled block.

    Returns:
        Attention output tensor [B, L, H*D].
    """
    if isinstance(qkv_or_q, list):
        assert k is None and v is None, "with a [q, k, v] list, attn_params is keyword-only"
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

    if attn_params.key_prefix is not None:
        # An oversized prefix slices nothing and attends to the keys the crop replaced.
        prefix = attn_params.key_prefix
        assert 0 < prefix <= k.shape[-2], f"key_prefix {prefix} is not a crop of {k.shape[-2]} keys"
        k = k[..., :prefix, :]
        v = v[..., :prefix, :]

    # GQA: expand k/v to q's head count. ``enable_gqa`` is the same answer and no
    # cheaper: with a mask it reaches no fused kernel at all, only MATH.
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
