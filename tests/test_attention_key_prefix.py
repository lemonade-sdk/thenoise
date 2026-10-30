"""The key-padding mask → key crop: what it must agree with, and what it must refuse.

A crop replaces a mask only where the mask constrains keys and never queries, so the
tests are agreement with the masked call, and refusal of every mask for which the two
differ.
"""

import pytest
import torch
import torch.nn.functional as F

from thenoise.utils.attention import AttentionParams, attention, shared_key_prefix


def qkv(batch, heads, kv, qlen, klen, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, heads, qlen, 64, generator=g).to(dtype=dtype)
    k = torch.randn(batch, kv, klen, 64, generator=g).to(dtype=dtype)
    v = torch.randn(batch, kv, klen, 64, generator=g).to(dtype=dtype)
    return q, k, v


def reference(q, k, v, mask, gqa=1):
    """The masked call in fp32, GQA expanded the slow obvious way, [B, L, H*D]."""
    if gqa > 1:
        k, v = k.float().repeat_interleave(gqa, 1), v.float().repeat_interleave(gqa, 1)
    else:
        k, v = k.float(), v.float()
    x = F.scaled_dot_product_attention(q.float(), k, v, attn_mask=mask)
    x = x.transpose(1, 2)
    return x.reshape(x.shape[0], x.shape[1], -1)


def key_padding(batch, klen, valid):
    """A [B, 1, 1, klen] key-padding mask: True for the first ``valid`` keys."""
    if isinstance(valid, int):
        valid = [valid] * batch
    return torch.stack([torch.arange(klen) < v for v in valid])[:, None, None, :].bool()


# ---------------------------------------------------------------- refusal table

def test_refuses_mask_with_per_head_structure():
    assert shared_key_prefix(torch.ones(1, 4, 3, 8, dtype=torch.bool)) is None


def test_refuses_mask_with_per_query_rows():
    assert shared_key_prefix(torch.ones(1, 1, 5, 8, dtype=torch.bool)) is None


def test_refuses_rows_that_disagree():
    mask = torch.tensor([[[[1, 1, 1, 0]]], [[[1, 1, 0, 0]]]], dtype=torch.bool)
    assert shared_key_prefix(mask) is None


def test_refuses_a_hole_inside_the_run():
    assert shared_key_prefix(torch.tensor([[[[1, 0, 1]]]], dtype=torch.bool)) is None


def test_refuses_no_valid_keys():
    # Nothing to attend to is the mask's problem; a crop to zero keys is a crash.
    assert shared_key_prefix(torch.zeros(1, 1, 1, 4, dtype=torch.bool)) is None


def test_accepts_a_shared_leading_prefix():
    assert shared_key_prefix(key_padding(3, 8, 5)) == 5


def test_accepts_every_key_valid():
    assert shared_key_prefix(key_padding(2, 8, 8)) == 8


def test_no_mask_means_no_crop():
    p = AttentionParams.create_attention_params_from_mask(4, None)
    assert p.key_prefix is None and p.attention_mask is None


def test_all_valid_mask_drops_itself_rather_than_crop():
    p = AttentionParams.create_attention_params_from_mask(4, torch.ones(2, 4, dtype=torch.bool))
    assert p.key_prefix is None and p.attention_mask is None


def test_shared_prefix_becomes_a_crop_not_a_mask():
    txt = torch.tensor([[1, 1, 0]], dtype=torch.bool)          # 2 valid of 3 text tokens
    p = AttentionParams.create_attention_params_from_mask(5, txt)
    assert p.key_prefix == 7 and p.attention_mask is None


def test_ragged_batch_keeps_the_mask():
    txt = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    p = AttentionParams.create_attention_params_from_mask(5, txt)
    assert p.key_prefix is None and p.attention_mask is not None


# ------------------------------------------------------------- agreement, or not

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("heads,kv", [(4, 4), (8, 2)])
def test_crop_matches_the_masked_call(dtype, heads, kv):
    qlen, valid = 24, 19                      # query rows past the crop too
    q, k, v = qkv(1, heads, kv, qlen, qlen, dtype)
    mask = key_padding(1, qlen, valid)

    out = attention(q, k, v, AttentionParams(key_prefix=valid)).float()
    ref = reference(q, k, v, mask, heads // kv)
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(out, ref, atol=tol, rtol=tol)


def test_crop_is_not_less_accurate_than_the_mask():
    q, k, v = qkv(1, 6, 6, 32, 32, torch.bfloat16, seed=7)
    mask = key_padding(1, 32, 27)
    truth = reference(q, k, v, mask)
    e_mask = (attention(q, k, v, AttentionParams(mask)).float() - truth).abs().max()
    e_crop = (attention(q, k, v, AttentionParams(key_prefix=27)).float() - truth).abs().max()
    assert e_crop <= e_mask * 2


def test_crop_leaves_query_rows_alone():
    # A query row past the crop answers like a query row inside it: that is what makes
    # the crop legal, so it is checked on every row, not just the cropped range.
    q, k, v = qkv(2, 4, 4, 16, 16, torch.float32, seed=3)
    mask = key_padding(2, 16, 11)
    out = attention(q, k, v, AttentionParams(key_prefix=11)).float()
    torch.testing.assert_close(out, reference(q, k, v, mask), atol=1e-5, rtol=1e-5)


def test_crop_of_a_packed_view_matches_the_mask():
    # Blocks hand over K and V as transposes of packed projections.
    q, k, v = qkv(1, 4, 2, 12, 12, torch.float32, seed=5)
    k = k.transpose(1, 2).contiguous().transpose(1, 2)
    v = v.transpose(1, 2).contiguous().transpose(1, 2)
    mask = key_padding(1, 12, 9)
    out = attention(q, k, v, AttentionParams(key_prefix=9)).float()
    torch.testing.assert_close(out, reference(q, k, v, mask, 2), atol=1e-5, rtol=1e-5)


def test_pad_tail_becomes_a_crop_at_block_shapes():
    # The DiT pads the sequence and marks the tail invalid.
    qlen, valid = 20, 15
    q, k, v = qkv(1, 3, 1, qlen, qlen, torch.float32, seed=11)
    mask = key_padding(1, qlen, valid)
    assert shared_key_prefix(mask) == valid
    out = attention(q, k, v, AttentionParams(key_prefix=valid)).float()
    torch.testing.assert_close(out, reference(q, k, v, mask, 3), atol=1e-5, rtol=1e-5)
