"""What LTX's prompt masks become, and that becoming it changes no answer.

The transformer turned a prompt mask into an additive operand for every block. It now decides
once per step, where the mask can still be read, and hands the block a crop of the shared keys
or no operand at all — legal only because the two answer the same question. That agreement is
what is tested here, at the decision's own level, plus the two things the decision must not do:
invent a mask where there was none, and consume a mask that is not key padding. The crop's own
semantics, refusal table included, are ``test_attention_key_prefix.py``.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from conftest import cuda_only
from thenoise.dit.ltx_video.transformer import _prompt_params
from thenoise.utils.attention import AttentionParams, attention


def prompt_mask(keys: int, valid: int) -> torch.Tensor:
    """The ``[B, keys]`` mask a connector returns: True for the first ``valid`` tokens."""
    mask = torch.ones(1, keys, dtype=torch.bool)
    mask[:, valid:] = False
    return mask


def additive(mask: torch.Tensor) -> torch.Tensor:
    """The operand LTX built from that mask before the decision moved out of the blocks."""
    return ((1 - mask.to(torch.float32)) * -10000.0).unsqueeze(1)


def qkv(tokens: int, keys: int, heads: int = 4, width: int = 64, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, generator=g)  # noqa: E731
    return rand(1, heads, tokens, width), rand(1, heads, keys, width), rand(1, heads, keys, width)


# ------------------------------------------------------------------ the decision

def test_an_unpadded_prompt_reaches_the_block_with_no_operand():
    # The point of the change: an all-valid mask says nothing, and saying nothing is free
    # where a mask is an operand rather than an annotation.
    params = _prompt_params(prompt_mask(60, 60))
    assert params.attention_mask is None
    assert params.key_prefix is None


@pytest.mark.parametrize("valid", [1, 30, 59])
def test_a_padded_prompt_reaches_the_block_as_a_key_crop(valid):
    params = _prompt_params(prompt_mask(60, valid))
    assert params.attention_mask is None
    assert params.key_prefix == valid


def test_no_mask_at_all_still_means_no_params():
    params = _prompt_params(None)
    assert params.attention_mask is None and params.key_prefix is None


def test_a_mask_that_is_not_key_padding_keeps_the_operand_it_arrived_with():
    # Per-query structure — the i2v self-attention mask, a causal ladder — is not a crop of
    # keys. It has to reach SDPA exactly as built upstream, untouched and unconverted.
    operand = torch.zeros(1, 1, 40, 60)
    params = _prompt_params(operand)
    assert params.attention_mask is operand
    assert params.key_prefix is None


# ------------------------------------------------------------------- the agreement

@pytest.mark.parametrize("tokens,keys", [(40, 60), (128, 512)])
def test_dropping_an_all_valid_operand_changes_no_answer(tokens, keys):
    q, k, v = qkv(tokens, keys)
    valid = prompt_mask(keys, keys)
    was = attention(q, k, v, attn_params=AttentionParams(additive(valid)))
    now = attention(q, k, v, attn_params=_prompt_params(valid))
    torch.testing.assert_close(now, was, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("valid", [1, 30, 63])
def test_the_crop_answers_as_the_operand_did(valid):
    q, k, v = qkv(48, 64, seed=3)
    mask = prompt_mask(64, valid)
    was = attention(q, k, v, attn_params=AttentionParams(additive(mask)))
    now = attention(q, k, v, attn_params=_prompt_params(mask))
    torch.testing.assert_close(now, was, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("tokens,keys,valid", [(2048, 512, 512), (2048, 512, 300)])
@cuda_only
def test_at_prompt_shapes_the_two_agree_on_the_gpu_kernels(tokens, keys, valid):
    """The swap moves the call to a different kernel, which is what this tests.

    Compared against an fp32 call rather than against itself in another dtype: an agreement
    between two bf16 kernels says nothing if both are wrong by the same amount.
    """
    device = "cuda"
    q, k, v = (t.to(device=device, dtype=torch.bfloat16) for t in qkv(tokens, keys, heads=32,
                                                                      width=128, seed=5))
    mask = prompt_mask(keys, valid).to(device)
    truth = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                           attn_mask=additive(mask).float()).transpose(1, 2)
    truth = truth.flatten(2)
    was = attention(q, k, v, attn_params=AttentionParams(additive(mask))).float()
    now = attention(q, k, v, attn_params=_prompt_params(mask)).float()
    scale = truth.abs().mean().item()
    assert (now - truth).abs().max() <= 2 * (was - truth).abs().max() + 1e-3 * scale
    assert F.cosine_similarity(now.flatten(), truth.flatten(), dim=0).item() > 1 - 1e-4
