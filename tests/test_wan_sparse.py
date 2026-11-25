"""Wan's own sparse policy: the gates it shares, the range it owns, where the plan lands.

The kernel's rules and the kept-fraction readout are tested in ``test_h3_sparse.py`` beside
the module that owns them; what is asserted here is Wan's contribution — the first frame as
the only sink, a window read through the run's flow shift — and the trunk wiring that decides
self-attention routes and the text branch does not.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import thenoise.dit.wan_video.model as wan_model
import thenoise.dit.wan_video.sparse as wan_sparse
import thenoise.models.wan_video as wan_adapter_module
from thenoise.dit.wan_video.model import (Wan22Params, WanNativeAttention, WanNativeBlock,
                                          WanNativeDiT)
from thenoise.dit.wan_video.sparse import (TAU, WINDOW, wan_plan, wan_sinks,
                                          window_sigmas)
from thenoise.utils.sol_attention import (MIN_TOKENS, TOKEN_AUG_MAX, SolAttention,
                                          reset_log)

CUDA = SimpleNamespace(type="cuda")
CPU = SimpleNamespace(type="cpu")
HEADS = 2
DIM = HEADS * 128
#: Sigmas routing is confined to, high first: what ``window_sigmas`` answers for a shift.
SIGMA_WINDOW = (0.6, 0.0)


@pytest.fixture(autouse=True)
def _fresh_log():
    reset_log()


def _plan(**over):
    """The plan for a clip that passes every gate, with these few details changed."""
    args = dict(tokens=MIN_TOKENS, tokens_per_frame=1600, dtype=torch.bfloat16,
                head_dim=128, device=CUDA, sigma=0.3, window=SIGMA_WINDOW)
    return wan_plan(**{**args, **over})


# ---------------------------------------------------------------- the policy


def test_a_long_bf16_clip_on_a_gpu_with_the_kernel_routes(monkeypatch):
    monkeypatch.setattr(wan_sparse, "available", lambda device: True)
    assert _plan() == SolAttention(TAU, TOKEN_AUG_MAX, (0, 25))


@pytest.mark.parametrize("change, why", [
    (dict(tokens=MIN_TOKENS - 1), "below the crossover"),
    (dict(head_dim=64), "the codec's head width"),
    (dict(dtype=torch.float32), "not a dtype the fused backends read"),
    (dict(device=CPU), "no kernel off CUDA"),
])
def test_a_clip_that_cannot_route_is_refused(change, why, monkeypatch):
    monkeypatch.setattr(wan_sparse, "available", lambda device: True)
    assert _plan(**change) is None, why


def test_no_compiled_kernel_means_dense(monkeypatch):
    monkeypatch.setattr(wan_sparse, "available", lambda device: False)
    assert _plan() is None


def test_the_earliest_steps_stay_dense(monkeypatch):
    """The window is read in sigmas here, because a shifted flow schedule is what Wan has."""
    monkeypatch.setattr(wan_sparse, "available", lambda device: True)
    assert _plan(sigma=SIGMA_WINDOW[0]) is not None
    assert _plan(sigma=SIGMA_WINDOW[1]) is not None
    assert _plan(sigma=SIGMA_WINDOW[0] + 1e-6) is None


def test_the_window_is_placed_by_the_runs_own_shift():
    """The adapter supplies the schedule; the percents stay this module's to own."""
    assert window_sigmas(lambda percent: 1.0 - percent) == (0.8, 0.0)
    assert window_sigmas(lambda percent: 1.0 - percent, window=(0.3, 1.0)) == (0.7, 0.0)
    assert WINDOW == (0.2, 1.0)


def test_the_first_frame_is_the_only_sink():
    """One frame of exact keys, however the frame is sized: rounded up to whole blocks.

    A frame smaller than a block still gets one, which is the kernel's granularity rather
    than a choice: it cannot attend half a block exactly.
    """
    assert wan_sinks(1600) == (0, 25)
    assert wan_sinks(64) == (0, 1)
    assert wan_sinks(65) == (0, 2)


# ---------------------------------------------------------------- the trunk


def _attn() -> WanNativeAttention:
    torch.manual_seed(5)
    return WanNativeAttention(DIM, HEADS, eps=1e-6).eval().requires_grad_(False)


TOKENS = 12
ROPE = torch.randn(1, TOKENS, 64, 2, 2)


def test_the_routed_branch_hands_the_kernel_token_major_operands(monkeypatch):
    seen = []

    def kernel(q, k, v, plan):
        seen.append((q, k, v, plan))
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2] * q.shape[3])

    monkeypatch.setattr(wan_model, "sol_attention", kernel)
    x = torch.randn(1, TOKENS, DIM)
    plan = SolAttention(TAU)
    out = _attn()(x, None, ROPE, plan)
    q, k, v, handed = seen[0]
    assert handed is plan
    assert q.shape == v.shape == torch.Size([1, TOKENS, HEADS, 128])
    assert out.shape == (1, TOKENS, DIM)


def test_dense_still_hands_sdpa_head_major_operands(monkeypatch):
    seen = []

    def backend(qkv):
        seen.extend(qkv)
        return torch.zeros(qkv[0].shape[0], qkv[0].shape[2], qkv[0].shape[1] * qkv[0].shape[3])

    monkeypatch.setattr(wan_model, "attention", backend)
    _attn()(torch.randn(1, TOKENS, DIM), None, ROPE)
    assert seen[0].shape == (1, HEADS, TOKENS, 128)


def test_cross_attention_is_never_offered_the_kernel(monkeypatch):
    """A plan reaching the text branch is refused, not obeyed: 226 prompt tokens are not a
    sequence worth pooling, and its sinks would be the wrong rows entirely."""
    routed = []
    monkeypatch.setattr(wan_model, "sol_attention",
                        lambda *a: routed.append(a) or torch.zeros(1, TOKENS, DIM))
    dense = []
    monkeypatch.setattr(wan_model, "attention",
                        lambda qkv: dense.append(qkv) or torch.zeros(1, TOKENS, DIM))
    attn = _attn()
    attn(torch.randn(1, TOKENS, DIM), torch.randn(1, 40, DIM), None, SolAttention(TAU))
    assert not routed and len(dense) == 1


def test_both_branches_rotate_the_same_queries(monkeypatch):
    """Token-major with the rope on the head axis is the same arithmetic, not an approximation.

    The claim is exact equality rather than closeness: nothing about routing a sequence should
    move a value that both branches compute.
    """
    captured = {}
    monkeypatch.setattr(wan_model, "attention",
                        lambda qkv: captured.__setitem__("dense", qkv[0].transpose(1, 2))
                        or torch.zeros(1, TOKENS, DIM))
    monkeypatch.setattr(wan_model, "sol_attention",
                        lambda q, k, v, plan: captured.__setitem__("routed", q)
                        or torch.zeros(1, TOKENS, HEADS * 128))
    attn = _attn()
    x = torch.randn(1, TOKENS, DIM)
    attn(x, None, ROPE)
    attn(x, None, ROPE, SolAttention(TAU))
    assert torch.equal(captured["dense"], captured["routed"])


def test_the_block_gives_the_plan_to_self_attention_only(monkeypatch):
    params = _params(layers=1)
    block = _block(params)
    seen = {}
    monkeypatch.setattr(block.attn1, "forward",
                        lambda x, context, rope, sol=None: seen.__setitem__("self", sol) or x)
    monkeypatch.setattr(block.attn2, "forward",
                        lambda x, context, rope, sol=None: seen.__setitem__("cross", sol) or x)
    plan = SolAttention(TAU)
    # The suite runs ``torch.compile`` as a no-op (tests/conftest.py), so this is the function
    # dynamo would be handed — the branch under test is inside it, before any kernel call.
    WanNativeBlock.forward(block, torch.zeros(1, 4, params.dim),
                           torch.zeros(1, 6, params.dim),
                           torch.zeros(1, 6, params.dim),
                           torch.zeros(1, 1, 4, 64, 2, 2), plan)
    assert seen["self"] is plan
    assert seen["cross"] is None


def _params(layers: int = 1) -> Wan22Params:
    return Wan22Params(in_channels=16, out_channels=16, dim=DIM, num_layers=layers,
                       ffn_dim=DIM, freq_dim=32, text_dim=64, patch_size=(1, 2, 2))


def _block(params: Wan22Params):
    block = wan_model.WanNativeBlock(params)
    block.eval().requires_grad_(False)
    return block


def test_the_trunk_hands_one_plan_to_every_block(monkeypatch):
    params = _params(layers=3)
    dit = WanNativeDiT(params).eval().requires_grad_(False)
    seen = []
    for block in dit.blocks:
        monkeypatch.setattr(block, "forward",
                            lambda *a, **kw: seen.append(kw.get("sol", a[-1])) or a[0])
    plan = SolAttention(TAU)
    dit(torch.zeros(1, 16, 2, 4, 4), torch.zeros(1), torch.zeros(1, 6, params.dim), plan)
    assert len(seen) == 3 and all(handed is plan for handed in seen)


# ---------------------------------------------------------------- the adapter


class _Shift:
    """An adapter's own sigma axis: the flow shift is its business, not the trunk's."""

    def percent_to_sigma(self, percent: float) -> float:
        return 1.0 - percent


def test_the_adapter_counts_the_tokens_the_trunk_will_see(monkeypatch):
    seen = {}
    monkeypatch.setattr(wan_adapter_module, "wan_plan",
                        lambda **kw: seen.update(kw) or None)
    module = SimpleNamespace(params=SimpleNamespace(patch_size=(1, 2, 2), head_dim=128))
    # 2 latent frames, 4x4 latents, patch (1,2,2): 2 * 2 * 2 tokens, 4 of them per frame.
    wan_adapter_module.Wan22Model._sol(_Shift(), module, torch.zeros(1, 16, 2, 4, 4), 0.5)
    assert seen["tokens"] == 8
    assert seen["tokens_per_frame"] == 4
    assert seen["sigma"] == 0.5
    assert seen["window"] == (0.8, 0.0)
    assert seen["head_dim"] == 128


def test_a_short_clip_is_never_routed_by_the_adapter(monkeypatch):
    """Nothing is plumbed in to say so: the floor is reached through the same gate."""
    monkeypatch.setattr(wan_adapter_module, "window_sigmas", lambda read: (1.0, 0.0))
    module = SimpleNamespace(params=SimpleNamespace(patch_size=(1, 2, 2), head_dim=128))
    assert wan_adapter_module.Wan22Model._sol(_Shift(), module,
                                              torch.zeros(1, 16, 2, 4, 4), 0.5) is None
