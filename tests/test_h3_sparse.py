"""Sol attention on the trunk: when a run uses it, what it is called with, and whether the
operands that reach the kernel are the ones the dense path would have been handed.

The kernel itself is not here: it is a compiled GPU thing, and every assertion below holds
without it. What is worth pinning weightlessly is the policy — the sinks, the refusals —
and the two things a branch in the middle of every block's attention could silently break:
that dense still gets head-major operands, and that the sparse branch rotates its queries
the same way from a table shaped for the other order.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import thenoise.dit.h3.av as av
import thenoise.dit.h3.sparse as sparse
import thenoise.dit.h3.trunk as trunk
from test_dit_h3_av import _av_dit, _av_layout, _run
from thenoise.dit.h3.layout import Layout
from thenoise.dit.h3.sparse import (BLOCK, DTYPES, HEAD_DIM, MIN_TOKENS, TAU, TOKEN_AUG,
                                    WINDOW, SolAttention, sol_plan, sol_sinks,
                                    window_sigmas)

CUDA = SimpleNamespace(type="cuda")
CPU = SimpleNamespace(type="cpu")
#: A schedule 20 steps deep, descending, which is what the window percents are read on.
SCHEDULE = torch.linspace(1.0, 0.0, 20)


def _lay(**over) -> Layout:
    """A packing wide enough to matter: prompt, two keyframes, audio, video."""
    opts = dict(text_len=200, latent_t=4, latent_h=8, latent_w=8, audio_t=7,
                keyframes=[0, 12])
    return Layout.av(**{**opts, **over})


@pytest.fixture(autouse=True)
def _gate_small_enough(monkeypatch):
    """A test pack is a few hundred rows, so the crossover is lowered out of the way.

    Refusing it is a case of its own, and it reads the patched value so the gate itself is
    still what is being exercised.
    """
    monkeypatch.setattr(sparse, "MIN_TOKENS", 8)


@pytest.fixture(autouse=True)
def _fresh_log():
    sparse.reset_log()


@pytest.fixture
def kernel_present(monkeypatch):
    monkeypatch.setattr(sparse, "sol_attn_is_available", lambda device: True)


def _plan(**over) -> SolAttention:
    """The plan for a run that passes every gate, with these few details changed."""
    lay = over.pop("layout", _lay())
    args = dict(layout=lay, tokens=lay.seq_len, dtype=torch.bfloat16, device=CUDA,
                head_dim=HEAD_DIM, sigma=float(SCHEDULE[5]), sample_sigmas=SCHEDULE,
                cached=False)
    return sol_plan(**{**args, **over})


# ---------------------------------------------------------------- sink policy


def test_every_row_before_the_video_is_exact_kv_and_the_audio_rows_run_dense():
    lay = _lay()
    exact, dense_rows = sol_sinks(lay)
    assert exact == (0, -(-lay.video_span[0] // BLOCK))
    assert dense_rows == (lay.audio_span[0] // BLOCK, exact[1])
    # The exact range is whole blocks and reaches into the block holding the video's own
    # first rows: conditioning that shares a block with the target is not left pooled.
    assert exact[1] * BLOCK >= lay.video_span[0]


def test_a_still_protects_its_prefix_without_any_audio_rows():
    lay = Layout.image(text_len=300, latent_h=64, latent_w=64)
    exact, dense_rows = sol_sinks(lay)
    assert lay.audio_span is None
    assert exact == (0, -(-lay.video_span[0] // BLOCK))
    assert dense_rows == exact


def test_the_audio_query_rows_can_be_asked_for_apart_from_the_prefix():
    lay = _lay()
    assert sol_sinks(lay, audio_rows_dense=False) == (sol_sinks(lay)[0], (0, 0))


def test_a_packing_that_starts_with_the_video_has_nothing_to_protect():
    class NoPrefix:
        video_span = (0, 640)
        audio_span = None

    assert sol_sinks(NoPrefix()) == ((0, 0), (0, 0))


# ------------------------------------------------------------------- the window


def test_the_window_is_read_on_the_schedule_the_run_is_stepping_through():
    high, low = window_sigmas(SCHEDULE)
    last = SCHEDULE.shape[0] - 1
    assert high == pytest.approx(float(SCHEDULE[round(WINDOW[0] * last)]))
    assert low == pytest.approx(float(SCHEDULE[round(WINDOW[1] * last)]))
    assert high > low


def test_a_two_step_schedule_leaves_every_step_inside_the_window():
    """Two sigmas are one step: there is no early part of it to hold dense.

    The window protects the first fraction of a run, and a run this short has nothing to
    protect — the endpoints meet, so nothing is excluded rather than everything.
    """
    assert window_sigmas(torch.tensor([1.0, 0.0])) == (1.0, 0.0)


# ------------------------------------------------------------------- the gate


def test_a_long_bf16_clip_on_a_gpu_with_the_kernel_goes_sparse(kernel_present):
    assert _plan() == SolAttention(TAU, TOKEN_AUG, *sol_sinks(_lay()))


def test_fp16_operands_go_sparse_too(kernel_present):
    # Every weight mode this engine runs computes in bf16 or fp16, so an int8-convrot
    # trunk arrives here the same way a bf16 one does.
    assert _plan(dtype=torch.float16) is not None


@pytest.mark.parametrize("change, why", [
    (dict(tokens=sparse.MIN_TOKENS - 1), "below the crossover"),
    (dict(head_dim=64), "the clip decoder's head width"),
    (dict(dtype=torch.float32), "not a dtype the fused backends read"),
    (dict(device=CPU), "no kernel off CUDA"),
    (dict(cached=True), "the K/V cache holds the keys"),
    (dict(sample_sigmas=None), "no schedule to place the window on"),
    (dict(sigma=float(window_sigmas(SCHEDULE)[0]) + 0.05), "before the window"),
    (dict(sigma=float(window_sigmas(SCHEDULE)[1]) - 0.05), "after the window"),
])
def test_what_keeps_a_step_dense(kernel_present, change, why):
    assert _plan(**change) is None, why


def test_a_layout_that_does_not_describe_the_call_keeps_it_dense(kernel_present):
    """Refusing beats running sparse with no idea where the conditioning is."""
    lay = _lay()
    assert _plan(layout=lay, tokens=lay.seq_len) is not None
    assert _plan(layout=lay, tokens=lay.seq_len - 1) is None


def test_no_compiled_kernel_means_dense(monkeypatch):
    monkeypatch.setattr(sparse, "sol_attn_is_available", lambda device: False)
    assert _plan() is None


def test_the_defaults_are_the_releases():
    """Pinned to the reference's, so a local edit is a decision and not a slip."""
    assert (TAU, TOKEN_AUG, MIN_TOKENS, WINDOW) == (1.3, 256, 12288, (0.2, 1.0))
    assert (HEAD_DIM, BLOCK) == (128, 64)
    assert DTYPES == (torch.bfloat16, torch.float16)


def test_a_run_says_why_once(kernel_present, caplog):
    caplog.set_level("INFO")
    for _ in range(40):
        assert _plan(tokens=sparse.MIN_TOKENS - 1) is None
    assert len([r for r in caplog.records if "dense at" in r.getMessage()]) == 1
    sparse.reset_log()
    for _ in range(40):
        _plan()
    assert len([r for r in caplog.records if "sol attention" in r.getMessage()]) == 1
    sparse.reset_log()


# ---------------------------------------------------------------- the plumbing


HIDDEN, HEADS, HEAD_DIM_TINY, TOKENS = 32, 2, 8, 12
ROT = 3   # a rope narrower than the head, so the idle tail is part of the check


def _attention() -> trunk.Attention:
    torch.manual_seed(3)
    return trunk.Attention(HIDDEN, HEADS, HEAD_DIM_TINY, eps=1e-6).eval().requires_grad_(False)


def _rope():
    return (torch.randn(1, 1, TOKENS, ROT), torch.randn(1, 1, TOKENS, ROT))


class Backend:
    """Stands in for an attention backend, keeping what it was handed.

    Both answer token-major, which is what the block's output projection reads: SDPA
    returns ``[B, L, H*D]``, the kernel ``[B, T, H, D]``.
    """

    def __init__(self, token_major: bool):
        self.args = None
        self.token_major = token_major

    def __call__(self, q, k, v, *rest):
        self.args = (q, k, v) + tuple(rest)
        if self.token_major:
            return torch.zeros_like(q)
        return torch.zeros(q.shape[0], q.shape[2], q.shape[1] * q.shape[3])


def test_dense_hands_the_backend_head_major_operands(monkeypatch):
    backend = Backend(token_major=False)
    monkeypatch.setattr(trunk, "attend", backend)
    out = _attention()(torch.randn(1, TOKENS, HIDDEN), _rope())
    q, k, v = backend.args[:3]
    assert q.shape == (1, HEADS, TOKENS, HEAD_DIM_TINY)
    assert out.shape == (1, TOKENS, HIDDEN)


def test_the_sparse_branch_reaches_the_kernel_token_major(monkeypatch):
    backend = Backend(token_major=True)
    monkeypatch.setattr(trunk, "sol_attention", backend)
    plan = SolAttention(TAU, TOKEN_AUG, (0, 1), (0, 1))
    _attention()(torch.randn(1, TOKENS, HIDDEN), _rope(), None, plan)
    q, k, v, handed = backend.args
    assert handed is plan
    assert q.shape == (1, TOKENS, HEADS, HEAD_DIM_TINY)


def test_the_two_branches_rotate_the_same_queries(monkeypatch):
    """The kernel reads ``[B, T, H, D]``, so the rope table is turned to meet it.

    Same weights, same input: the only difference allowed between the branches is the
    order the operands are written in.
    """
    dense, sparsed = Backend(token_major=False), Backend(token_major=True)
    monkeypatch.setattr(trunk, "attend", dense)
    monkeypatch.setattr(trunk, "sol_attention", sparsed)
    x, rope = torch.randn(1, TOKENS, HIDDEN), _rope()
    _attention()(x, rope)
    _attention()(x, rope, None, SolAttention(TAU, TOKEN_AUG, (0, 1), (0, 1)))
    for i in (0, 1, 2):
        assert torch.equal(dense.args[i].transpose(1, 2), sparsed.args[i])


# ---------------------------------------------------------------- the trunks


def test_the_av_trunk_hands_the_steps_plan_to_every_block(monkeypatch):
    dit, lay = _av_dit(), _av_layout()
    plan = SolAttention(TAU, TOKEN_AUG, (0, 2), (1, 2))
    seen = []

    def kernel(q, k, v, handed):
        seen.append((q.shape[1], handed))
        return torch.zeros_like(q)

    monkeypatch.setattr(av, "sol_plan", lambda **kw: plan)
    monkeypatch.setattr(trunk, "sol_attention", kernel)
    _run(dit, lay)
    assert len(seen) == dit.params.num_layers
    assert {handed for _, handed in seen} == {plan}
    assert {tokens for tokens, _ in seen} == {lay.seq_len}


def test_a_dense_run_never_reaches_the_kernel(monkeypatch):
    """A CPU run, so the gate refuses and the branch inside the block is inert."""
    dit, lay = _av_dit(), _av_layout()
    reached = []
    monkeypatch.setattr(trunk, "sol_attention",
                        lambda q, k, v, plan: reached.append(q) or torch.zeros_like(q))
    out = _run(dit, lay)
    assert not reached
    assert torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all()
