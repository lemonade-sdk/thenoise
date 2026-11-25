"""Sol block-sparse attention on the trunk: what it takes to run it, and with what numbers.

``comfy_kitchen.sol_attn`` routes each 64-token query block to a subset of key blocks
exactly and covers the rest with one pooled term per block, so the whole sequence still
carries the softmax denominator. The routing is training-free — adaptive thresholding on a
proxy row — which is why it can be turned on for weights that never saw it, and why the
question it raises is a quality question about a finished clip rather than a checkpoint
question.

It is on whenever it applies. There is no request for it and no way to ask for dense: a run
whose tensors fit the kernel uses it, and logs the decision once. What decides is here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from comfy_kitchen import sol_attn, sol_attn_is_available

logger = logging.getLogger(__name__)

#: The kernel's query and key block, in tokens: sinks are counted in these.
BLOCK = 64
#: The head width the fused backends are built for. The trunk is 56x128; the clip decoder
#: is 64 wide, which is why this is a trunk feature and not an engine-wide one.
HEAD_DIM = 128
#: Routing threshold in sigmas of the proxy row. Higher routes fewer key blocks exactly:
#: cheaper and less accurate. 1.3 keeps about 11 % of them.
TAU = 1.3
#: Tokens per query block routed individually rather than as part of a block — the
#: highest-scoring ones outside the routed subset, and the kernel's own default for H3.
TOKEN_AUG = 256
#: Below this the dense path is faster. The kernel names the same crossover; on gfx1151 the
#: kernel already wins at 1.3 k tokens, by 2 ms, which is not worth an opinion about.
MIN_TOKENS = 12288
#: Percents of the schedule inside which routing is used; the earliest steps stay dense,
#: where the run decides its structure. Converted to sigmas on the run's own schedule, so
#: what they protect is steps rather than a sigma no shift would reach.
WINDOW = (0.2, 1.0)
#: What the fused backends read. Every weight mode this engine runs computes in one of
#: them — an int8-convrot trunk hands attention bf16 operands — so this is a question
#: about activations, not about how the checkpoint stores its weights.
DTYPES = (torch.bfloat16, torch.float16)


@dataclass(frozen=True)
class SolAttention:
    """What one run's trunk calls the kernel with. Both sink pairs are block indices."""

    tau: float
    token_aug: int
    sink_blocks: Tuple[int, int]
    sink_q: Tuple[int, int]


def sol_sinks(layout, *, audio_rows_dense: bool = True
              ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """``(exact-KV blocks, dense-query blocks)`` for a packing, :data:`BLOCK` rows a block.

    Every row before the video span is attended exactly by every query: the prompt and the
    keyframes are what a clip is conditioned on, and pooling them into one term per block is
    how a run loses them. ``audio_rows_dense`` runs the audio *query* rows dense too, which
    costs whole query blocks doing full attention and is the reference's answer to routing
    them damaging the sound.

    No video span, or a prefix that starts at row 0, and there is nothing to protect: the
    answer is no sinks rather than a refusal.
    """
    video = layout.video_span
    if video is None or video[0] <= 0:
        return (0, 0), (0, 0)
    exact = (0, -(-video[0] // BLOCK))
    audio = layout.audio_span
    if not audio_rows_dense:
        return exact, (0, 0)
    # From the audio's first row to the end of the prefix: rounding the top up keeps the
    # partial block that holds the video's own first rows, as the reference does.
    return exact, ((audio[0] // BLOCK, exact[1]) if audio else exact)


def window_sigmas(sample_sigmas: torch.Tensor,
                  window: Tuple[float, float] = WINDOW) -> Tuple[float, float]:
    """The schedule's sigma at each end of ``window``, high first.

    Percents are read as step fractions on the run's own schedule rather than as sigmas, so
    a shifted schedule (12 for H3, which parks most steps above 0.9) is protected in the
    part of itself that is early, not in a range a shift would never visit.
    """
    last = sample_sigmas.shape[0] - 1
    read = lambda percent: float(sample_sigmas[round(percent * last)])
    return read(window[0]), read(window[1])


def sol_plan(*, layout, tokens: int, dtype: torch.dtype, head_dim: int, device,
             sigma: float, sample_sigmas: Optional[torch.Tensor],
             cached: bool) -> Optional[SolAttention]:
    """This run's kernel arguments, or ``None`` to keep the step dense.

    The whole step takes one answer, decided outside the compiled blocks: the alternative is
    reading a per-step sigma inside them, which dynamo would guard on and recompile every
    block of every step. The first transition from dense to sparse recompiles once, which is
    the cost the window buys and is paid whether or not the window exists.
    """
    def dense(reason: str) -> None:
        _log_once(reason, "dense at %d tokens: %s", tokens, reason)

    if tokens < MIN_TOKENS:
        dense(f"under {MIN_TOKENS} tokens")
        return None
    if head_dim != HEAD_DIM:
        dense(f"head_dim {head_dim} != {HEAD_DIM}")
        return None
    if dtype not in DTYPES:
        dense(f"dtype {dtype}, the fused backends take bf16 or fp16")
        return None
    if device.type != "cuda":
        dense(f"device {device}")
        return None
    if cached:
        # The KV-cache path attends a longer key than query out of a buffer the layout does
        # not describe, so there is no honest sink range to give the kernel.
        dense("the K/V cache holds this call's keys")
        return None
    if layout.seq_len != tokens:
        # Sinks are the whole reason a sparse run of this model is trustworthy, and a
        # layout that does not describe the call cannot say where they are.
        dense(f"layout describes {layout.seq_len} rows, the call has {tokens}")
        return None
    if sample_sigmas is None:
        dense("no schedule to place the start/end window on")
        return None
    high, low = window_sigmas(sample_sigmas)
    if sigma > high or sigma < low:
        dense(f"sigma {sigma:.3g} outside the {WINDOW[0]}-{WINDOW[1]} window")
        return None
    if not sol_attn_is_available(device):
        dense("no compiled sol_attn kernel for this GPU")
        return None
    sinks, rows = sol_sinks(layout)
    _log_once("sparse", "sol attention: %d tokens, tau %g, sinks %s/%s", tokens, TAU,
              sinks, rows)
    return SolAttention(TAU, TOKEN_AUG, sinks, rows)


@torch._dynamo.disable()
def sol_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  plan: SolAttention) -> torch.Tensor:
    """The kernel, on ``[B, T, H, D]`` operands, returning the same order.

    An island of its own because it is an external kernel the compiled block has no reason
    to trace, and because its arguments are block counts in Python.

    The operands arrive as the fused ``qkv_proj``'s three slices, so a token's pitch is
    ``3 * H * D`` rather than ``H * D``. The kernel reads that: strided and packed operands
    come back bit-identical on gfx1151, at a length that is not a multiple of the block and
    a length that is. Nothing is copied on the way in, which the dense path does not get to
    say about itself — see :func:`thenoise.utils.attention.uniform_layout`.
    """
    return sol_attn(q, k, v, tau=plan.tau, sink_blocks=list(plan.sink_blocks),
                    sink_q=list(plan.sink_q), token_aug=plan.token_aug)


_LOGGED = set()
_LOG_LIMIT = 32


def _log_once(key: str, message: str, *args) -> None:
    """One line per distinct reason per process, so a dense run says why once and not 1000x.

    A run that flips between dense and sparse every step produces a bounded number of keys
    either way (reason, token count); the cap is only so a pathological loop cannot grow it.
    """
    if key in _LOGGED or len(_LOGGED) >= _LOG_LIMIT:
        return
    _LOGGED.add(key)
    logger.info(message, *args)


def reset_log() -> None:
    """Forget what has been reported. For tests, and for a second run in one process."""
    _LOGGED.clear()
