"""The recompile ceiling has to outlive the thread that set it.

``torch._dynamo.config`` stores an assignment as a ``ContextVar`` override, which reads
back as the stock default on any other thread. The ceiling is applied when a model
loads and consumed on whatever thread runs a compiled block — the job queue's own.
"""
from __future__ import annotations

import threading

import torch

from thenoise.utils.dynamo import RECOMPILE_LIMIT, apply_recompile_limit


def test_recompile_limit_reaches_a_fresh_thread():
    apply_recompile_limit()

    seen: dict[str, int] = {}
    thread = threading.Thread(
        target=lambda: seen.update(limit=torch._dynamo.config.recompile_limit))
    thread.start()
    thread.join()

    assert torch._dynamo.config.recompile_limit == RECOMPILE_LIMIT
    assert seen["limit"] == RECOMPILE_LIMIT
