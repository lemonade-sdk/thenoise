from __future__ import annotations

from contextlib import contextmanager

import pytest

from thenoise.stages import Stages
from thenoise.utils.pipeline_cache import PipelineCache


class _Memory:
    """The residency surface ``Stages`` uses: enter a stage, run work inside it."""

    def __init__(self):
        self.entered: list[tuple] = []
        self.ran: list[str] = []

    @contextmanager
    def stage(self, *names, pixels=None, reserve=None):
        self.entered.append((names, pixels, reserve))
        memory = self

        class _Run:
            def run(self, fn, *args, **kwargs):
                memory.ran.append("work")
                return fn(*args, **kwargs)

        yield _Run()


class _Model:
    def __init__(self):
        self.memory = _Memory()


@pytest.fixture
def stages():
    model = _Model()
    return Stages(model, PipelineCache()), model, []


def _work(sink: list, value: str):
    def run():
        sink.append("work")
        return value

    return run


def test_a_named_stage_caches_under_its_key(stages):
    runner, model, sink = stages
    first = runner.run(_work(sink, "cond"), resident=("text_encoder",), pixels=64,
                       cache="prompt", key=("p",))
    second = runner.run(_work(sink, "cond"), resident=("text_encoder",), pixels=64,
                        cache="prompt", key=("p",))

    assert first == second == "cond"
    assert sink == ["work"], "a hit must not run the work again"
    assert len(model.memory.entered) == 1, "a hit must not enter residency"


def test_a_different_key_runs_the_stage_again(stages):
    runner, model, sink = stages
    runner.run(_work(sink, "a"), resident=("dit",), pixels=64,
               cache="sampling", key=("a",))
    runner.run(_work(sink, "b"), resident=("dit",), pixels=64,
               cache="sampling", key=("b",))

    assert sink == ["work", "work"]
    assert len(model.memory.entered) == 2


def test_an_unnamed_stage_never_caches(stages):
    runner, _, sink = stages
    for _ in range(2):
        runner.run(_work(sink, "pixels"), resident=("vae",), pixels=64)

    assert sink == ["work", "work"]


def test_prepare_runs_inside_the_stage_before_the_work(stages):
    runner, model, sink = stages

    def prepare():
        sink.append("prepare")

    runner.run(_work(sink, "x"), resident=("dit",), pixels=64,
               cache="prompt", key=("p",), prepare=prepare)
    runner.run(_work(sink, "x"), resident=("dit",), pixels=64,
               cache="prompt", key=("p",), prepare=prepare)

    assert sink == ["prepare", "work"], "a hit skips preparation too"
    assert model.memory.entered == [(("dit",), 64, None)]


def test_storing_a_stage_clears_every_later_slot():
    cache = PipelineCache()
    cache.store("reference", ("r",), "ref")
    cache.store("prompt", ("p",), "cond")

    assert cache.hit("reference", ("r",))
    cache.store("reference", ("r2",), "ref")
    assert not cache.hit("prompt", ("p",)), "prompt is downstream of reference"
    assert cache.hit("reference", ("r2",))


def test_an_unknown_stage_is_named():
    cache = PipelineCache()
    with pytest.raises(ValueError, match="no cache stage 'volume' .*reference"):
        cache.store("volume", ("k",), 1)
