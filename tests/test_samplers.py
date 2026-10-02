"""Sampler (solver) tests.

The contract shared by every solver — exactly one ``denoise_step`` per schedule step —
plus the solver-specific integration maths, on a stub model with no weights.
"""
from __future__ import annotations

import pytest
import torch

from conftest import StubModel
from thenoise.pipeline import _sigma_steps
from thenoise.samplers import SAMPLERS, Step, create_sampler
from thenoise.samplers.er_sde import ErSdeSampler
from thenoise.samplers.euler import EulerSampler


class _VelocityModel(StubModel):
    """Stub returning a constant velocity and logging the timesteps it saw."""

    velocity = 1.0

    def __init__(self, *args, velocity=None, **kwargs):
        super().__init__(*args, **kwargs)
        if velocity is not None:
            self.velocity = velocity
        self.timesteps = []
        self.percent_calls = []

    def denoise_step(self, latents, t, cond, guidance_scale, i):
        self.calls["denoise_step"] += 1
        self.timesteps.append(float(t))
        return torch.full_like(latents, self.velocity)

    def percent_to_sigma(self, percent):
        self.percent_calls.append(percent)
        return super().percent_to_sigma(percent)


def _schedule(steps=8):
    grid = torch.linspace(1.0, 0.0, steps + 1)
    return [Step(t=grid[i], delta=grid[i] - grid[i + 1]) for i in range(steps)]


def test_create_sampler_binds_the_registered_class_or_rejects_the_name():
    model = _VelocityModel()
    for name, cls in [("euler", EulerSampler), ("er_sde", ErSdeSampler)]:
        sampler = create_sampler(name, model)
        assert isinstance(sampler, cls)
        assert sampler.model is model

    # The message names the valid choices: it is the only hint a user gets.
    with pytest.raises(ValueError, match="unknown sampler: 'midpoint'.*(er_sde|euler)"):
        create_sampler("midpoint", model)


@pytest.mark.parametrize("name", sorted(SAMPLERS))
def test_one_denoise_step_per_schedule_step(name):
    """The shared contract: N schedule steps -> exactly N ``denoise_step`` calls."""
    model = _VelocityModel()
    schedule = _schedule(6)
    out = create_sampler(name, model).sample(
        torch.randn(1, 4, 8, 8), schedule, None, guidance_scale=1.0, seed=11
    )
    assert model.calls["denoise_step"] == len(schedule)
    expected = [float(step.t) for step in schedule]
    if name == "er_sde":
        # ER-SDE hands the model its own nudged sigma_0 (see percent_to_sigma);
        # every later step is the schedule's own value.
        assert model.timesteps[0] < expected[0]
        assert model.timesteps[1:] == expected[1:]
    else:
        # Euler walks the schedule in order, starting at the first timestep.
        assert model.timesteps == expected
    assert out.shape == (1, 4, 8, 8)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", sorted(SAMPLERS))
def test_solvers_accept_an_arbitrary_custom_sigma_grid(name):
    """User sigmas need neither uniform spacing nor a full-strength start."""
    model = _VelocityModel()
    schedule = _sigma_steps([0.9, 0.7, 0.65, 0.2, 0.0], "cpu", torch.float32)

    out = create_sampler(name, model).sample(
        torch.randn(1, 4, 8, 8), schedule, None, guidance_scale=1.0, seed=7
    )

    assert model.calls["denoise_step"] == 4
    assert out.shape == (1, 4, 8, 8)
    assert torch.isfinite(out).all()


def test_euler_integrates_the_flow_ode_in_fp32():
    """A constant velocity integrates to ``x - sum(delta) * v`` exactly, and the seed
    is ignored: Euler is deterministic.
    """
    steps = 4
    schedule = [
        Step(t=torch.tensor(1.0), delta=torch.tensor(1.0 / steps)) for _ in range(steps)
    ]
    x = torch.full((1, 2, 3, 3), 4.0)
    a = EulerSampler(_VelocityModel(velocity=2.0)).sample(x, schedule, None, 1.0, 0)
    assert torch.equal(a, x - 2.0)  # sum(delta) == 1.0

    b = EulerSampler(_VelocityModel(velocity=2.0)).sample(x, schedule, None, 1.0, 99)
    assert torch.equal(a, b)


@pytest.mark.parametrize("cls", [EulerSampler, ErSdeSampler], ids=["euler", "er_sde"])
def test_the_solver_returns_the_latent_dtype(cls):
    x = torch.zeros(1, 2, 3, 3, dtype=torch.bfloat16)
    schedule = _schedule(2)
    out = cls(_VelocityModel()).sample(x, schedule, None, 1.0, 0)
    assert out.dtype == torch.bfloat16


def test_er_sde_is_seed_deterministic_but_seed_sensitive():
    schedule = _schedule(8)
    x = torch.randn(1, 4, 8, 8)

    def run(seed):
        return ErSdeSampler(_VelocityModel()).sample(x, schedule, None, 1.0, seed)

    assert run(5).shape == x.shape
    assert torch.equal(run(5), run(5))
    assert not torch.equal(run(5), run(6))


def test_er_sde_nudges_the_first_sigma_below_one():
    """``sigma/(1 - sigma)`` blows up at sigma == 1, so t=1 goes through the model's
    nudge rather than into the solver.
    """
    model = _VelocityModel()
    schedule = _schedule(4)
    assert float(schedule[0].t) == 1.0

    out = ErSdeSampler(model).sample(torch.randn(1, 4, 8, 8), schedule, None, 1.0, 1)

    assert model.percent_calls == [1e-4]  # nudged exactly once, for sigma[0]
    assert 0.0 < object.__new__(StubModel).percent_to_sigma(1e-4) < 1.0
    assert torch.isfinite(out).all()


