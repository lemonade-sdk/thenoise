"""Sampler (solver) tests.

The contract shared by every solver — exactly one ``denoise_step`` per schedule
step — plus the solver-specific integration maths, on a stub model with no
weights. ``er_sde`` is a whole stochastic solver, so its determinism and its
sigma handling (the first sigma must stay strictly below 1) are pinned here.
"""
from __future__ import annotations

import pytest
import torch

from conftest import StubModel
from thenoise.models.base import DiffusionModel
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


@pytest.mark.parametrize(
    "name,cls", [("euler", EulerSampler), ("er_sde", ErSdeSampler)], ids=["euler", "er_sde"]
)
def test_create_sampler_returns_the_registered_class(name, cls):
    model = _VelocityModel()
    sampler = create_sampler(name, model)
    assert isinstance(sampler, cls)
    assert SAMPLERS[name] is cls
    assert sampler.model is model


def test_create_sampler_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown sampler: 'midpoint'"):
        create_sampler("midpoint", _VelocityModel())
    # The error names the valid choices (it is the only hint a user gets).
    with pytest.raises(ValueError, match="er_sde.*euler|euler.*er_sde"):
        create_sampler("midpoint", _VelocityModel())


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


def test_euler_integrates_the_flow_ode_in_fp32():
    """A constant velocity integrates to ``x - sum(delta) * v`` exactly."""
    steps = 4
    schedule = [
        Step(t=torch.tensor(1.0), delta=torch.tensor(1.0 / steps)) for _ in range(steps)
    ]
    x = torch.full((1, 2, 3, 3), 4.0)
    out = EulerSampler(_VelocityModel(velocity=2.0)).sample(x, schedule, None, 1.0, 0)
    assert torch.equal(out, x - 2.0)  # sum(delta) == 1.0


def test_euler_casts_back_to_the_latent_dtype():
    schedule = [Step(t=torch.tensor(1.0), delta=torch.tensor(0.5))]
    x = torch.zeros(1, 2, 3, 3, dtype=torch.bfloat16)
    out = EulerSampler(_VelocityModel()).sample(x, schedule, None, 1.0, 0)
    assert out.dtype == torch.bfloat16


def test_euler_ignores_the_seed():
    """Euler is deterministic: the same latent and schedule give the same result."""
    schedule = _schedule(5)
    x = torch.randn(1, 4, 6, 6)
    a = EulerSampler(_VelocityModel()).sample(x, schedule, None, 1.0, 1)
    b = EulerSampler(_VelocityModel()).sample(x, schedule, None, 1.0, 2)
    assert torch.equal(a, b)


def test_er_sde_is_seed_deterministic_but_seed_sensitive():
    schedule = _schedule(8)
    x = torch.randn(1, 4, 8, 8)

    def run(seed):
        return ErSdeSampler(_VelocityModel()).sample(x, schedule, None, 1.0, seed)

    assert run(5).shape == x.shape
    assert torch.equal(run(5), run(5))
    assert not torch.equal(run(5), run(6))


def test_er_sde_nudges_the_first_sigma_below_one():
    """``sigma/(1-sigma)`` blows up at sigma == 1, so t=1 goes through the model."""
    model = _VelocityModel()
    schedule = _schedule(4)
    assert float(schedule[0].t) == 1.0

    out = ErSdeSampler(model).sample(torch.randn(1, 4, 8, 8), schedule, None, 1.0, 1)

    assert model.percent_calls == [1e-4]  # nudged exactly once, for sigma[0]
    # The nudged sigma is strictly inside (0, 1) -- which is what keeps the
    # solver's ``sigma / (1 - sigma)`` term finite.
    assert 0.0 < object.__new__(StubModel).percent_to_sigma(1e-4) < 1.0
    assert torch.isfinite(out).all()


def test_er_sde_keeps_the_output_in_the_latent_dtype():
    schedule = _schedule(4)
    x = torch.randn(1, 4, 8, 8, dtype=torch.bfloat16)
    out = ErSdeSampler(_VelocityModel()).sample(x, schedule, None, 1.0, 3)
    assert out.dtype == torch.bfloat16


def test_base_percent_to_sigma_is_the_linear_fallback():
    """The fallback (models with no shifted schedule) is ``1 - percent``."""
    model = object.__new__(StubModel)
    assert isinstance(model, DiffusionModel)
    assert model.percent_to_sigma(0.0) == 1.0
    assert model.percent_to_sigma(0.25) == 0.75
    assert model.percent_to_sigma(1.0) == 0.0
