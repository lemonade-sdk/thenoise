"""Shared flow and UniPC regression tests without model weights."""
import pytest
import torch

from thenoise.samplers.flow import flow_sigmas, sample_flow
from thenoise.samplers import create_sampler, Step


def test_flow_schedule():
    assert flow_sigmas(2, 1) == [1.0, 0.5, 0.0]
    sigmas = flow_sigmas(4, 5)
    assert sigmas[0] == 1 and sigmas[-1] == 0
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))


@pytest.mark.parametrize("shape", [(1, 8, 4, 6), (1, 48, 3, 4, 6)])
@pytest.mark.parametrize("sampler", ["euler", "unipc"])
def test_shared_sampler_constant_velocity(shape, sampler):
    x = torch.zeros(shape)
    events = []
    out = sample_flow(x, flow_sigmas(5, 5), lambda y, sigma: torch.ones_like(y),
                      sampler=sampler, progress=lambda i, n: events.append((i, n)))
    torch.testing.assert_close(out, -torch.ones_like(x), atol=1e-4, rtol=1e-4)
    assert events[-1] == (5, 5)


def test_image_sampler_available():
    class Model:
        def begin_sampling(self, schedule):
            pass

        def denoise_step(self, x, t, cond, guidance_scale, step):
            return torch.ones_like(x)

    sampler = create_sampler("unipc", Model())
    schedule = [Step(torch.tensor(t), torch.tensor(d))
                for t, d in [(1., .25), (.75, .25), (.5, .25), (.25, .25)]]
    out = sampler.sample(torch.zeros(1, 3, 4, 4), schedule, None, 1.0, 0)
    torch.testing.assert_close(out, -torch.ones_like(out), atol=1e-4, rtol=1e-4)


def test_flow_cancellation():
    def stop():
        raise RuntimeError("cancel")

    with pytest.raises(RuntimeError, match="cancel"):
        sample_flow(torch.zeros(1, 3, 4, 4), flow_sigmas(3, 5),
                    lambda y, t: y, sampler="euler", cancel=stop,
                    progress=lambda *_: None)
