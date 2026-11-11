"""Native Wan schedule uses the common samplers; no second video sampling loop."""
import pytest
import torch

from thenoise.dit.wan_video.sampling import get_sigmas, FLOW_SHIFT
from thenoise.samplers import create_sampler, Step
from thenoise.models.wan_video import WanVideo
from thenoise.video.spec import VideoSpec


def test_wan_fixed_flow_shift():
    assert FLOW_SHIFT == 5.0
    sigmas = get_sigmas(4)
    assert sigmas[0] == 1.0 and sigmas[-1] == 0.0
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))
    assert WanVideo.resolve_video(VideoSpec(prompt="x", frames=10)).frame_count == 13
    with pytest.raises(ValueError, match="supports only"):
        WanVideo.resolve_video(VideoSpec(prompt="x", sampler="unknown"))


@pytest.mark.parametrize("shape", [(1, 8, 4, 6), (1, 48, 3, 4, 6)])
@pytest.mark.parametrize("sampler", ["euler", "unipc"])
def test_shared_sampler_constant_velocity_and_step_callback(shape, sampler):
    class Model:
        def begin_sampling(self, schedule):
            pass

        def denoise_step(self, x, t, cond, guidance_scale, i):
            return torch.ones_like(x)

    x = torch.zeros(shape)
    sigmas = get_sigmas(5)
    schedule = [Step(torch.tensor(s), torch.tensor(s - next_s))
                for s, next_s in zip(sigmas, sigmas[1:])]
    events = []
    out = create_sampler(sampler, Model()).sample(
        x, schedule, None, 1.0, 0,
        step_callback=lambda y, i, total: (events.append((i, total)) or y),
    )
    torch.testing.assert_close(out, -torch.ones_like(x), atol=1e-4, rtol=1e-4)
    assert events == [(i, 5) for i in range(1, 6)]


def test_step_callback_cancellation():
    class Model:
        def begin_sampling(self, schedule):
            pass
        def denoise_step(self, x, t, cond, guidance_scale, i):
            return x

    schedule = [Step(torch.tensor(1.0), torch.tensor(0.5)),
                Step(torch.tensor(0.5), torch.tensor(0.5))]
    def stop(x, i, n):
        raise RuntimeError("cancel")

    with pytest.raises(RuntimeError, match="cancel"):
        create_sampler("euler", Model()).sample(
            torch.zeros(1, 3, 4, 4), schedule, None, 1.0, 0,
            step_callback=stop,
        )


def test_comfy_checkpoints_allow_normal_model_file_paths(tmp_path):
    from thenoise.dit.wan_video.checkpoint import _find_checkpoint
    from thenoise.text_encoders.wan_video import _text_path
    from thenoise.vae.wan_video import _comfy_vae_checkpoint

    path = tmp_path / "checkpoint.safetensors"
    path.touch()
    assert _find_checkpoint(path) == path
    assert _text_path(path) == path
    assert _comfy_vae_checkpoint(path) == path
