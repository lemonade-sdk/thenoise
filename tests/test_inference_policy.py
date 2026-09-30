"""The inference-only policy: one mechanism, declared at the entry points only."""

import pytest
import torch
from torch import nn
from PIL import Image

import thenoise
import thenoise.models as model_catalog
from conftest import StubLatentUpscaler, StubModel
from thenoise.inference import freeze, inference, inference_lock
from thenoise.memory import MemoryManager
from thenoise.models.config import GenerateRequest
from thenoise.pipeline import PipelineController
from thenoise.runtime import ModelPaths, Runtime, Settings
from thenoise.upscale.pixel import PixelUpscalerManager
from thenoise.upscale_controller import PixelUpscaleController

class _SpyingUpscaler(StubLatentUpscaler):
    """Latent upscaler that records the grad mode of the upscale stage."""

    def __init__(self, scale, model):
        super().__init__(scale)
        self._model = model

    def __call__(self, latents):
        self._model.spy("upscale", latents)
        return super().__call__(latents)


class _SpyingModel(StubModel):
    """Stub adapter that records the grad mode each of its kernels runs in."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_states: dict[str, bool] = {}
        self.inference_states: dict[str, bool] = {}

    def spy(self, where: str, *tensors) -> None:
        self.grad_states[where] = torch.is_grad_enabled()
        if tensors:
            self.inference_states[where] = all(
                torch.is_inference(t) for t in tensors
            )

    def encode_prompt(self, args):
        self.spy("encode_prompt")
        return super().encode_prompt(args)

    def prepare_latent(self, latents, cond, params, ref=None, ref_method="index"):
        self.spy("prepare_latent", latents)
        return super().prepare_latent(latents, cond, params, ref, ref_method)

    def denoise_step(self, latents, t, cond, guidance_scale, i):
        self.spy("denoise_step", latents)
        return super().denoise_step(latents, t, cond, guidance_scale, i)

    def encode_reference(self, pixels):
        self.spy("encode_reference", pixels)
        return super().encode_reference(pixels)

    def decode(self, latents):
        self.spy("decode", latents)
        return super().decode(latents)

    def _create_upscaler(self):
        return _SpyingUpscaler(self.UPSCALE_SCALE, self)


class _SpyingController(PipelineController):
    """Controller that records the grad mode of the post-decode tail."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail_states: list[bool] = []

    def postprocess(self, pixels, **kwargs):
        self.tail_states.append(torch.is_grad_enabled())
        return super().postprocess(pixels, **kwargs)


def _pipeline(model):
    return _SpyingController(model, PixelUpscalerManager(upscaler_dir="", device="cpu"))


def _request(**kwargs) -> GenerateRequest:
    return GenerateRequest(**{"prompt": "a fox", "seed": 1, **kwargs})


def test_generate_runs_every_stage_and_the_tail_in_inference_mode():
    model = _SpyingModel()
    controller = _pipeline(model)

    controller.generate(_request())

    assert model.grad_states == {
        "encode_prompt": False,
        "prepare_latent": False,
        "denoise_step": False,
        "decode": False,
    }
    # Nothing is differentiable, and what reaches the next stage is already an
    # inference tensor, so the pipeline cache never mixes modes across requests.
    assert model.inference_states == {
        "prepare_latent": True,
        "denoise_step": True,
        "decode": True,
    }
    # The post-decode tail (notch filter, pixel upscaler, resize, postprocess,
    # PIL) runs inside the same boundary.
    assert controller.tail_states and all(not state for state in controller.tail_states)


def test_the_boundary_is_scoped_to_the_request():
    """A boundary is not a global toggle: the caller keeps its own grad mode."""
    controller = _pipeline(_SpyingModel())
    controller.generate(_request())
    assert torch.is_grad_enabled() is True


def test_edit_runs_the_reference_stage_in_inference_mode():
    model = _SpyingModel()
    model.CAPABILITIES = {**model.CAPABILITIES, "edit": True}

    _pipeline(model).edit(_request(image=Image.new("RGB", (64, 64))))

    assert model.grad_states["encode_reference"] is False
    assert model.inference_states["encode_reference"] is True
    assert model.grad_states["denoise_step"] is False


def test_upscale_and_refine_run_in_inference_mode():
    model = _SpyingModel()

    _pipeline(model).generate(_request(upscale=True))

    assert model.grad_states["upscale"] is False
    # The refine denoises on top of the upscaled latent.
    assert model.grad_states["denoise_step"] is False


def test_pixel_upscale_endpoint_is_its_own_boundary(monkeypatch, tmp_path):
    """The model-free ``/upscale`` path never touches the pipeline."""
    seen: list[bool] = []

    class _FakeUpscaler:
        def forward_tiled(self, x):
            seen.append(torch.is_grad_enabled())
            return torch.zeros(
                1, 3, x.shape[-2] * 2, x.shape[-1] * 2, device=x.device
            )

    (tmp_path / "x2.safetensors").write_text("x")
    manager = PixelUpscalerManager(upscaler_dir=str(tmp_path), device="cpu")
    manager._pixel_upscaler_scales = {"x2": 2}
    monkeypatch.setattr(
        "thenoise.upscale.pixel.load_pixel_upscaler",
        lambda path, device: (_FakeUpscaler(), 2),
    )

    PixelUpscaleController(manager).upscale(
        Image.new("RGB", (8, 8)), upscale_factor=2.0, pixel_upscaler="x2"
    )

    assert seen and all(state is False for state in seen)
    assert torch.is_grad_enabled() is True


def test_model_load_runs_in_inference_mode(monkeypatch, fake_model_cls, tmp_path):
    seen: list[bool] = []

    def __init__(self, **kwargs):
        seen.append(torch.is_grad_enabled())
        # Weights built here are inference tensors for the life of the model.
        seen.append(torch.is_inference(torch.zeros(1)))

    cls = fake_model_cls(name="fake", __init__=__init__)
    monkeypatch.setattr(model_catalog, "MODEL_CATALOG", [cls])
    monkeypatch.setattr(model_catalog, "resolve", lambda path: cls)

    runtime = Runtime(Settings(device="cpu", offload_device="cpu"))
    runtime.load(
        ModelPaths(
            dit_path=str(tmp_path / "dit.safetensors"),
            vae_path=str(tmp_path / "vae.safetensors"),
            text_encoder_path=str(tmp_path / "te.safetensors"),
        )
    )

    assert seen == [False, True]


# ------------------------------------------------------------------ the helpers
class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.bn = nn.BatchNorm2d(2)


class _Wrapper:
    """A plain wrapper embedder (``Qwen3Embedder``-style): holds a module."""

    def __init__(self):
        self.model = _Net()
        self.tokenizer = object()


def test_freeze_puts_the_module_in_eval_mode_with_frozen_weights():
    net = _Net()
    assert net.training is True
    assert net.proj.weight.requires_grad is True

    returned = freeze(net)

    assert returned is net
    assert net.training is False
    assert net.proj.weight.requires_grad is False
    assert net.bn.training is False  # the recursion covers submodules


def test_freeze_accepts_a_plain_wrapper():
    """The memory manager also takes wrappers that are not ``nn.Module``s."""
    wrapper = _Wrapper()

    returned = freeze(wrapper)

    assert returned is wrapper
    assert wrapper.model.training is False
    assert wrapper.model.proj.weight.requires_grad is False


def test_register_freezes_the_component():
    """The single place every adapter's DiT / text encoder / VAE passes through."""
    net = _Net()
    manager = MemoryManager("cpu", "cpu")

    manager.register("dit", net)

    assert net.training is False
    assert net.proj.weight.requires_grad is False


def test_register_tolerates_a_missing_component():
    manager = MemoryManager("cpu", "cpu")
    manager.register("vae", None)
    assert manager.resident() == set()


def test_inference_helper_disables_grad_and_nests():
    assert torch.is_grad_enabled() is True
    with inference():
        assert torch.is_grad_enabled() is False
        with inference():
            assert torch.is_grad_enabled() is False
        assert torch.is_grad_enabled() is False
    assert torch.is_grad_enabled() is True


def test_inference_helper_works_as_a_decorator():
    @inference()
    def build():
        return torch.zeros(1)

    assert torch.is_inference(build())
    assert torch.is_grad_enabled() is True


def test_both_controllers_share_the_one_process_wide_lock():
    pipeline = _pipeline(_SpyingModel())
    upscaler = PixelUpscaleController(PixelUpscalerManager("", "cpu"))

    assert pipeline._lock is upscaler._lock is inference_lock
