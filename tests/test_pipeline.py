"""``PipelineController`` tests — the generate path, its cache, refine and finalize.

Every test drives the real controller over the weight-free ``StubModel`` (CPU,
fp32): request resolution, the single-entry stage cache, the latent
upscale-and-refine, the postprocess dispatch and the PNG metadata are all
exercised end to end, with no weights and no device moves.
"""
from __future__ import annotations

import io
import json

import pytest
import torch

from conftest import StubModel
from thenoise.models.config import GenerateRequest, SamplingParams
from thenoise.pipeline import PipelineController
from thenoise.upscale.inference_adaptors import LatentFormatAdaptor
from thenoise.upscale.pixel import PixelUpscalerManager
from thenoise.utils.png import build_pnginfo


def _controller(model=None, *, upscaler_dir="", upscaler_scales=None):
    manager = PixelUpscalerManager(upscaler_dir=upscaler_dir, device="cpu")
    manager._pixel_upscaler_scales = dict(upscaler_scales or {})
    return PipelineController(model or StubModel(), manager)


def _request(**kwargs) -> GenerateRequest:
    """A request with a fixed seed (cache tests need a deterministic seed)."""
    return GenerateRequest(**{"prompt": "a fox", "seed": 1, **kwargs})


# ----------------------------------------------------------- request resolution


def test_resolve_falls_back_to_model_defaults():
    """Every ``None`` request field takes the model's own default."""
    model = StubModel()
    r = _controller(model)._resolve_pipeline(GenerateRequest(prompt="p"))

    assert (r.width, r.height) == (model.DEFAULT_WIDTH, model.DEFAULT_HEIGHT)
    assert r.steps == model.DEFAULT_STEPS
    assert r.guidance_scale == model.DEFAULT_GUIDANCE_SCALE
    assert r.effective_sampler == model.SAMPLER
    # No upscale requested -> identity plan, and no pixel upscaler.
    assert (r.factor, r.upscale_type) == (1.0, "refined")
    assert r.refined is False
    assert (r.target_width, r.target_height) == (r.width, r.height)
    assert r.pixel_scale == 0 and r.pixel_upscaler is None


def test_resolve_explicit_request_wins_over_defaults():
    r = _controller()._resolve_pipeline(
        _request(width=128, height=64, steps=5, guidance_scale=3.5, sampler="er_sde")
    )
    assert (r.width, r.height, r.steps) == (128, 64, 5)
    assert r.guidance_scale == 3.5
    assert r.effective_sampler == "er_sde"


def test_resolve_upscale_flag_without_factor_uses_the_latent_scale():
    """``upscale=True`` with the 1.0 default factor means "the model's 2x"."""
    model = StubModel()
    r = _controller(model)._resolve_pipeline(_request(upscale=True))
    assert r.factor == float(model.UPSCALE_SCALE)
    assert r.refined is True
    assert (r.target_width, r.target_height) == (
        r.width * model.UPSCALE_SCALE,
        r.height * model.UPSCALE_SCALE,
    )


def test_resolve_target_size_is_the_rounded_factor():
    r = _controller()._resolve_pipeline(_request(upscale_factor=1.5, upscale=True))
    assert (r.target_width, r.target_height) == (
        round(r.width * 1.5),
        round(r.height * 1.5),
    )


def test_resolve_drops_pixel_upscaler_when_no_dir_is_configured(tmp_path):
    """A requested pixel upscaler is silently ignored without ``--upscaler-dir``.

    Documented fallback: rather than failing the request, the pipeline degrades to
    a refined (latent-only) upscale. If this ever stops happening the request
    would hard-fail for every user who left the directory unset.
    """
    r = _controller()._resolve_pipeline(
        _request(pixel_upscaler="RealESRGAN_x4", upscale=True)
    )
    assert r.pixel_upscaler is None
    assert r.pixel_scale == 0
    assert r.refined is True  # still a latent 2x

    # With a directory configured the same name is validated and used.
    (tmp_path / "RealESRGAN_x4.safetensors").write_text("x")
    keep = _controller(upscaler_dir=str(tmp_path), upscaler_scales={"RealESRGAN_x4": 4})
    r2 = keep._resolve_pipeline(
        _request(pixel_upscaler="RealESRGAN_x4", upscale_factor=4.0, upscale=True)
    )
    assert r2.pixel_upscaler == "RealESRGAN_x4"
    assert r2.pixel_scale == 4


@pytest.mark.parametrize("seed", [None, -1])
def test_resolve_randomizes_negative_and_missing_seed(seed):
    seen = {_controller()._resolve_pipeline(GenerateRequest(prompt="p", seed=seed)).seed
            for _ in range(4)}
    assert all(isinstance(s, int) and 0 <= s < 2**32 for s in seen)
    assert len(seen) > 1  # "random" really is random, not a fixed sentinel


def test_resolve_explicit_seed_is_passed_through():
    r = _controller()._resolve_pipeline(_request(seed=42, prompt="p"))
    assert r.seed == 42


# ------------------------------------------------------------------ generate path


def test_generate_runs_the_full_pipeline_and_returns_a_png():
    controller = _controller()
    image = controller.generate(_request(width=64, height=64, steps=2))

    assert image.size == (64, 64)
    model = controller.model
    assert model.calls["encode_prompt"] == 1
    assert model.calls["init_latents"] == 1
    assert model.calls["denoise_step"] == 2  # one per schedule step
    assert model.calls["decode"] == 1
    assert image._pnginfo is not None


def test_same_request_twice_reuses_every_stage():
    controller = _controller()
    controller.generate(_request(steps=2))
    after_first = dict(controller.model.calls)

    controller.generate(_request(steps=2))
    assert dict(controller.model.calls) == after_first  # nothing re-ran


def test_changing_the_prompt_reencodes_but_keeps_the_cache_shape():
    controller = _controller()
    controller.generate(_request(prompt="one", steps=2, seed=1))
    assert controller.model.calls["encode_prompt"] == 1

    controller.generate(_request(prompt="two", steps=2, seed=2))
    assert controller.model.calls["encode_prompt"] == 2
    assert controller.model.calls["denoise_step"] == 4  # sampling re-ran too


def test_changing_lora_specs_invalidates_the_cached_conditioning():
    """Prompt conditioning embeds ``lora_specs`` (text fusion runs on DiT weights)."""
    controller = _controller()
    controller.generate(_request(steps=2))
    assert controller.model.calls["encode_prompt"] == 1

    controller.generate(_request(steps=2, lora_specs=["style.safetensors:0.8"]))
    assert controller.model.calls["encode_prompt"] == 2
    assert controller.model.lora_switches == [None, ["style.safetensors:0.8"]]


@pytest.mark.parametrize(
    "override",
    [{"seed": 7}, {"steps": 3}, {"sampler": "er_sde"}],
    ids=["seed", "steps", "sampler"],
)
def test_changing_sampling_params_reuses_prompt_but_resamples(override):
    controller = _controller()
    controller.generate(_request(steps=2))
    controller.generate(_request(**{"steps": 2, **override}))

    model = controller.model
    assert model.calls["encode_prompt"] == 1  # prompt stage reused
    assert model.calls["denoise_step"] == 2 + override.get("steps", 2)


def test_decode_key_changes_when_refined():
    """The refined path produces a different latent, so it must not share a key."""
    controller = _controller()
    model = controller.model
    sampling_key = controller._cache_key_sampling(
        ("prompt",), model.DEFAULT_WIDTH, model.DEFAULT_HEIGHT, 2, 1, "euler"
    )
    plain = controller._cache_key_decode(sampling_key, False)
    refined = controller._cache_key_decode(sampling_key, True)

    assert plain != refined
    # The refine constants are part of the refined key: changing the model's
    # refine schedule must invalidate an otherwise identical decode.
    assert refined == (
        "decode_refined",
        sampling_key,
        model.UPSCALE_SCALE,
        model.REFINE_STEPS,
        model.REFINE_DENOISE,
    )


# ------------------------------------------------------- latent upscale + refine


class _FakeUpscaler:
    """Records the requested target and returns a blank latent of that size."""

    def __init__(self):
        self.targets = []

    def __call__(self, raw, target):
        self.targets.append(tuple(target))
        return torch.full(
            (1, raw.shape[1], target[0], target[1]), 0.5, dtype=raw.dtype
        )


class _RefineSpyModel(StubModel):
    """Stub with an injected latent upscaler; logs the refine's inputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upscaler = _FakeUpscaler()
        self.adaptor = LatentFormatAdaptor(external_channels=self.LATENT_CHANNELS)
        self.refine_inputs = []

    def load_latent_upscaler(self):
        return self.upscaler, self.adaptor

    def prepare_latent(self, latents, cond, params, ref=None, ref_method="index"):
        self.refine_inputs.append((latents.clone(), params))
        return latents


@pytest.mark.parametrize(
    "refine_steps,refine_denoise",
    [(1, 0.1), (2, 0.25), (1, 0.3)],
    ids=["1of10", "2of8", "truncated-schedule"],
)
def test_upscale_and_refine(refine_steps, refine_denoise):
    """The refine is the last ``REFINE_STEPS`` of an INDEPENDENT schedule.

    ``new_steps = int(REFINE_STEPS / REFINE_DENOISE)`` (an int() truncation, not a
    round), the noise level is that sub-schedule's first ``t``, and the noised
    latent follows ComfyUI's CONST scaling ``σ·noise + (1-σ)·z``.
    """
    model = _RefineSpyModel()
    model.REFINE_STEPS = refine_steps
    model.REFINE_DENOISE = refine_denoise
    controller = _controller(model)

    params = SamplingParams(
        height=64, width=64, steps=2, seed=3, guidance_scale=1.0, sampler="euler"
    )
    latents = torch.ones(1, model.LATENT_CHANNELS, 8, 8)
    out = controller._upscale_and_refine(latents, model.encode_prompt(None), params)

    # 2x latent upscale through the (identity) adaptor, in external coordinates.
    assert model.upscaler.targets == [(16, 16)]
    assert out.shape == (1, model.LATENT_CHANNELS, 16, 16)

    new_steps = int(refine_steps / refine_denoise)
    refine_params = model.refine_inputs[0][1]
    assert refine_params.steps == new_steps  # the original step count is NOT reused
    assert (refine_params.height, refine_params.width) == (128, 128)

    full = model.schedule(refine_params)
    assert full is not None and len(full) == new_steps
    sub = full[-refine_steps:]
    strength = float(sub[0].t)
    assert model.calls["denoise_step"] == refine_steps  # only the tail ran

    # ComfyUI CONST noise scaling ``σ·noise + (1-σ)·z``, reproduced with the seed.
    z_up = torch.full((1, model.LATENT_CHANNELS, 16, 16), 0.5)
    generator = torch.Generator(device="cpu").manual_seed(params.seed)
    noise = torch.randn(z_up.shape, generator=generator)
    noised = model.refine_inputs[0][0]
    assert torch.allclose(noised, strength * noise + (1.0 - strength) * z_up)


def test_generate_with_upscale_refines_and_decodes_the_larger_latent():
    controller = _controller(_RefineSpyModel())
    image = controller.generate(_request(upscale=True, steps=2))

    model = controller.model
    assert image.size == (128, 128)  # 64 * latent 2x
    assert model.calls["denoise_step"] == 2 + model.REFINE_STEPS
    assert model.calls["decode"] == 1
    assert model.calls["schedule"] == 2  # main + refine, never a merged schedule


# --------------------------------------------------------------------- postprocess


def test_postprocess_dispatches_only_what_is_requested(monkeypatch):
    controller = _controller()
    calls = []
    monkeypatch.setattr(
        "thenoise.pipeline.rcas",
        lambda pixels, strength: calls.append(("rcas", strength)) or pixels,
    )
    monkeypatch.setattr(
        "thenoise.pipeline.film_grain",
        lambda pixels, strength: calls.append(("film", strength)) or pixels,
    )
    pixels = torch.zeros(3, 4, 4)

    # Both off -> the input object is returned untouched.
    assert controller.postprocess(pixels) is pixels
    assert calls == []

    out = controller.postprocess(pixels, film_grain_strength=2.0, sharpening=0.5)
    assert out is pixels
    # film_grain is requested on a 0-10 scale and scaled /10 for the filter.
    assert calls == [("rcas", 0.5), ("film", 0.2)]


def test_finalize_applies_the_notch_filter_only_when_asked(monkeypatch):
    controller = _controller()
    seen = []
    monkeypatch.setattr(
        "thenoise.pipeline.nyquist_notch",
        lambda pixels: seen.append(1) or pixels,
    )
    pixels = torch.zeros(3, 8, 8)

    plain = _request()
    controller._finalize(pixels, plain, controller._resolve_pipeline(plain))
    assert seen == []

    enhanced = _request(qwen_vae_enhance=True)
    controller._finalize(pixels, enhanced, controller._resolve_pipeline(enhanced))
    assert seen == [1]


def test_finalize_resizes_to_the_upscaled_target():
    controller = _controller()
    request = _request(upscale=True)
    image = controller._finalize(
        torch.zeros(3, 64, 64), request, controller._resolve_pipeline(request)
    )
    assert image.size == (128, 128)


def test_finalize_attaches_png_metadata():
    """``_finalize`` hangs the metadata on the PIL image the API/CLI then saves."""
    controller = _controller()
    image = controller.generate(_request(steps=2, seed=5))

    texts = _all_texts(image._pnginfo)
    data = json.loads(texts["generation_data"])
    assert data["prompt"] == "a fox"
    assert data["steps"] == 2
    assert data["seed"] == 5
    assert data["model"] == "stub"
    assert "Steps: 2" in texts["parameters"]

    # ...and the chunks really survive a PNG round-trip.
    assert json.loads(_text_chunk(image, "generation_data"))["seed"] == 5


def _text_chunk(image, keyword: str) -> str:
    """Read a text chunk back from the image by saving and re-opening it."""
    from PIL import Image

    buf = io.BytesIO()
    image.save(buf, format="PNG", pnginfo=image._pnginfo)
    return Image.open(io.BytesIO(buf.getvalue())).info[keyword]


# ----------------------------------------------------------- utils/png.py formats


def _parameters(**overrides) -> str:
    kwargs = dict(
        model="flux_klein",
        prompt="a fox",
        negative_prompt="",
        width=1024,
        height=1024,
        steps=4,
        guidance_scale=1.0,
        seed=7,
        upscale=False,
        upscale_factor=1.0,
        upscale_type="refined",
        sampler="euler",
        qwen_vae_enhance=False,
        film_grain=0.0,
        sharpening=0.0,
        lora_specs=None,
        pixel_upscaler=None,
    )
    kwargs.update(overrides)
    return _chunk_text(build_pnginfo(**kwargs), "parameters")


def _chunk_text(pnginfo, keyword: str) -> str:
    return _all_texts(pnginfo)[keyword]


def _all_texts(pnginfo) -> dict:
    out = {}
    for _cid, data, _after in pnginfo.chunks:
        key, _, value = data.partition(b"\0")
        out[key.decode()] = value.decode()
    return out


def test_pnginfo_json_block_covers_every_field():
    info = build_pnginfo(
        model="anima", prompt="p", negative_prompt="n", width=512, height=256,
        steps=8, guidance_scale=1.5, seed=3, upscale=True, upscale_factor=2.0,
        upscale_type="no-refiner", sampler="er_sde", qwen_vae_enhance=True,
        film_grain=0.5, sharpening=0.2, lora_specs=["a:1.0"], pixel_upscaler="x4",
    )
    data = json.loads(_chunk_text(info, "generation_data"))
    assert data == {
        "model": "anima", "prompt": "p", "negative_prompt": "n",
        "width": 512, "height": 256, "steps": 8, "guidance_scale": 1.5, "seed": 3,
        "upscale": True, "upscale_factor": 2.0, "upscale_type": "no-refiner",
        "sampler": "er_sde", "qwen_vae_enhance": True, "film_grain": 0.5,
        "sharpening": 0.2, "lora_specs": ["a:1.0"], "pixel_upscaler": "x4",
    }


def test_pnginfo_parameters_plain_prompt_only():
    """No negative, no upscale, no LoRA: prompt line + the key/value tail."""
    text = _parameters()
    lines = text.splitlines()
    assert lines[0] == "a fox"
    assert "Negative prompt:" not in text
    assert lines[-1] == (
        "Model: flux_klein, Steps: 4, Sampler: euler, Cfg scale: 1.0, Seed: 7"
    )


def test_pnginfo_parameters_includes_negative_prompt_line():
    lines = _parameters(negative_prompt="blurry, text").splitlines()
    assert lines[:2] == ["a fox", "Negative prompt: blurry, text"]


def test_pnginfo_parameters_upscale_pair_is_all_or_nothing():
    """``Upscale type`` only makes sense next to ``Upscale factor``."""
    plain = _parameters()
    assert "Upscale" not in plain

    factor = _parameters(upscale_factor=2.0, upscale_type="no-refiner")
    assert "Upscale: true" not in factor  # the flag itself is a separate field
    assert "Upscale factor: 2, Upscale type: no-refiner" in factor

    flag = _parameters(upscale=True)
    assert "Upscale: true" in flag
    assert "Upscale factor" not in flag and "Upscale type" not in flag


def test_pnginfo_parameters_joins_loras_and_pixel_upscaler():
    text = _parameters(lora_specs=["style:0.8", "pose:1.0"], pixel_upscaler="x4")
    assert "LoRA: style:0.8; pose:1.0" in text
    assert "Pixel upscaler: x4" in text


def test_build_upscale_pnginfo_carries_over_and_replaces():
    from PIL import Image

    from thenoise.utils.png import build_upscale_pnginfo

    source = Image.new("RGB", (2, 2))
    source.info["prompt"] = "keep me"
    source.info["upscale_data"] = '{"stale": true}'

    info = build_upscale_pnginfo(source, "RealESRGAN_x4", 4.0)
    texts = _all_texts(info)
    assert texts["prompt"] == "keep me"
    assert json.loads(texts["upscale_data"]) == {
        "upscaler_model": "RealESRGAN_x4",
        "upscale_factor": 4.0,
    }

