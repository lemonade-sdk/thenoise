"""``PipelineController`` tests: the generate path, its cache, refine and finalize.

Every test drives the real controller over the weight-free ``StubModel`` (CPU, fp32),
with no weights and no device moves.
"""
from __future__ import annotations

import io
import json
import logging

import pytest
import torch

from conftest import StubLatentUpscaler, StubModel, TrackedModule
from thenoise.memory import MemoryManager
from thenoise.models.config import GenerateRequest, SamplingParams
from thenoise.postprocess import FILTERS, POSTPROCESS_NAMES, PostprocessOptions
from thenoise.pipeline import (
    PipelineController,
    _normalize_sigmas,
    _refine_schedule,
    _sigma_steps,
)
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
    r = _controller(model)._resolve_pipeline(GenerateRequest(prompt="p"), is_edit=False)

    defaults = model.DEFAULT_PREFS
    assert (r.width, r.height) == (defaults["width"], defaults["height"])
    assert r.steps == defaults["steps"]
    assert r.guidance_scale == defaults["guidance_scale"]
    assert r.effective_sampler == defaults["sampler"]
    # No upscale requested -> identity plan, and no pixel upscaler.
    assert (r.factor, r.upscale_type) == (1.0, "refined")
    assert r.refined is False
    assert (r.target_width, r.target_height) == (r.width, r.height)
    assert r.pixel_scale == 0 and r.pixel_upscaler is None


def test_resolve_explicit_request_wins_over_defaults():
    r = _controller()._resolve_pipeline(
        _request(width=128, height=64, steps=5, guidance_scale=3.5, sampler="er_sde"),
        is_edit=False,
    )
    assert (r.width, r.height, r.steps) == (128, 64, 5)
    assert r.guidance_scale == 3.5
    assert r.effective_sampler == "er_sde"


def test_resolve_upscale_flag_without_factor_uses_the_latent_scale():
    """``upscale=True`` with the 1.0 default factor means "the model's 2x"."""
    model = StubModel()
    r = _controller(model)._resolve_pipeline(_request(upscale=True), is_edit=False)
    assert r.factor == float(model.UPSCALE_SCALE)
    assert r.refined is True
    assert (r.target_width, r.target_height) == (
        r.width * model.UPSCALE_SCALE,
        r.height * model.UPSCALE_SCALE,
    )


def test_resolve_target_size_is_the_rounded_factor():
    r = _controller()._resolve_pipeline(_request(upscale_factor=1.5, upscale=True), is_edit=False)
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
        _request(pixel_upscaler="RealESRGAN_x4", upscale=True), is_edit=False
    )
    assert r.pixel_upscaler is None
    assert r.pixel_scale == 0
    assert r.refined is True  # still a latent 2x

    # With a directory configured the same name is validated and used.
    (tmp_path / "RealESRGAN_x4.safetensors").write_text("x")
    keep = _controller(upscaler_dir=str(tmp_path), upscaler_scales={"RealESRGAN_x4": 4})
    r2 = keep._resolve_pipeline(
        _request(pixel_upscaler="RealESRGAN_x4", upscale_factor=4.0, upscale=True),
        is_edit=False,
    )
    assert r2.pixel_upscaler == "RealESRGAN_x4"
    assert r2.pixel_scale == 4


@pytest.mark.parametrize("seed", [None, -1])
def test_resolve_randomizes_negative_and_missing_seed(seed):
    seen = {
        _controller()._resolve_pipeline(
            GenerateRequest(prompt="p", seed=seed), is_edit=False
        ).seed
        for _ in range(4)
    }
    assert all(isinstance(s, int) and 0 <= s < 2**32 for s in seen)
    assert len(seen) > 1  # "random" really is random, not a fixed sentinel

    assert _controller()._resolve_pipeline(_request(seed=42), is_edit=False).seed == 42


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
        ("prompt",), model.DEFAULT_PREFS["width"], model.DEFAULT_PREFS["height"], 2, 1, "euler"
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


class _RefineSpyModel(StubModel):
    """Stub with a mock latent upscaler injected; logs the refine's inputs.

    The upscaler is the plain ``LatentUpscaler`` mock from conftest: what the
    pipeline does with a real strategy is that strategy's own test, these are
    about the refine.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upscaler = StubLatentUpscaler(self.UPSCALE_SCALE)
        self.refine_inputs = []

    def _create_upscaler(self):
        return self.upscaler

    def prepare_latent(self, latents, cond, params, ref=None, ref_method="index"):
        self.refine_inputs.append((latents.clone(), params))
        return latents


@pytest.mark.parametrize(
    "refine_steps,refine_denoise",
    [(1, 0.25), (2, 0.25), (3, 0.1)],
    ids=["1step", "2steps", "lighter"],
)
def test_upscale_and_refine(refine_steps, refine_denoise):
    """The refine starts AT the ``REFINE_DENOISE`` sigma, for ``REFINE_STEPS`` steps.

    The strength is a sigma rather than a position in the model's schedule, so it
    means the same thing on every model regardless of its shift. The noised latent
    follows ComfyUI's CONST scaling ``σ·noise + (1-σ)·z``.
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

    # The pipeline's whole share of the upscale: ask the model's upscaler once and
    # take its canonical output (8x8 -> 16x16 latent) at face value.
    assert model.upscaler.calls == 1
    assert out.shape == (1, model.LATENT_CHANNELS, 16, 16)

    refine_params = model.refine_inputs[0][1]
    assert refine_params.steps == refine_steps  # the main step count is not reused
    assert (refine_params.height, refine_params.width) == (128, 128)
    assert model.calls["denoise_step"] == refine_steps
    assert model.calls["schedule"] == 0  # the refine brings its own sub-schedule

    # ComfyUI CONST noise scaling ``σ·noise + (1-σ)·z``, reproduced with the seed.
    z_up = torch.full((1, model.LATENT_CHANNELS, 16, 16), 0.5)
    generator = torch.Generator(device="cpu").manual_seed(params.seed)
    noise = torch.randn(z_up.shape, generator=generator)
    noised = model.refine_inputs[0][0]
    assert torch.allclose(noised, refine_denoise * noise + (1.0 - refine_denoise) * z_up)


@pytest.mark.parametrize(
    "steps,sigma", [(1, 0.25), (2, 0.25), (4, 0.4)], ids=["1", "2", "4"]
)
def test_refine_schedule_integrates_sigma_down_to_zero(steps, sigma):
    """Any step count starts at the requested sigma and lands exactly on x0."""
    sub = _refine_schedule(sigma, steps, "cpu", torch.float32)
    ts = [float(s.t) for s in sub]

    assert len(sub) == steps
    assert ts[0] == pytest.approx(sigma)
    assert ts == sorted(ts, reverse=True)
    assert min(ts) > 0.0  # every step is conditioned above 0
    assert sum(float(s.delta) for s in sub) == pytest.approx(sigma)
    # Tensors, not floats: denoise_step consumes Step.t like its own schedule()'s.
    assert all(isinstance(s.t, torch.Tensor) for s in sub)


class _TensorTimestepModel(_RefineSpyModel):
    """Stub that consumes ``Step.t`` the way Anima does (``t.expand(...)``)."""

    def denoise_step(self, latents, t, cond, guidance_scale, i):
        self.calls["denoise_step"] += 1
        t.expand(latents.shape[0])  # raises unless the timestep arrived as a tensor
        return torch.zeros_like(latents)


def test_refine_hands_denoise_step_a_timestep_tensor():
    model = _TensorTimestepModel()
    controller = _controller(model)

    params = SamplingParams(
        height=64, width=64, steps=2, seed=3, guidance_scale=1.0, sampler="euler"
    )
    out = controller._upscale_and_refine(
        torch.ones(1, model.LATENT_CHANNELS, 8, 8), model.encode_prompt(None), params
    )

    assert out.shape == (1, model.LATENT_CHANNELS, 16, 16)
    assert model.calls["denoise_step"] == model.REFINE_STEPS


def test_refine_steps_zero_opts_out_of_the_refine():
    """A model may skip the refine and keep the raw upscaled latent."""
    model = _RefineSpyModel()
    model.REFINE_STEPS = 0
    controller = _controller(model)

    params = SamplingParams(
        height=64, width=64, steps=2, seed=3, guidance_scale=1.0, sampler="euler"
    )
    latents = torch.ones(1, model.LATENT_CHANNELS, 8, 8)
    out = controller._upscale_and_refine(latents, model.encode_prompt(None), params)

    assert out.shape == (1, model.LATENT_CHANNELS, 16, 16)  # still upscaled 2x
    assert model.calls["denoise_step"] == 0
    assert model.refine_inputs == []  # no noise, no prepare, no denoise


# --------------------------------------------------------------- custom sigmas


@pytest.mark.parametrize(
    "values,grid",
    [
        ([1.0, 0.5], (1.0, 0.5, 0.0)),
        ([1.0, 0.5, 0.0], (1.0, 0.5, 0.0)),
        ([0.9, 0.6, 0.3, 0.0], (0.9, 0.6, 0.3, 0.0)),
    ],
    ids=["implied-zero", "explicit-zero", "starts-below-one"],
)
def test_normalize_sigmas_returns_the_full_grid(values, grid):
    """ComfyUI-style: the trailing 0.0 is implied, so N values are N steps."""
    assert _normalize_sigmas(values) == grid


def test_normalize_sigmas_none_or_empty_is_no_grid():
    """``None`` is "unset", ``[]`` is "switch the grid off"."""
    assert _normalize_sigmas(None) is None
    assert _normalize_sigmas([]) is None


@pytest.mark.parametrize(
    "values",
    [
        [0.0],
        [0.25, 0.75],
        [0.5, 0.5],
        [1.0, 1.0],
        [1.5, 0.5],
        [-0.1],
        [float("nan")],
        [float("inf")],
        ["a", "b"],
        "1.0, 0.5",
    ],
    ids=[
        "only-zero", "ascending", "repeat", "repeat-one", "above-one",
        "negative", "nan", "inf", "non-numeric", "string-not-list",
    ],
)
def test_normalize_sigmas_rejects_a_bad_grid(values):
    with pytest.raises(ValueError):
        _normalize_sigmas(values)


def test_sigma_steps_walk_the_grid_verbatim():
    sub = _sigma_steps([1.0, 0.6, 0.2, 0.0], "cpu", torch.float32)

    assert [float(s.t) for s in sub] == pytest.approx([1.0, 0.6, 0.2])
    assert [float(s.delta) for s in sub] == pytest.approx([0.4, 0.4, 0.2])
    assert sum(float(s.delta) for s in sub) == pytest.approx(1.0)  # lands on x0
    # Tensors in the model's dtype, not floats: Anima's denoise_step calls t.expand().
    assert all(isinstance(s.t, torch.Tensor) for s in sub)


def test_resolve_sigmas_set_the_step_count():
    model = StubModel()
    model.checkpoint_prefs = {"steps": 6}  # a marker is overridden too
    r = _controller(model)._resolve_pipeline(_request(sigmas=[1.0, 0.5]), is_edit=False)

    assert r.sigmas == (1.0, 0.5, 0.0)
    assert r.steps == 2


def test_resolve_sigmas_override_steps_with_a_warning(caplog):
    caplog.set_level(logging.WARNING, logger="thenoise.pipeline")
    r = _controller()._resolve_pipeline(_request(steps=8, sigmas=[1.0, 0.5]), is_edit=False)

    assert r.steps == 2
    assert "overrides steps=8" in caplog.text


def test_resolve_sigmas_agreeing_with_steps_stay_quiet(caplog):
    caplog.set_level(logging.WARNING, logger="thenoise.pipeline")
    _controller()._resolve_pipeline(_request(steps=2, sigmas=[1.0, 0.5]), is_edit=False)
    assert caplog.text == ""


class _GridModel(StubModel):
    """An adapter shipping a fixed sigma grid as a default (Qwen-Image 2.1 Turbo)."""

    DEFAULT_PREFS = {
        **StubModel.DEFAULT_PREFS,
        "steps": 4,
        "sigmas": [1.0, 0.75, 0.5, 0.25],
    }


_GRID = (1.0, 0.75, 0.5, 0.25, 0.0)


def test_resolve_uses_the_model_default_sigma_grid():
    r = _controller(_GridModel())._resolve_pipeline(_request(), is_edit=False)

    assert r.sigmas == _GRID
    assert r.steps == 4


def test_resolve_explicit_steps_drop_an_inherited_sigma_grid(caplog):
    """An inherited grid is a default, not a mandate: a pinned step count wins."""
    caplog.set_level(logging.WARNING, logger="thenoise.pipeline")
    r = _controller(_GridModel())._resolve_pipeline(_request(steps=6), is_edit=False)

    assert r.sigmas is None
    assert r.steps == 6  # not 4: the discarded grid must not decide
    assert caplog.text == ""


def test_resolve_inherited_grid_matching_the_request_steps_stays_quiet(caplog):
    caplog.set_level(logging.WARNING, logger="thenoise.pipeline")
    r = _controller(_GridModel())._resolve_pipeline(_request(steps=4), is_edit=False)

    assert (r.sigmas, r.steps) == (_GRID, 4)
    assert caplog.text == ""


def test_resolve_empty_sigmas_switch_off_a_model_default_grid():
    """``sigmas: []`` is the explicit "no grid" answer, at any step count."""
    auto = _controller(_GridModel())._resolve_pipeline(_request(sigmas=[]), is_edit=False)
    assert (auto.sigmas, auto.steps) == (None, 4)

    pinned = _controller(_GridModel())._resolve_pipeline(
        _request(sigmas=[], steps=6), is_edit=False
    )
    assert (pinned.sigmas, pinned.steps) == (None, 6)


def test_resolve_request_grid_replaces_a_model_default_grid():
    r = _controller(_GridModel())._resolve_pipeline(
        _request(sigmas=[1.0, 0.5]), is_edit=False
    )

    assert (r.sigmas, r.steps) == ((1.0, 0.5, 0.0), 2)


def test_generate_with_sigmas_bypasses_the_model_schedule():
    controller = _controller()
    controller.generate(_request(sigmas=[1.0, 0.6, 0.2]))

    assert controller.model.calls["schedule"] == 0
    assert controller.model.calls["denoise_step"] == 3


def test_generate_without_sigmas_still_uses_the_model_schedule():
    controller = _controller()
    controller.generate(_request(steps=3))

    assert controller.model.calls["schedule"] == 1
    assert controller.model.calls["denoise_step"] == 3


def test_generate_runs_a_model_default_grid_without_the_model_schedule():
    controller = _controller(_GridModel())
    controller.generate(_request())

    assert controller.model.calls["schedule"] == 0
    assert controller.model.calls["denoise_step"] == 4


def test_generate_with_pinned_steps_falls_back_to_the_model_schedule():
    controller = _controller(_GridModel())
    controller.generate(_request(steps=5))

    assert controller.model.calls["schedule"] == 1
    assert controller.model.calls["denoise_step"] == 5


def test_custom_sigmas_hand_denoise_step_a_timestep_tensor():
    model = _TensorTimestepModel()
    _controller(model).generate(_request(sigmas=[1.0, 0.5, 0.2]))
    assert model.calls["denoise_step"] == 3


def test_different_sigma_grids_do_not_share_the_cached_latents():
    """Same step count and seed, different grid: the latents are not reusable."""
    controller = _controller()
    controller.generate(_request(sigmas=[1.0, 0.5]))
    controller.generate(_request(sigmas=[1.0, 0.25]))
    assert controller.model.calls["denoise_step"] == 4


def test_same_sigma_grid_twice_reuses_the_cached_latents():
    controller = _controller()
    controller.generate(_request(sigmas=[1.0, 0.5]))
    controller.generate(_request(sigmas=[1.0, 0.5]))
    assert controller.model.calls["denoise_step"] == 2


def test_sigmas_do_not_leak_into_the_refine():
    """The refine keeps its own strength and step count next to a custom grid."""
    model = _RefineSpyModel()
    model.REFINE_STEPS = 3
    controller = _controller(model)
    controller.generate(_request(sigmas=[1.0, 0.5], upscale=True))

    assert model.calls["denoise_step"] == 2 + model.REFINE_STEPS
    assert model.refine_inputs[-1][1].steps == model.REFINE_STEPS


def test_finalize_records_custom_sigmas_in_the_metadata():
    controller = _controller()
    image = controller.generate(_request(sigmas=[1.0, 0.5]))

    texts = _all_texts(image._pnginfo)
    assert json.loads(texts["generation_data"])["sigmas"] == [1.0, 0.5, 0.0]
    assert "Sigmas: 1/0.5/0" in texts["parameters"]


def test_finalize_records_no_sigmas_for_the_model_schedule():
    controller = _controller()
    image = controller.generate(_request(steps=2))

    texts = _all_texts(image._pnginfo)
    assert json.loads(texts["generation_data"])["sigmas"] is None
    assert "Sigmas" not in texts["parameters"]


def test_generate_with_upscale_refines_and_decodes_the_larger_latent():
    controller = _controller(_RefineSpyModel())
    image = controller.generate(_request(upscale=True, steps=2))

    model = controller.model
    assert image.size == (128, 128)  # 64 * latent 2x
    assert model.upscaler.calls == 1  # upscaled once, not once per refine step
    assert model.calls["denoise_step"] == 2 + model.REFINE_STEPS
    assert model.calls["decode"] == 1
    assert model.calls["schedule"] == 1  # only the main run asks the model


def test_generate_without_upscale_never_touches_the_upscaler():
    """A plain run must not pay for the latent upscaler (it loads lazily)."""
    model = _RefineSpyModel()
    controller = _controller(model)
    controller.generate(_request(steps=2))

    assert model.upscaler.calls == 0
    assert model._upscaler is None  # the plain path never asked the model for one


# --------------------------------------------------------------------- postprocess


def test_postprocess_dispatches_only_what_is_requested(monkeypatch):
    """The chain runs a filter only when a knob of its own was asked for."""
    controller = _controller()
    calls = []
    monkeypatch.setitem(
        FILTERS, "rcas",
        lambda pixels, *, sharpening: calls.append(("rcas", sharpening)) or pixels,
    )
    monkeypatch.setitem(
        FILTERS, "film_grain",
        lambda pixels, *, film_grain: calls.append(("film", film_grain)) or pixels,
    )
    pixels = torch.zeros(3, 4, 4)

    # Nothing asked for -> no filter runs, and the input object comes back untouched.
    assert controller.postprocess(pixels, PostprocessOptions()) is pixels
    assert calls == []

    out = controller.postprocess(
        pixels, PostprocessOptions(film_grain=2.0, sharpening=0.5)
    )
    assert out is pixels
    # Chain order: sharpen first, then grain, each seeing the value as the request
    # stated it (the grain adapter is what turns the 0-10 scale into the filter's 0-1).
    assert calls == [("rcas", 0.5), ("film", 2.0)]


def test_finalize_applies_the_notch_filter_only_when_asked(monkeypatch):
    controller = _controller()
    seen = []
    monkeypatch.setitem(
        FILTERS, "nyquist_notch",
        lambda pixels, **_switch: seen.append(1) or pixels,
    )
    pixels = torch.zeros(3, 8, 8)

    plain = _request()
    controller._finalize(pixels, plain, controller._resolve_pipeline(plain, is_edit=False))
    assert seen == []

    enhanced = _request(qwen_vae_enhance=True)
    controller._finalize(pixels, enhanced, controller._resolve_pipeline(enhanced, is_edit=False))
    assert seen == [1]


def test_finalize_resizes_to_the_upscaled_target():
    controller = _controller()
    request = _request(upscale=True)
    image = controller._finalize(
        torch.zeros(3, 64, 64), request, controller._resolve_pipeline(request, is_edit=False)
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


def test_edit_keeps_the_input_metadata_and_records_itself_separately():
    """An edit lands in ``edit_data`` next to the input's chunks, never over them."""
    from PIL import Image

    from conftest import EditingStubModel

    controller = _controller(EditingStubModel())
    source = Image.new("RGB", (64, 64), "white")
    source.info["generation_data"] = '{"prompt": "a fox", "seed": 111}'
    source.info["parameters"] = "a fox\nModel: stub, Steps: 9, Seed: 111"
    source.info["Software"] = "whatever wrote the input"
    source.info["edit_data"] = '{"prompt": "an earlier edit"}'  # replaced, not appended

    image = controller.edit(
        _request(image=source, prompt="a fox with a scarf", steps=1, seed=5)
    )

    texts = _all_texts(image._pnginfo)
    assert json.loads(texts["generation_data"]) == {"prompt": "a fox", "seed": 111}
    assert texts["parameters"] == "a fox\nModel: stub, Steps: 9, Seed: 111"
    assert texts["Software"] == "whatever wrote the input"

    edit = json.loads(texts["edit_data"])
    assert (edit["prompt"], edit["steps"], edit["seed"]) == (
        "a fox with a scarf",
        1,
        5,
    )
    # The chunks really survive the save, as every other section does.
    assert json.loads(_text_chunk(image, "edit_data"))["seed"] == 5


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
        postprocess=PostprocessOptions(),
        lora_specs=None,
        pixel_upscaler=None,
        sigmas=None,
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
        upscale_type="no-refiner", sampler="er_sde",
        postprocess=PostprocessOptions(
            qwen_vae_enhance=True, saturation=0.4, vibrance=-0.2, temperature=0.3,
            tint=-0.1, sharpening=0.2, film_grain=0.5,
        ),
        lora_specs=["a:1.0"], pixel_upscaler="x4",
        sigmas=[1.0, 0.5, 0.0], ref_method="index_timestep_zero", kv_cache=True,
    )
    data = json.loads(_chunk_text(info, "generation_data"))
    assert data == {
        "model": "anima", "prompt": "p", "negative_prompt": "n",
        "width": 512, "height": 256, "steps": 8, "guidance_scale": 1.5, "seed": 3,
        "upscale": True, "upscale_factor": 2.0, "upscale_type": "no-refiner",
        "sampler": "er_sde", "qwen_vae_enhance": True, "saturation": 0.4,
        "vibrance": -0.2, "temperature": 0.3, "tint": -0.1,
        "sharpening": 0.2, "film_grain": 0.5,
        "lora_specs": ["a:1.0"], "pixel_upscaler": "x4",
        "sigmas": [1.0, 0.5, 0.0], "ref_method": "index_timestep_zero",
        "kv_cache": True,
    }


def test_generation_data_carries_every_post_decode_knob():
    """A knob is recorded flat, under its declared name, whatever the chain does with it."""
    info = build_pnginfo(
        model="anima", prompt="p", negative_prompt="", width=64, height=64,
        steps=4, guidance_scale=1.0, seed=1, upscale=False, upscale_factor=1.0,
        upscale_type="refined", sampler="euler", postprocess=PostprocessOptions(),
        lora_specs=None, pixel_upscaler=None,
    )

    assert set(POSTPROCESS_NAMES) <= set(json.loads(_chunk_text(info, "generation_data")))


def test_pnginfo_keeps_the_reference_flags_out_of_the_a1111_text():
    """``parameters`` is read by third-party tools: it stays byte-compatible."""
    assert "kv_cache" not in _parameters(kv_cache=True)
    assert "ref_method" not in _parameters(ref_method="index")


def test_pnginfo_parameters_layout():
    """A1111-compatible: prompt line, optional negative line, then the tail line."""
    lines = _parameters().splitlines()
    assert lines[0] == "a fox"
    assert "Negative prompt:" not in lines
    assert lines[-1] == (
        "Model: flux_klein, Steps: 4, Sampler: euler, Cfg scale: 1.0, Seed: 7"
    )

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


def test_pnginfo_parameters_joins_loras_upscaler_and_sigmas():
    """The tail line is parsed as comma-separated pairs, so sigmas use ``/``."""
    text = _parameters(
        lora_specs=["style:0.8", "pose:1.0"], pixel_upscaler="x4", sigmas=[1.0, 0.5, 0.125]
    )
    assert "LoRA: style:0.8; pose:1.0" in text
    assert "Pixel upscaler: x4" in text
    assert text.splitlines()[-1] == (
        "Model: flux_klein, Steps: 4, Sampler: euler, Cfg scale: 1.0, Seed: 7, "
        "LoRA: style:0.8; pose:1.0, Pixel upscaler: x4, Sigmas: 1/0.5/0.125"
    )


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


def test_save_output_writes_the_provenance_the_controller_attached(tmp_path):
    from PIL import Image

    from thenoise.utils.png import build_upscale_pnginfo, save_output

    image = Image.new("RGB", (2, 2))
    image._pnginfo = build_upscale_pnginfo(image, "RealESRGAN_x4", 2.0)

    # A name with no extension cannot tell PIL a format, so it becomes the PNG the
    # metadata needs; the path actually written is handed back for the caller to log.
    path = save_output(image, str(tmp_path / "out"))
    assert path.endswith("out.png")
    saved = Image.open(path)
    assert json.loads(saved.info["upscale_data"])["upscaler_model"] == "RealESRGAN_x4"


def test_pnginfo_of_an_image_that_carries_none():
    from PIL import Image

    from thenoise.utils.png import pnginfo_of

    assert pnginfo_of(Image.new("RGB", (2, 2))) is None


def test_kv_cache_requires_an_edit_request():
    """``kv_cache`` is a reference-latent optimization: it needs a reference image."""
    controller = _controller()
    with pytest.raises(ValueError, match="requires an edit request"):
        controller.generate(_request(kv_cache=True))


def test_kv_cache_requires_a_model_that_supports_it():
    """The KV cache is per-adapter: the pipeline refuses instead of ignoring it.

    ``EditingStubModel`` edits (so it has a reference latent to freeze) but never
    wired ``thenoise.dit.kvcache`` into its blocks, which is what the ``kv_cache``
    capability advertises.
    """
    from PIL import Image

    from conftest import EditingStubModel

    controller = _controller(EditingStubModel())
    with pytest.raises(ValueError, match="does not support the reference-latent KV cache"):
        controller.edit(
            _request(image=Image.new("RGB", (64, 64), "white"), steps=1, kv_cache=True)
        )


# ---------------------------------------------------- residency across the stages
class _Device(MemoryManager):
    """A manager whose device reports a fixed capacity, so placement is decidable."""

    def __init__(self, capacity: int):
        super().__init__("cpu", "meta")
        self.capacity = capacity

    def free_bytes(self) -> int:
        return self.capacity - sum(self.weight_bytes(n) for n in self.resident())


def _model_on(monkeypatch, capacity: int, *, reserve: int = 1 << 30):
    """A StubModel with tracked weight sets on a device of ``capacity``.

    ``reserve`` pins what every stage keeps free, so the fixtures are about placement at
    GiB scale and not at the mercy of the fleet's figure.
    """
    monkeypatch.setattr("thenoise.memory._RESERVE_BASE_BYTES", reserve)
    monkeypatch.setattr("thenoise.memory._RESERVE_BYTES_PER_MEGAPIXEL", 0)

    model = StubModel()
    memory = _Device(capacity)
    parts = {"dit": TrackedModule(1 << 30), "text_encoder": TrackedModule(1 << 30),
             "vae": TrackedModule(128 << 20, device="cpu")}    # 1 GiB, 1 GiB, 128 MiB
    for name, part in parts.items():
        memory.register(name, part)
    model.memory = memory
    return model, parts


def _request_at(model, **kwargs) -> GenerateRequest:
    return _request(**{"width": 64, "height": 64, **kwargs})


def test_a_request_leaves_the_weights_it_used_on_the_device(monkeypatch):
    """The point of the whole arrangement: no round trip between two stages."""
    model, parts = _model_on(monkeypatch, 24 << 30)
    _controller(model).generate(_request_at(model))

    assert parts["dit"].moves == ["cpu"], "up once, and not back down again"
    assert parts["text_encoder"].moves == ["cpu"]
    assert model.memory.resident() == {"dit", "text_encoder", "vae"}


def test_two_requests_on_a_cached_prompt_upload_the_dit_once(monkeypatch):
    model, parts = _model_on(monkeypatch, 24 << 30)
    controller = _controller(model)

    controller.generate(_request_at(model))
    controller.generate(_request_at(model))

    assert parts["dit"].moves == ["cpu"]
    assert parts["text_encoder"].moves == ["cpu"]


def test_the_idle_weights_pay_for_a_stage_that_needs_the_room(monkeypatch):
    """The text encoder is finished with by the time the DiT needs the device.

    Sized so the room will not fit beside encoder and DiT together: leaving a stage moves
    nothing, so it is the DiT's own entry that evicts the encoder.
    """
    model, parts = _model_on(monkeypatch, int(2.5 * (1 << 30)))
    _controller(model).generate(_request_at(model))

    assert parts["text_encoder"].moves == ["cpu", "meta"]
    assert parts["dit"].moves == ["cpu"]
    assert model.memory.resident() == {"dit", "vae"}


def test_a_decode_with_a_big_claim_takes_the_last_weight_set_too(monkeypatch):
    """Nothing is exempt from a claim it cannot be met without — except the VAE running."""
    model, parts = _model_on(monkeypatch, 3 << 30, reserve=4 << 30)
    _controller(model).generate(_request_at(model))

    assert parts["dit"].moves == ["cpu", "meta"]
    assert model.memory.resident() == {"vae"}, "only the VAE running the decode is exempt"


def test_a_decode_that_ran_out_gets_the_idle_weights_off_the_device(monkeypatch):
    """A stage that claimed less than it needed is not the end of the image."""
    model, parts = _model_on(monkeypatch, 24 << 30)
    real_decode = model.decode
    attempts = {"n": 0}

    def once(latents):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise torch.cuda.OutOfMemoryError("the claim was too small")
        return real_decode(latents)

    model.decode = once
    image = _controller(model).generate(_request_at(model))

    assert attempts["n"] == 2 and image.size == (64, 64)
    assert parts["dit"].moves == ["cpu", "meta"], "everything idle went, the VAE stayed"
    assert model.memory.resident() == {"vae"}


def test_a_prompt_that_ran_out_gets_the_idle_weights_off_the_device(monkeypatch):
    """The stage that ran out is not always the one the device was sized for.

    The second request finds the first one's DiT still aboard — residency is sticky —
    and its encoder is the one that runs short.
    """
    model, parts = _model_on(monkeypatch, 24 << 30)
    controller = _controller(model)
    controller.generate(_request_at(model, prompt="first"))

    real = model.encode_prompt
    attempts = {"n": 0}

    def once(args):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise torch.cuda.OutOfMemoryError("the encoder wanted more")
        return real(args)

    model.encode_prompt = once
    image = controller.generate(_request_at(model, prompt="second"))

    assert attempts["n"] == 2 and image.size == (64, 64)
    assert parts["dit"].moves == ["cpu", "meta", "cpu"], "paid for, then loaded again"


def test_a_second_oom_with_nothing_left_to_take_is_not_retried(monkeypatch):
    model, _ = _model_on(monkeypatch, 24 << 30)

    def always(latents):
        raise torch.cuda.OutOfMemoryError("nothing left to give")

    model.decode = always
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _controller(model).generate(_request_at(model))


def test_a_stage_only_asks_for_the_room_it_declared(monkeypatch):
    """The manager never evicts on a claim the device can already satisfy."""
    model, parts = _model_on(monkeypatch, 5 << 30)
    _controller(model).generate(_request_at(model))

    assert parts["text_encoder"].moves == ["cpu"], "there was room for all of it"


def test_the_prompt_stage_claims_the_pixels_the_encoder_sees(monkeypatch):
    """An edit feeds the encoder its images; the size of the result is not its business."""
    from PIL import Image

    from conftest import EditingStubModel

    model = EditingStubModel()
    memory = model.memory
    stage = memory.stage
    asked: list[tuple[tuple[str, ...], int]] = []

    def spy(*names, pixels):
        asked.append((names, pixels))
        return stage(*names, pixels=pixels)

    memory.stage = spy
    _controller(model).edit(_request(image=Image.new("RGB", (32, 48)), steps=1))

    claims = [pixels for names, pixels in asked if names == ("text_encoder",)]
    assert claims == [32 * 48], "the vision tower is sized by what it is shown"


def test_a_stage_sizes_its_room_by_the_pixels_it_moves(monkeypatch):
    """Each stage says how big an image it runs, once, at the size it really runs at."""
    model, _ = _model_on(monkeypatch, 24 << 30)
    memory = model.memory
    stage = memory.stage
    asked: list[tuple[tuple[str, ...], int]] = []

    def spy(*names, pixels):
        asked.append((names, pixels))
        return stage(*names, pixels=pixels)

    memory.stage = spy
    _controller(model).generate(_request(width=64, height=64, upscale=True))

    scale = model.UPSCALE_SCALE
    target = (64 * scale) ** 2
    assert asked == [(("text_encoder",), 0),
                     (("dit",), 64 * 64),
                     (("dit", "vae"), target),
                     (("vae",), target)], (
        # The refine holds the VAE too: a latent upscale can be a round trip through it.
        "the tail sizes itself by the final image")
