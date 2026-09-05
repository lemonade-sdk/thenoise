"""Per-model schedules and size resolution, over the whole catalog.

One parametrized contract for all four adapters: the step list is ``steps`` long,
starts at t=1, is strictly decreasing, ends on a grid whose last point is 0, and
its ``delta`` is the step to the next grid point (that is what both solvers
integrate). Resolution dependence is asserted where a model's schedule has it and
asserted *absent* where it does not.

The adapters are built bare (``object.__new__``) with the couple of attributes
their kernel reads: no checkpoints, no device.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from conftest import CATALOG_IDS
from thenoise.models import (
    MODEL_CATALOG,
    AnimaModel,
    FluxKleinModel,
    Krea2Model,
    QwenImageModel,
    ZImageModel,
)
from thenoise.models.base import DiffusionModel
from thenoise.models.config import SamplingParams


def _bare(cls, **attrs):
    """An adapter instance without ``__init__`` (no weights, no device moves)."""
    model = object.__new__(cls)
    model.device = "cpu"
    model.dtype = torch.float32
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def _params(steps=8, width=1024, height=1024):
    return SamplingParams(
        height=height, width=width, steps=steps, seed=0, guidance_scale=1.0, sampler="euler"
    )


# The shipped patch/latent geometry of each adapter (the shipped configs all
# patchify 2x2 on an 8x-compressed VAE latent -> a 16px pixel alignment).
BARE = {
    "anima": {},
    "krea2": {"dit": SimpleNamespace(config=SimpleNamespace(patch=2)), "_compression": 8},
    "zimage": {"dit": SimpleNamespace(patch_size=2)},
    "flux_klein": {},
    "qwen_image": {},
}

# The adapters whose step schedule shifts with the image token count. Krea 2 is
# NOT one of them: it pins ``mu=DEFAULT_MU`` (the distilled checkpoint was trained
# at a fixed shift), so its grid is resolution independent by design.
RESOLUTION_AWARE = {"flux_klein", "qwen_image"}


@pytest.mark.parametrize("model_cls", MODEL_CATALOG, ids=CATALOG_IDS)
@pytest.mark.parametrize("steps", [1, 4, 8])
def test_schedule_contract(model_cls, steps):
    model = _bare(model_cls, **BARE[model_cls.name])
    schedule = model.schedule(_params(steps=steps))

    # One Step per denoise iteration (the solvers call denoise_step once each).
    assert len(schedule) == steps
    # The grid starts at pure noise.
    assert float(schedule[0].t) == pytest.approx(1.0, abs=1e-6)
    # Strictly decreasing towards zero, with every positive step recorded.
    ts = [float(s.t) for s in schedule]
    deltas = [float(s.delta) for s in schedule]
    assert all(t > 0 for t in ts)
    assert all(d > 0 for d in deltas)
    assert all(a > b for a, b in zip(ts, ts[1:]))  # strictly decreasing
    # ``delta`` is the distance to the next grid point...
    assert deltas[:-1] == pytest.approx([a - b for a, b in zip(ts, ts[1:])])
    # ...and the last step lands on 0, i.e. the grid really runs 1 -> 0.
    assert deltas[-1] == pytest.approx(ts[-1])


@pytest.mark.parametrize("model_cls", MODEL_CATALOG, ids=CATALOG_IDS)
def test_schedule_resolution_dependence_matches_the_model(model_cls):
    small = _bare(model_cls, **BARE[model_cls.name]).schedule(_params(steps=8, width=512, height=512))
    large = _bare(model_cls, **BARE[model_cls.name]).schedule(_params(steps=8, width=1024, height=1024))

    ts_small = [float(s.t) for s in small]
    ts_large = [float(s.t) for s in large]
    if model_cls.name in RESOLUTION_AWARE:
        # A bigger image gets a stronger time shift, so the grids differ.
        assert ts_small != ts_large
    else:
        # A static schedule must not drift with resolution.
        assert ts_small == ts_large


@pytest.mark.parametrize("model_cls", MODEL_CATALOG, ids=CATALOG_IDS)
def test_resolve_size_aligns_to_the_patched_latent_grid(model_cls):
    """Odd sizes are rounded up to the model's pixel alignment (16 = 8 * patch 2)."""
    model = _bare(model_cls, **BARE[model_cls.name])
    assert model.resolve_size(100, 60) == (112, 64)
    # An already-aligned size is untouched.
    assert model.resolve_size(1024, 512) == (1024, 512)


@pytest.mark.parametrize("model_cls", MODEL_CATALOG, ids=CATALOG_IDS)
def test_percent_to_sigma_stays_strictly_below_one(model_cls):
    """ER-SDE divides by ``1 - sigma``: sigma_0 must be nudged below 1, never on it.

    The same function maps the ends of the percent axis onto the sigma axis.
    """
    model = _bare(model_cls, **BARE[model_cls.name])
    nudged = model.percent_to_sigma(1e-4)
    assert 0.0 < nudged < 1.0
    assert model.percent_to_sigma(0.0) == 1.0
    assert model.percent_to_sigma(1.0) == 0.0


@pytest.mark.parametrize(
    "model_cls,expect",
    [
        (AnimaModel, False),
        (Krea2Model, False),
        (ZImageModel, False),
        (FluxKleinModel, True),
        (QwenImageModel, True),
    ],
    ids=CATALOG_IDS,
)
def test_reference_editing_capability(model_cls, expect):
    """Only adapters that override the reference kernels advertise ``supports_edit``.

    The pipeline raises for a model that advertises editing without the kernels,
    so the flag and the overrides must never drift apart.
    """
    model = _bare(model_cls, **BARE[model_cls.name])
    assert model.supports_edit is expect
    overrides_reference_kernels = (
        model_cls.encode_reference is not DiffusionModel.encode_reference
        and model_cls.pack_reference_latent is not DiffusionModel.pack_reference_latent
    )
    assert overrides_reference_kernels is expect


def test_base_encode_reference_is_not_implemented():
    """A non-editing adapter raises instead of returning a bogus latent."""
    anima = _bare(AnimaModel)
    with pytest.raises(NotImplementedError, match="does not support reference editing"):
        anima.encode_reference(torch.zeros(3, 8, 8))

    # The generic pack helper is a no-op marker (None = "no reference tokens").
    assert anima.pack_reference_latent(torch.zeros(1, 4, 4)) is None


def test_decode_squeezes_a_frame_axis_and_returns_float32():
    """``decode`` accepts a 5D ``[B,C,1,H,W]`` VAE output and hands back ``[C,H,W]``."""
    model = _bare(AnimaModel)
    model.vae = _FakeVAE(out_5d=True, dtype=torch.bfloat16)
    pixels = model.decode(torch.zeros(1, 16, 4, 4))

    assert pixels.shape == (3, 8, 8)
    assert pixels.dtype == torch.float32  # the postprocess/convert tail expects fp32


def test_decode_passes_a_4d_vae_output_through():
    model = _bare(AnimaModel)
    model.vae = _FakeVAE(out_5d=False, dtype=torch.float32)
    pixels = model.decode(torch.zeros(1, 16, 4, 4))
    assert pixels.shape == (3, 8, 8)
    assert pixels.dtype == torch.float32


class _FakeVAE(torch.nn.Module):
    """Returns a fixed pixel tensor, optionally with the legacy frame axis."""

    def __init__(self, out_5d: bool, dtype: torch.dtype):
        super().__init__()
        self.out_5d = out_5d
        self.dtype = dtype
        self.seen = []

    def decode_to_pixels(self, latents):
        self.seen.append((tuple(latents.shape), latents.dtype))
        pixels = torch.zeros(1, 3, 8, 8, dtype=self.dtype)
        return pixels.unsqueeze(2) if self.out_5d else pixels


# ----------------------------------------------------------- base-class defaults


def test_base_fuse_text_and_prepare_latent_are_identities():
    """Adapters without a DiT-side text fusion or a latent reshape fall back to
    the identity, so the controller can call them uniformly."""
    model = _bare(ZImageModel)  # uses the base identity for fuse_text
    cond = "raw-cond"
    assert model.fuse_text(cond) == cond
    latents = torch.zeros(1, 4, 8, 8)
    # ZImage overrides ``prepare_latent``; exercise the base identity directly.
    assert DiffusionModel.prepare_latent(model, latents, cond, _params()) is latents


def test_file_size_counts_missing_as_zero():
    """An unreadable path must not blow the offload estimate up."""
    assert DiffusionModel._file_size("/nonexistent/nope.safetensors") == 0


def test_load_latent_upscaler_is_lazy_and_cached(monkeypatch):
    from thenoise.models import base as base_mod

    calls = []

    def fake_load(fmt, device, dtype):
        calls.append((fmt, device, dtype))
        return "upscaler", "adaptor"

    monkeypatch.setattr(base_mod, "load_latent_upscaler", fake_load)
    model = _bare(AnimaModel, _upscaler=None, _adaptor=None)
    assert model.load_latent_upscaler() == ("upscaler", "adaptor")
    assert calls == [("wan21", "cpu", torch.float32)]
    # Second call is served from the cache; the loader never runs again.
    assert model.load_latent_upscaler() == ("upscaler", "adaptor")
    assert len(calls) == 1
