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

from dataclasses import replace
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
    QwenImage21Model,
    ZImageModel,
)
from thenoise.models.base import DiffusionModel
from thenoise.models.config import SamplingParams
from thenoise.utils.math import round_up


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


# The attributes each adapter's kernels read on a bare instance. Latent geometry
# comes from the VAE (``z_dim`` / ``spatial_compression``), so a stand-in stands in
# for it; the shipped configs all patchify 2x2 on an 8x-compressed latent -> a 16px
# pixel alignment (Flux.2's VAE is already 16x on a packed latent and its DiT does
# not patchify further).
def _vae(z_dim, spatial_compression, pixel_channels=3):
    return SimpleNamespace(
        z_dim=z_dim, spatial_compression=spatial_compression, pixel_channels=pixel_channels
    )


BARE = {
    "anima": {"vae": _vae(16, 8), "dit": SimpleNamespace(patch_spatial=2)},
    "krea2": {
        "vae": _vae(16, 8),
        "dit": SimpleNamespace(config=SimpleNamespace(patch=2)),
        "_compression": 8,
    },
    "zimage": {"vae": _vae(16, 8), "dit": SimpleNamespace(patch_size=2)},
    "flux_klein": {"vae": _vae(128, 16)},
    "qwen_image": {"vae": _vae(16, 8), "dit": SimpleNamespace(patch_size=2)},
    "qwen_image21": {"vae": _vae(64, 16, pixel_channels=4)},
}

# The adapters whose step schedule shifts with the image token count. Krea 2 is
# NOT one of them: it pins ``mu=DEFAULT_MU`` (the distilled checkpoint was trained
# at a fixed shift), so its grid is resolution independent by design.
RESOLUTION_AWARE = {"flux_klein", "qwen_image", "qwen_image21"}


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


# Pixel alignment each adapter rounds request sizes to. All the shipped models land
# on 16 except Qwen-Image 2.1, which needs 32 (one Qwen3-VL vision token).
ALIGN = {"qwen_image21": 32}
DEFAULT_ALIGN = 16


@pytest.mark.parametrize("model_cls", MODEL_CATALOG, ids=CATALOG_IDS)
def test_resolve_size_aligns_to_the_patched_latent_grid(model_cls):
    """Odd sizes are rounded up to the model's pixel alignment."""
    align = ALIGN.get(model_cls.name, DEFAULT_ALIGN)
    model = _bare(model_cls, **BARE[model_cls.name])
    assert model.resolve_size(100, 60) == (round_up(100, align), round_up(60, align))
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
    "model_cls,edit,kv_cache",
    [
        (AnimaModel, False, False),
        (Krea2Model, False, False),
        (ZImageModel, False, False),
        (FluxKleinModel, True, True),
        (QwenImageModel, True, True),
        (QwenImage21Model, True, True),
    ],
    ids=CATALOG_IDS,
)
def test_model_capabilities(model_cls, edit, kv_cache):
    """``CAPABILITIES`` is the one source of truth, and it must describe the adapter.

    Both halves are checked against the machinery they name, since the pipeline
    rejects a request the model cannot serve and ``/health`` lets the UI grey out
    what the loaded model lacks:

      * ``edit``           -> the reference kernels are really overridden.
      * ``kv_cache``       -> the shared cache protocol really starts a run cache,
        and freezing reference K/V needs a reference latent, so it implies ``edit``.
    """
    model = _bare(model_cls, **BARE[model_cls.name])
    assert model.capability("edit") is edit
    assert model.capability("kv_cache") is kv_cache
    assert not kv_cache or edit  # freezing reference K/V needs a reference latent
    overrides_reference_kernels = (
        model_cls.encode_reference is not DiffusionModel.encode_reference
        and model_cls.pack_reference_latent is not DiffusionModel.pack_reference_latent
    )
    assert overrides_reference_kernels is edit
    model.start_kv_caches(replace(_params(), kv_cache=True), True, True)
    assert (model._kv_caches is not None) is kv_cache


def test_unknown_capability_is_an_error():
    """An unlisted capability raises rather than quietly reading as "cannot do it"."""
    model = _bare(AnimaModel)
    with pytest.raises(KeyError, match="unknown capability"):
        model.capability("controlnet")


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


def test_decode_keeps_every_channel_the_vae_returned():
    """An RGBA VAE's alpha survives the decode — nothing narrows it to RGB here.

    This is the whole point of the shared decode being channel-count agnostic: an
    adapter that dropped the extra channel would silently turn a model that can
    draw transparency into one that cannot.
    """
    model = _bare(QwenImage21Model)
    model.vae = _FakeVAE(out_5d=False, dtype=torch.bfloat16, channels=4)
    pixels = model.decode(torch.zeros(1, 64, 4, 4))
    assert pixels.shape == (4, 8, 8)


@pytest.mark.parametrize("model_cls", MODEL_CATALOG, ids=CATALOG_IDS)
def test_pixel_channels_is_read_off_the_vae(model_cls):
    """The adapter reports its VAE's pixel width, and RGB is the fallback."""
    model = _bare(model_cls, **BARE[model_cls.name])
    assert model.pixel_channels == model.vae.pixel_channels

    # An adapter with no VAE opinion to offer is RGB, not an AttributeError.
    bare = _bare(model_cls)
    assert bare.pixel_channels == 3


class _FakeVAE(torch.nn.Module):
    """Returns a fixed pixel tensor, optionally with the legacy frame axis."""

    def __init__(self, out_5d: bool, dtype: torch.dtype, channels: int = 3):
        super().__init__()
        self.out_5d = out_5d
        self.dtype = dtype
        self.channels = channels
        self.seen = []

    def decode_to_pixels(self, latents):
        self.seen.append((tuple(latents.shape), latents.dtype))
        pixels = torch.zeros(1, self.channels, 8, 8, dtype=self.dtype)
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


def test_get_upscaler_is_lazy_and_cached():
    """The model builds its upscaler once, on the first request that needs it.

    ``_create_upscaler`` is what loads weights, so building twice would mean
    loading twice and handing out two objects; building eagerly would tax every
    plain generation.
    """
    built = []

    class _Spy(AnimaModel):
        def _create_upscaler(self):
            built.append(self)
            return "upscaler"

    model = _bare(_Spy, _upscaler=None)
    assert built == []  # constructing the adapter loads nothing
    assert model.get_upscaler() == "upscaler"
    assert model.get_upscaler() == "upscaler"
    assert len(built) == 1
