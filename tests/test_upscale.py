"""Upscale factor/type planning tests (no torch, no weights)."""
from __future__ import annotations

import pytest

from thenoise.models.base import DiffusionModel


def _make_model(esrgan_path=None, esrgan_scale=None):
    """Build a minimal concrete subclass (bypassing __init__ / VAE)."""

    class _M(DiffusionModel):
        name = "test"

        @staticmethod
        def detect(f):
            return False

        def encode_prompt(self, prompt, negative_prompt="", *, guidance_scale):
            pass

        def init_latents(self, height, width, seed):
            pass

        def schedule(self, steps, height, width):
            pass

        def denoise_step(self, latents, t, cond, guidance_scale, i):
            pass

        def _upscale_format(self):
            return "wan21"

    m = object.__new__(_M)
    m.esrgan_path = esrgan_path
    m._esrgan_scale_val = esrgan_scale
    return m


def test_esrgan_scale_mapping_with_4x():
    m = _make_model(esrgan_path="/tmp/x4.safetensors", esrgan_scale=4)
    # refined: latent gives 2x, ESRGAN 4x only above the latent 2x.
    assert m._esrgan_scale_for(1.0, "refined") == 0
    assert m._esrgan_scale_for(2.0, "refined") == 0
    assert m._esrgan_scale_for(2.5, "refined") == 4
    assert m._esrgan_scale_for(8.0, "refined") == 4
    assert m._esrgan_scale_for(0.5, "refined") == 0
    # fast: no latent multiplier, always ESRGAN for any upscale.
    assert m._esrgan_scale_for(1.5, "fast") == 4
    assert m._esrgan_scale_for(4.0, "fast") == 4
    assert m._esrgan_scale_for(0.5, "fast") == 0


def test_esrgan_scale_mapping_with_2x():
    m = _make_model(esrgan_path="/tmp/x2.safetensors", esrgan_scale=2)
    assert m._esrgan_scale_for(2.0, "refined") == 0
    assert m._esrgan_scale_for(2.5, "refined") == 2
    assert m._esrgan_scale_for(4.0, "refined") == 2
    assert m._esrgan_scale_for(1.5, "fast") == 2
    assert m._esrgan_scale_for(2.0, "fast") == 2


def test_resolve_valid_refined_without_esrgan():
    m = _make_model(esrgan_path=None)
    # f <= latent 2x in refined mode needs no ESRGAN.
    assert m._resolve_upscale(1.0, "refined") == (1.0, "refined")
    assert m._resolve_upscale(2.0, "refined") == (2.0, "refined")


def test_resolve_needs_esrgan_when_absent():
    m = _make_model(esrgan_path=None)
    with pytest.raises(ValueError):
        m._resolve_upscale(2.5, "refined")
    with pytest.raises(ValueError):
        m._resolve_upscale(1.5, "fast")


def test_resolve_max_ranges_depend_on_scale():
    # 4x model: refined up to 8, fast up to 4.
    m4 = _make_model("/tmp/x4.safetensors", esrgan_scale=4)
    assert m4._resolve_upscale(8.0, "refined") == (8.0, "refined")
    assert m4._resolve_upscale(4.0, "fast") == (4.0, "fast")
    with pytest.raises(ValueError):
        m4._resolve_upscale(5.0, "fast")
    with pytest.raises(ValueError):
        m4._resolve_upscale(9.0, "refined")

    # 2x model: refined up to 4, fast up to 2.
    m2 = _make_model("/tmp/x2.safetensors", esrgan_scale=2)
    assert m2._resolve_upscale(4.0, "refined") == (4.0, "refined")
    assert m2._resolve_upscale(2.0, "fast") == (2.0, "fast")
    with pytest.raises(ValueError):
        m2._resolve_upscale(5.0, "refined")
    with pytest.raises(ValueError):
        m2._resolve_upscale(3.0, "fast")


def test_resolve_invalid_factor():
    m = _make_model("/tmp/x4.safetensors", esrgan_scale=4)
    for bad in (0.0, -1.0, 8.5):
        with pytest.raises(ValueError):
            m._resolve_upscale(bad, "refined")


def test_resolve_invalid_type():
    m = _make_model("/tmp/x4.safetensors", esrgan_scale=4)
    with pytest.raises(ValueError):
        m._resolve_upscale(2.0, "bogus")
