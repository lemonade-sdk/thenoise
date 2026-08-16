"""Upscale factor/type planning tests (no torch, no weights)."""
from __future__ import annotations

import pytest

from thenoise.models.base import DiffusionModel


def _make_model(upscaler_scales=None, upscaler_dir="/tmp"):
    """Build a minimal concrete subclass (bypassing __init__ / VAE).

    ``upscaler_scales`` maps pixel-upscaler name -> detected scale, injected into
    ``_pixel_upscaler_scales`` to skip file access (mirrors the old
    ``_esrgan_scale_val`` shortcut).
    """
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
    m.device = "cuda"
    m.upscaler_dir = upscaler_dir
    m._pixel_upscaler_name = None
    m._pixel_upscaler_scales = dict(upscaler_scales or {})
    return m


def test_pixel_upscaler_scale_mapping_with_4x():
    m = _make_model(upscaler_scales={"x4": 4})
    # refined: latent gives 2x, pixel upscaler 4x only above the latent 2x.
    assert m._pixel_upscaler_scale_for(1.0, "refined", "x4") == 0
    assert m._pixel_upscaler_scale_for(2.0, "refined", "x4") == 0
    assert m._pixel_upscaler_scale_for(2.5, "refined", "x4") == 4
    assert m._pixel_upscaler_scale_for(8.0, "refined", "x4") == 4
    assert m._pixel_upscaler_scale_for(0.5, "refined", "x4") == 0
    # no-refiner: no latent multiplier, always pixel upscaler for any upscale.
    assert m._pixel_upscaler_scale_for(1.5, "no-refiner", "x4") == 4
    assert m._pixel_upscaler_scale_for(4.0, "no-refiner", "x4") == 4
    assert m._pixel_upscaler_scale_for(0.5, "no-refiner", "x4") == 0


def test_pixel_upscaler_scale_mapping_with_2x():
    m = _make_model(upscaler_scales={"x2": 2})
    assert m._pixel_upscaler_scale_for(2.0, "refined", "x2") == 0
    assert m._pixel_upscaler_scale_for(2.5, "refined", "x2") == 2
    assert m._pixel_upscaler_scale_for(4.0, "refined", "x2") == 2
    assert m._pixel_upscaler_scale_for(1.5, "no-refiner", "x2") == 2
    assert m._pixel_upscaler_scale_for(2.0, "no-refiner", "x2") == 2


def test_resolve_valid_refined_without_pixel_upscaler():
    m = _make_model()
    # f <= latent 2x in refined mode needs no pixel upscaler.
    assert m._resolve_upscale(1.0, "refined") == (1.0, "refined")
    assert m._resolve_upscale(2.0, "refined") == (2.0, "refined")


def test_resolve_needs_pixel_upscaler_when_absent():
    m = _make_model()
    with pytest.raises(ValueError):
        m._resolve_upscale(2.5, "refined")
    with pytest.raises(ValueError):
        m._resolve_upscale(1.5, "no-refiner")


def test_resolve_max_ranges_depend_on_scale():
    # 4x model: refined up to 8, no-refiner up to 4.
    m4 = _make_model(upscaler_scales={"x4": 4})
    assert m4._resolve_upscale(8.0, "refined", "x4") == (8.0, "refined")
    assert m4._resolve_upscale(4.0, "no-refiner", "x4") == (4.0, "no-refiner")
    with pytest.raises(ValueError):
        m4._resolve_upscale(5.0, "no-refiner", "x4")
    with pytest.raises(ValueError):
        m4._resolve_upscale(9.0, "refined", "x4")

    # 2x model: refined up to 4, no-refiner up to 2.
    m2 = _make_model(upscaler_scales={"x2": 2})
    assert m2._resolve_upscale(4.0, "refined", "x2") == (4.0, "refined")
    assert m2._resolve_upscale(2.0, "no-refiner", "x2") == (2.0, "no-refiner")
    with pytest.raises(ValueError):
        m2._resolve_upscale(5.0, "refined", "x2")
    with pytest.raises(ValueError):
        m2._resolve_upscale(3.0, "no-refiner", "x2")


def test_resolve_invalid_factor():
    m = _make_model(upscaler_scales={"x4": 4})
    for bad in (0.0, -1.0, 8.5):
        with pytest.raises(ValueError):
            m._resolve_upscale(bad, "refined", "x4")


def test_resolve_invalid_type():
    m = _make_model(upscaler_scales={"x4": 4})
    with pytest.raises(ValueError):
        m._resolve_upscale(2.0, "bogus")
    # the old 'fast' name is gone
    with pytest.raises(ValueError):
        m._resolve_upscale(2.0, "fast")


def test_validate_pixel_upscaler_requires_dir_and_file():
    m = _make_model()
    with pytest.raises(ValueError, match="no pixel upscaler configured"):
        m.upscaler_dir = ""
        m._validate_pixel_upscaler("x4")


def test_validate_pixel_upscaler_strips_suffix(tmp_path):
    (tmp_path / "RealESRGAN_x4.safetensors").write_text("x")
    m = _make_model(upscaler_dir=str(tmp_path))
    assert m._validate_pixel_upscaler("RealESRGAN_x4.safetensors") == "RealESRGAN_x4"
    assert m._validate_pixel_upscaler("RealESRGAN_x4") == "RealESRGAN_x4"
    with pytest.raises(ValueError, match="not found"):
        m._validate_pixel_upscaler("missing")


def test_list_pixel_upscalers(tmp_path):
    (tmp_path / "a.safetensors").write_text("x")
    (tmp_path / "not_a_model.txt").write_text("x")
    m = _make_model(upscaler_dir=str(tmp_path))
    assert m.list_pixel_upscalers() == ["a"]

    m.upscaler_dir = ""
    assert m.list_pixel_upscalers() == []


def test_switch_pixel_upscaler_keeps_last_used(tmp_path, monkeypatch):
    """Only the last-used pixel upscaler stays loaded."""
    (tmp_path / "x2.safetensors").write_text("x")
    (tmp_path / "x4.safetensors").write_text("x")
    m = _make_model(upscaler_dir=str(tmp_path))

    calls = []
    fake_model = object()
    from thenoise.upscale import load_pixel_upscaler as _real_load
    def _fake_load(path, device):
        calls.append(path)
        scale = 2 if "x2" in path else 4
        return fake_model, scale
    monkeypatch.setattr("thenoise.models.base.load_pixel_upscaler", _fake_load)

    m._switch_pixel_upscaler("x2")
    assert m._pixel_upscaler_name == "x2"
    assert m._pixel_upscaler is fake_model

    m._switch_pixel_upscaler("x2")  # same -> no-op
    assert len(calls) == 1

    m._switch_pixel_upscaler("x4")  # different -> swap
    assert m._pixel_upscaler_name == "x4"
    assert m._pixel_upscaler is fake_model
    assert len(calls) == 2
    assert m._pixel_upscaler_scales == {"x2": 2, "x4": 4}
