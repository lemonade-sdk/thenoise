"""Upscale factor/type planning + pixel-upscaler manager tests (no torch, no weights).

Pixel-upscaling is a pixel-space / server concern, so its manager
(``PixelUpscalerManager``) and the pipeline controller's upscale planning
(``PipelineController._resolve_upscale`` / ``_pixel_upscaler_scale_for``) are
tested directly, without a diffusion model.
"""
from __future__ import annotations

import pytest

from thenoise.upscale.pixel import PixelUpscalerManager


def _make_manager(upscaler_scales=None, upscaler_dir="/tmp"):
    """Build a manager with injected per-name detected scales (skips file access)."""
    m = PixelUpscalerManager(upscaler_dir=upscaler_dir, device="cuda")
    m._pixel_upscaler_scales = dict(upscaler_scales or {})
    return m


def _make_controller(upscaler_scales=None):
    """Build a pipeline controller over a minimal fake model + manager."""
    from thenoise.pipeline import PipelineController

    class _Model:
        UPSCALE_SCALE = 2

    mgr = _make_manager(upscaler_scales)
    return PipelineController(_Model(), mgr)


# ------------------------------------------------------------- manager: scale
@pytest.mark.parametrize(
    "model_scale,upscaler_scale,factor,upscale_type,expected",
    [
        # refined: the latent path gives 2x, so the pixel upscaler is only needed
        # above that.
        (2, 4, 1.0, "refined", 0),
        (2, 4, 2.0, "refined", 0),
        (2, 4, 2.5, "refined", 4),
        (2, 4, 8.0, "refined", 4),
        (2, 4, 0.5, "refined", 0),  # sub-unit factor: no upscale at all
        # no-refiner: no latent multiplier, so any upscale needs the pixel model.
        (2, 4, 1.5, "no-refiner", 4),
        (2, 4, 4.0, "no-refiner", 4),
        (2, 4, 0.5, "no-refiner", 0),
        (2, 2, 2.0, "refined", 0),
        (2, 2, 2.5, "refined", 2),
        (2, 2, 4.0, "refined", 2),
        (2, 2, 1.5, "no-refiner", 2),
        (2, 2, 2.0, "no-refiner", 2),
        # A model with a bigger latent scale (4) moves the refined threshold.
        (4, 4, 4.0, "refined", 0),
        (4, 4, 4.5, "refined", 4),
        # No pixel upscaler selected -> never applied.
        (2, 0, 4.0, "refined", 0),
    ],
)
def test_pixel_upscaler_scale_mapping(model_scale, upscaler_scale, factor, upscale_type, expected):
    scales = {} if not upscaler_scale else {"up": upscaler_scale}
    controller = _make_controller(scales)
    controller.model.UPSCALE_SCALE = model_scale
    name = "up" if upscaler_scale else None
    assert controller._pixel_upscaler_scale_for(factor, upscale_type, name) == expected


# ------------------------------------------------------------- controller: resolve
def test_resolve_valid_refined_without_pixel_upscaler():
    c = _make_controller()
    # f <= latent 2x in refined mode needs no pixel upscaler.
    assert c._resolve_upscale(1.0, "refined") == (1.0, "refined")
    assert c._resolve_upscale(2.0, "refined") == (2.0, "refined")


def test_resolve_needs_pixel_upscaler_when_absent():
    c = _make_controller()
    with pytest.raises(ValueError):
        c._resolve_upscale(2.5, "refined")
    with pytest.raises(ValueError):
        c._resolve_upscale(1.5, "no-refiner")


def test_resolve_max_ranges_depend_on_scale():
    # 4x model: refined up to 8, no-refiner up to 4.
    c4 = _make_controller(upscaler_scales={"x4": 4})
    assert c4._resolve_upscale(8.0, "refined", "x4") == (8.0, "refined")
    assert c4._resolve_upscale(4.0, "no-refiner", "x4") == (4.0, "no-refiner")
    with pytest.raises(ValueError):
        c4._resolve_upscale(5.0, "no-refiner", "x4")
    with pytest.raises(ValueError):
        c4._resolve_upscale(9.0, "refined", "x4")

    # 2x model: refined up to 4, no-refiner up to 2.
    c2 = _make_controller(upscaler_scales={"x2": 2})
    assert c2._resolve_upscale(4.0, "refined", "x2") == (4.0, "refined")
    assert c2._resolve_upscale(2.0, "no-refiner", "x2") == (2.0, "no-refiner")
    with pytest.raises(ValueError):
        c2._resolve_upscale(5.0, "refined", "x2")
    with pytest.raises(ValueError):
        c2._resolve_upscale(3.0, "no-refiner", "x2")


def test_resolve_invalid_factor():
    c = _make_controller(upscaler_scales={"x4": 4})
    for bad in (0.0, -1.0, 8.5):
        with pytest.raises(ValueError):
            c._resolve_upscale(bad, "refined", "x4")


def test_resolve_invalid_type():
    c = _make_controller(upscaler_scales={"x4": 4})
    with pytest.raises(ValueError):
        c._resolve_upscale(2.0, "bogus")
    # the old 'fast' name is gone
    with pytest.raises(ValueError):
        c._resolve_upscale(2.0, "fast")


# ------------------------------------------------------------- manager: validate/list
def test_validate_pixel_upscaler_requires_dir_and_file():
    m = _make_manager(upscaler_dir="")
    with pytest.raises(ValueError, match="no pixel upscaler configured"):
        m.validate("x4")


def test_validate_pixel_upscaler_strips_suffix(tmp_path):
    (tmp_path / "RealESRGAN_x4.safetensors").write_text("x")
    m = _make_manager(upscaler_dir=str(tmp_path))
    assert m.validate("RealESRGAN_x4.safetensors") == "RealESRGAN_x4"
    assert m.validate("RealESRGAN_x4") == "RealESRGAN_x4"


# --------------------------------------------------- standalone upscale controller
def _make_upscale_controller(upscaler_dir="/tmp", scales=None):
    """Build a PixelUpscaleController over an injected-scale manager."""
    from thenoise.upscale_controller import PixelUpscaleController

    m = PixelUpscalerManager(upscaler_dir=upscaler_dir, device="cpu")
    m._pixel_upscaler_scales = dict(scales or {})
    return PixelUpscaleController(m)


def test_upscale_controller_factor_validation(tmp_path, monkeypatch):
    """The 0.0 sentinel means "native scale"; anything outside [1, scale] is out."""
    (tmp_path / "x4.safetensors").write_text("x")
    c = _make_upscale_controller(upscaler_dir=str(tmp_path), scales={"x4": 4})

    # Stub the pixel conversion so we can observe whether validation passed (the
    # controller would otherwise try to load the model onto a device).
    def _boom(_img):
        raise RuntimeError("reached-pixels")

    monkeypatch.setattr("thenoise.upscale_controller.pil_to_pixels", _boom)

    # 0.0 resolves to the detected scale (4) and passes validation, so the flow
    # proceeds past validation into the pixel conversion.
    with pytest.raises(RuntimeError, match="reached-pixels"):
        c.upscale(object(), 0.0, "x4")
    # A factor above the native scale, and a positive sub-unit factor, are both
    # rejected before any pixel work.
    for bad in (5, 5.0, 0.5):
        with pytest.raises(ValueError, match="upscale_factor"):
            c.upscale(object(), bad, "x4")


def test_upscale_controller_attaches_resolved_factor(tmp_path, monkeypatch):
    from PIL import Image

    (tmp_path / "x4.safetensors").write_text("x")
    c = _make_upscale_controller(upscaler_dir=str(tmp_path), scales={"x4": 4})
    img = Image.new("RGB", (2, 2))

    # Stub the pixel/tensor helpers so the flow runs without torch/weights.
    class _Pixels:
        def to(self, _device):
            return self

    monkeypatch.setattr("thenoise.upscale_controller.pil_to_pixels", lambda _i: _Pixels())
    monkeypatch.setattr("thenoise.upscale_controller.pixels_to_pil", lambda _p: img)
    monkeypatch.setattr("thenoise.upscale_controller.resize_to_target", lambda p, _w, _h: p)
    monkeypatch.setattr("thenoise.upscale_controller.build_upscale_pnginfo", lambda *a, **k: {})
    c._pixel_upscalers.apply = lambda _name, pixels, _scale: pixels

    # 0.0 sentinel -> the detected scale (4) should be exposed on the result.
    out = c.upscale(img, 0.0, "x4")
    assert out._upscale_factor == 4
    # an explicit factor is exposed as-is.
    out = c.upscale(img, 2, "x4")
    assert out._upscale_factor == 2


def test_upscale_controller_requires_upscaler_dir():
    # no upscaler dir configured -> "no pixel upscaler"
    c = _make_upscale_controller(upscaler_dir="", scales={})
    with pytest.raises(ValueError, match="no pixel upscaler"):
        c.upscale(object(), 2, "x4")

    # upscaler dir configured but named model missing -> "not found"
    c = _make_upscale_controller(upscaler_dir="/tmp", scales={})
    with pytest.raises(ValueError, match="not found"):
        c.upscale(object(), 2, "missing")


def test_list_pixel_upscalers(tmp_path):
    (tmp_path / "a.safetensors").write_text("x")
    (tmp_path / "not_a_model.txt").write_text("x")
    m = _make_manager(upscaler_dir=str(tmp_path))
    assert m.list() == ["a"]

    m.upscaler_dir = ""
    assert m.list() == []


def test_switch_pixel_upscaler_keeps_last_used(tmp_path, monkeypatch):
    """Only the last-used pixel upscaler stays loaded."""
    (tmp_path / "x2.safetensors").write_text("x")
    (tmp_path / "x4.safetensors").write_text("x")
    m = _make_manager(upscaler_dir=str(tmp_path))

    calls = []
    fake_model = object()
    from thenoise.upscale import load_pixel_upscaler as _real_load
    def _fake_load(path, device):
        calls.append(path)
        scale = 2 if "x2" in path else 4
        return fake_model, scale
    monkeypatch.setattr("thenoise.upscale.pixel.load_pixel_upscaler", _fake_load)

    m.switch("x2")
    assert m._pixel_upscaler_name == "x2"
    assert m._pixel_upscaler is fake_model

    m.switch("x2")  # same -> no-op
    assert len(calls) == 1

    m.switch("x4")  # different -> swap
    assert m._pixel_upscaler_name == "x4"
    assert m._pixel_upscaler is fake_model
    assert len(calls) == 2
    assert m._pixel_upscaler_scales == {"x2": 2, "x4": 4}


def test_forward_tiled_pads_odd_dimensions():
    """Odd-dimension inputs (non-divisible edge tiles) must not crash.

    Scale-2 ESRGAN pixel-unshuffles the input by 2, so any non-divisible tile
    size crashes the reshape. ``forward_tiled`` must pad to a multiple of
    ``scale`` and crop back, for both scale 2 and 4.
    """
    import torch
    import torch.nn.functional as F
    from thenoise.upscale.esrgan import RRDBNet

    for scale in (2, 4):
        model = RRDBNet(scale=scale, num_feat=8, num_block=1, num_grow_ch=4)
        model.eval()
        h, w = 17, 15  # odd, and multi-tile under a small tile size
        x = torch.randn(1, 3, h, w)
        out = model.forward_tiled(x, tile_size=8, tile_pad=2)
        assert out.shape == (1, 3, h * scale, w * scale)

        # Cross-check: tiling an explicitly pre-padded image and cropping must
        # match the internal pad/crop, i.e. only the original region is kept.
        padded = F.pad(x, (0, (-w) % scale, 0, (-h) % scale))
        ref = model.forward_tiled(padded, tile_size=8, tile_pad=2)
        assert torch.allclose(out, ref[:, :, : h * scale, : w * scale], atol=1e-5)
