"""Generation-preference tests: checkpoint markers and the resolution precedence.

Preferences resolve as **request (API/CLI) > checkpoint marker > model default**
(``DiffusionModel.pref``). The marker layer is deliberately model-independent:
``thenoise.utils.checkpoint`` maps safetensors keys to preference values, so no
adapter mentions a checkpoint key and a model that has no use for a preference is
unaffected by a marker.
"""
from __future__ import annotations

import pytest
import torch
from PIL import Image

from conftest import EditingStubModel, StubModel, write_safetensors
from thenoise.models.config import GenerateRequest
from thenoise.pipeline import PipelineController
from thenoise.upscale.pixel import PixelUpscalerManager
from thenoise.utils.checkpoint import detect_checkpoint_prefs

ZERO_COND_KEY = "__index_timestep_zero__"


def _controller(model=None) -> PipelineController:
    manager = PixelUpscalerManager(upscaler_dir="", device="cpu")
    return PipelineController(model or StubModel(), manager)


def _request(**kwargs) -> GenerateRequest:
    return GenerateRequest(**{"prompt": "a fox", "seed": 1, **kwargs})


# ----------------------------------------------------------- checkpoint markers


def test_marker_implies_a_preference(tmp_path):
    """An edit checkpoint carrying ``__index_timestep_zero__`` implies the method."""
    plain = tmp_path / "plain.safetensors"
    write_safetensors(plain, {"img_in.weight": torch.zeros(1), "txt_in.weight": torch.zeros(1)})
    assert detect_checkpoint_prefs(str(plain)) == {}

    zero_cond = tmp_path / "zero_cond.safetensors"
    write_safetensors(zero_cond, {ZERO_COND_KEY: torch.zeros(1)})
    assert detect_checkpoint_prefs(str(zero_cond)) == {"ref_method": "index_timestep_zero"}


def test_marker_matching_is_wrapper_prefix_agnostic(tmp_path):
    """A repackaged checkpoint marks the same thing (prefixes are stripped)."""
    wrapped = tmp_path / "wrapped.safetensors"
    write_safetensors(wrapped, {f"model.diffusion_model.{ZERO_COND_KEY}": torch.zeros(1)})
    assert detect_checkpoint_prefs(str(wrapped)) == {"ref_method": "index_timestep_zero"}


def test_unreadable_checkpoint_yields_no_prefs(tmp_path):
    """Markers are an optional hint: an unreadable file must not fail the load."""
    assert detect_checkpoint_prefs(str(tmp_path / "missing.safetensors")) == {}


# ---------------------------------------------------------------- precedence


def test_pref_falls_back_to_the_model_default():
    model = StubModel()
    assert model.checkpoint_prefs == {}
    assert model.pref("ref_method") == "index"
    assert model.pref("kv_cache") is False


def test_pref_checkpoint_beats_default_and_request_beats_checkpoint():
    model = StubModel()
    # An explicit request wins with nothing in the checkpoint to say otherwise:
    # the official edit checkpoints are themselves unmarked, so a missing marker
    # must never veto a request.
    assert model.pref("ref_method", "index_timestep_zero") == "index_timestep_zero"
    assert model.pref("ref_method", None) == "index"  # auto -> default

    model.checkpoint_prefs = {"ref_method": "index_timestep_zero"}
    assert model.pref("ref_method") == "index_timestep_zero"  # layer 2 wins over 3
    assert model.pref("ref_method", "index") == "index"  # the request still wins


def test_pref_rejects_unknown_names():
    """Asking for an unregistered preference is a bug, not a silent default."""
    with pytest.raises(KeyError, match="unknown preference"):
        StubModel().pref("upscale_type")


# ------------------------------------------------------- pipeline resolution


def test_resolve_takes_the_reference_method_from_the_checkpoint():
    """The marker layer feeds the pipeline: no request field, method auto-detected."""
    model = EditingStubModel()
    model.checkpoint_prefs = {"ref_method": "index_timestep_zero"}
    r = _controller(model)._resolve_pipeline(_request())
    assert r.ref_method == "index_timestep_zero"
    # The marker implies the *conditioning*, not the cache (that stays opt-in).
    assert r.kv_cache is False


def test_resolve_kv_cache_pulls_in_the_reference_method():
    """``kv_cache`` on with the method on auto picks the method that makes it valid."""
    r = _controller(EditingStubModel())._resolve_pipeline(_request(kv_cache=True))
    assert r.kv_cache is True
    assert r.ref_method == "index_timestep_zero"


def test_resolve_explicit_index_with_kv_cache_is_rejected():
    """An explicit ``index`` stays explicit: the frozen K/V would not be valid."""
    with pytest.raises(ValueError, match="index_timestep_zero"):
        _controller(EditingStubModel())._resolve_pipeline(
            _request(kv_cache=True, ref_method="index")
        )


def test_resolve_kv_cache_stays_off_by_default():
    r = _controller(EditingStubModel())._resolve_pipeline(_request())
    assert r.kv_cache is False
    assert r.ref_method == "index"


def test_edit_passes_the_resolved_method_to_the_model():
    """The resolved preference — not the raw request — reaches ``prepare_latent``."""
    model = EditingStubModel()
    model.checkpoint_prefs = {"ref_method": "index_timestep_zero"}
    controller = _controller(model)
    controller.edit(_request(image=Image.new("RGB", (64, 64), "white"), steps=1))
    assert model.prepared[-1]["method"] == "index_timestep_zero"
