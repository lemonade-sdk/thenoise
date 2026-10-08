"""Generation preferences: checkpoint markers and the resolution precedence.

Preferences resolve as **request (API/CLI) > checkpoint marker > model default**
(``DiffusionModel.pref``). The marker layer is model-independent: one registry in
``thenoise.utils.checkpoint`` both implies preferences and drives the loader's
dropping of those keys, so no adapter names a checkpoint key.
"""
from __future__ import annotations

import pytest
import torch
from PIL import Image

from conftest import EditingStubModel, StubModel, write_safetensors
from thenoise.models.config import GenerateRequest
from thenoise.pipeline import PipelineController
from thenoise.upscale.pixel import PixelUpscalerManager
from thenoise.utils.checkpoint import CHECKPOINT_MARKERS, detect_checkpoint_prefs
from thenoise.utils.loader import drop_checkpoint_markers

ZERO_COND_KEY = "__index_timestep_zero__"


def _controller(model=None) -> PipelineController:
    manager = PixelUpscalerManager(upscaler_dir="", device="cpu")
    return PipelineController(model or StubModel(), manager)


def _request(**kwargs) -> GenerateRequest:
    return GenerateRequest(**{"prompt": "a fox", "seed": 1, **kwargs})


def _edit_request(**kwargs) -> GenerateRequest:
    return _request(image=Image.new("RGB", (64, 64), "white"), **kwargs)


# ----------------------------------------------------------- checkpoint markers


def test_marker_implies_a_preference(tmp_path):
    """An edit checkpoint carrying ``__index_timestep_zero__`` implies the method —
    raw or repackaged under the generic wrapper prefix.
    """
    plain = tmp_path / "plain.safetensors"
    write_safetensors(plain, {"img_in.weight": torch.zeros(1), "txt_in.weight": torch.zeros(1)})
    assert detect_checkpoint_prefs(str(plain)) == {}

    zero_cond = tmp_path / "zero_cond.safetensors"
    write_safetensors(zero_cond, {ZERO_COND_KEY: torch.zeros(1)})
    assert detect_checkpoint_prefs(str(zero_cond)) == {"ref_method": "index_timestep_zero"}

    wrapped = tmp_path / "wrapped.safetensors"
    write_safetensors(wrapped, {f"model.diffusion_model.{ZERO_COND_KEY}": torch.zeros(1)})
    assert detect_checkpoint_prefs(str(wrapped)) == {"ref_method": "index_timestep_zero"}


def test_unreadable_checkpoint_yields_no_prefs(tmp_path):
    """Markers are an optional hint: an unreadable file must not fail the load."""
    assert detect_checkpoint_prefs(str(tmp_path / "missing.safetensors")) == {}


@pytest.mark.parametrize("marker", CHECKPOINT_MARKERS, ids=lambda m: m.key)
def test_a_registered_marker_is_also_dropped_from_weights(marker):
    """One registry entry covers reading *and* dropping, so a marker can never leak
    into a strict ``load_state_dict`` for any model, raw or repackaged.
    """
    kept = {"img_in.weight": torch.zeros(1)}
    sd = {
        marker.key: torch.zeros(1),
        f"model.diffusion_model.{marker.key}": torch.zeros(1),
        **kept,
    }
    assert drop_checkpoint_markers(sd) == kept


# ---------------------------------------------------------------- precedence


def test_pref_falls_back_to_the_model_default():
    model = StubModel()
    assert model.checkpoint_prefs == {}
    assert model.pref("ref_method") == "index"


def test_pref_checkpoint_beats_default_and_request_beats_checkpoint():
    model = StubModel()
    # The official edit checkpoints are themselves unmarked, so a missing marker
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
    # The KV cache is derived from the resolved reference method, never a preference.
    with pytest.raises(KeyError, match="unknown preference"):
        StubModel().pref("kv_cache")


# ------------------------------------------------------- pipeline resolution


class _KvCacheModel(EditingStubModel):
    """An editing adapter that wired the reference KV cache into its blocks."""

    CAPABILITIES = {**EditingStubModel.CAPABILITIES, "kv_cache": True}


class _ZeroCondModel(_KvCacheModel):
    """An adapter conditioning its references at timestep zero (Qwen-Image 2.1)."""

    DEFAULT_PREFS = {
        **EditingStubModel.DEFAULT_PREFS,
        "ref_method": "index_timestep_zero",
    }


def test_resolve_takes_the_reference_method_from_the_checkpoint():
    """The marker layer feeds the pipeline: no request field, method auto-detected."""
    model = EditingStubModel()
    model.checkpoint_prefs = {"ref_method": "index_timestep_zero"}
    r = _controller(model)._resolve_pipeline(_request(), is_edit=False)
    assert r.ref_method == "index_timestep_zero"
    assert r.kv_cache is False  # a generation has no reference K/V to freeze


def test_the_reference_method_decides_the_cache():
    """Auto cache = timestep-zero conditioning on an edit."""
    zero = _controller(_ZeroCondModel())._resolve_pipeline(_edit_request(), is_edit=True)
    assert (zero.kv_cache, zero.ref_method) == (True, "index_timestep_zero")

    # ``index`` re-conditions the references every step, so freezing them is off.
    assert _controller(_KvCacheModel())._resolve_pipeline(
        _edit_request(), is_edit=True
    ).kv_cache is False


def test_the_cache_default_respects_the_capability():
    """Auto never asks an adapter for a cache it never wired in."""
    model = _ZeroCondModel(capabilities={"kv_cache": False})
    r = _controller(model)._resolve_pipeline(_edit_request(), is_edit=True)
    assert r.kv_cache is False


def test_the_cache_default_stays_off_for_a_generation():
    """Even on a model whose reference method is timestep-zero by default."""
    controller = _controller(_ZeroCondModel())
    assert controller._resolve_pipeline(_request(), is_edit=False).kv_cache is False


def test_resolve_kv_cache_pulls_in_the_reference_method():
    """``kv_cache`` on with the method on auto picks the method that makes it valid."""
    r = _controller(_KvCacheModel())._resolve_pipeline(
        _edit_request(kv_cache=True), is_edit=True
    )
    assert r.kv_cache is True
    assert r.ref_method == "index_timestep_zero"


def test_resolve_explicit_index_with_kv_cache_is_rejected():
    """An explicit ``index`` stays explicit: the frozen K/V would not be valid."""
    with pytest.raises(ValueError, match="index_timestep_zero"):
        _controller(_KvCacheModel())._resolve_pipeline(
            _edit_request(kv_cache=True, ref_method="index"), is_edit=True
        )


def test_an_explicit_false_wins_over_the_derived_default():
    r = _controller(_ZeroCondModel())._resolve_pipeline(
        _edit_request(kv_cache=False), is_edit=True
    )
    assert r.kv_cache is False


def test_an_explicit_kv_cache_without_a_reference_is_still_refused():
    """Only the derived default yields to a plain generation; a request does not."""
    controller = _controller(_ZeroCondModel())
    with pytest.raises(ValueError, match="requires an edit request"):
        controller.generate(_request(kv_cache=True, steps=1))


def test_edit_passes_the_resolved_method_to_the_model():
    """The resolved preference — not the raw request — reaches ``prepare_latent``."""
    model = EditingStubModel()
    model.checkpoint_prefs = {"ref_method": "index_timestep_zero"}
    controller = _controller(model)
    controller.edit(_request(image=Image.new("RGB", (64, 64), "white"), steps=1))
    assert model.prepared[-1]["method"] == "index_timestep_zero"
