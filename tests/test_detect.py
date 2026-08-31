"""Model-type detection tests.

Detection reads safetensors header keys only (no tensors, no weights). We test the
``detect(f)`` logic against a fake handle with synthetic keys, plus the catalog
``resolve()`` against a tiny real safetensors file.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from thenoise.models import (
    AnimaModel,
    SdxlModel,
    Krea2Model,
    ZImageModel,
    resolve,
)


class _FakeHandle:
    """Mimics ``safetensors.safe_open``'s ``keys()`` for detection."""

    def __init__(self, keys):
        self._keys = keys

    def keys(self):
        return self._keys


_ANIMA_KEYS = [
    "model.diffusion_model.blocks.0.adaln_modulation_self_attn.1.weight",
    "model.diffusion_model.llm_adapter.layers.0.weight",
    "model.diffusion_model.x_embedder.linear.weight",
]

# A transformer DiT with adaln modulation but WITHOUT Anima's LLM adapter —
# e.g. a generic adaLN-style block. Must NOT be claimed as Anima.
_ADALN_ONLY_KEYS = [
    "model.diffusion_model.blocks.0.adaln_modulation_self_attn.1.weight",
    "model.diffusion_model.blocks.0.mlp.gate.weight",
]

_KREA2_KEYS = [
    "x_embedder.linear.weight",
    "txtfusion.layerwise_blocks.0.attn.q_proj.weight",
    "blocks.0.mod.lin.weight",
    "txtmlp.1.weight",
]

_ZIMAGE_KEYS = [
    "x_embedder.weight",
    "cap_embedder.1.weight",
    "context_refiner.0.attention_norm1.weight",
    "t_embedder.mlp.0.weight",
    "layers.0.attention_norm1.weight",
]

# Z-Image repackaged under ComfyUI's generic "model.diffusion_model." wrapper.
_ZIMAGE_WRAPPED_KEYS = [
    "model.diffusion_model.x_embedder.weight",
    "model.diffusion_model.cap_embedder.1.weight",
    "model.diffusion_model.context_refiner.0.attention_norm1.weight",
    "model.diffusion_model.layers.0.attention_norm1.weight",
]


def test_anima_detect_true():
    assert AnimaModel.detect(_FakeHandle(_ANIMA_KEYS)) is True


def test_anima_rejects_adaln_without_llm_adapter():
    # adaln modulation alone is not enough — Anima requires its LLM adapter.
    assert AnimaModel.detect(_FakeHandle(_ADALN_ONLY_KEYS)) is False


def test_krea2_detect_true():
    assert Krea2Model.detect(_FakeHandle(_KREA2_KEYS)) is True


def test_krea2_rejects_anima_keys():
    assert Krea2Model.detect(_FakeHandle(_ANIMA_KEYS)) is False


def test_anima_rejects_krea2_keys():
    assert AnimaModel.detect(_FakeHandle(_KREA2_KEYS)) is False


def test_zimage_detect_true():
    assert ZImageModel.detect(_FakeHandle(_ZIMAGE_KEYS)) is True


def test_zimage_detect_true_on_wrapped_repackaged_keys():
    assert ZImageModel.detect(_FakeHandle(_ZIMAGE_WRAPPED_KEYS)) is True


def test_zimage_rejects_anima_keys():
    assert ZImageModel.detect(_FakeHandle(_ANIMA_KEYS)) is False


def test_zimage_rejects_krea2_keys():
    assert ZImageModel.detect(_FakeHandle(_KREA2_KEYS)) is False


def test_anima_rejects_zimage_keys():
    assert AnimaModel.detect(_FakeHandle(_ZIMAGE_KEYS)) is False


# SDXL is an SDXL LDM UNet: ``input_blocks`` / ``middle_block`` block
# lists plus the ``label_emb`` / ``time_embed`` conditioning MLPs.
_SDXL_KEYS = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_q.weight",
    "model.diffusion_model.label_emb.0.0.weight",
    "model.diffusion_model.time_embed.0.weight",
]

# An SDXL-style checkpoint repackaged under the ComfyUI ``model.diffusion_model.``
# wrapper (already covered by ``_SDXL_KEYS``), plus a bare (unwrapped)
# variant.
_SDXL_BARE_KEYS = [
    "input_blocks.0.0.weight",
    "middle_block.1.transformer_blocks.0.attn1.to_q.weight",
    "label_emb.0.0.weight",
    "time_embed.0.weight",
]


def test_sdxl_detect_true():
    assert SdxlModel.detect(_FakeHandle(_SDXL_KEYS)) is True


def test_sdxl_detect_true_on_bare_keys():
    assert SdxlModel.detect(_FakeHandle(_SDXL_BARE_KEYS)) is True


def test_sdxl_rejects_other_models():
    assert SdxlModel.detect(_FakeHandle(_ANIMA_KEYS)) is False
    assert SdxlModel.detect(_FakeHandle(_KREA2_KEYS)) is False
    assert SdxlModel.detect(_FakeHandle(_ZIMAGE_KEYS)) is False


def test_other_models_reject_sdxl():
    assert AnimaModel.detect(_FakeHandle(_SDXL_KEYS)) is False
    assert Krea2Model.detect(_FakeHandle(_SDXL_KEYS)) is False
    assert ZImageModel.detect(_FakeHandle(_SDXL_KEYS)) is False


def test_resolve_sdxl(tmp_path):
    p = tmp_path / "sdxl.safetensors"
    _write_safetensors(p, _SDXL_KEYS)
    assert resolve(str(p)) is SdxlModel


# A Krea2 checkpoint repackaged under ComfyUI's generic "model.diffusion_model."
# wrapper. The prefix is NOT a Krea2/Anima signature — it must be stripped
# before matching, otherwise Krea2 fails and Anima false-positives on it.
_KREA2_WRAPPED_KEYS = [
    "model.diffusion_model.x_embedder.linear.weight",
    "model.diffusion_model.txtfusion.layerwise_blocks.0.attn.q_proj.weight",
    "model.diffusion_model.blocks.0.mod.lin.weight",
    "model.diffusion_model.txtmlp.1.weight",
]


def test_krea2_detect_true_on_wrapped_repackaged_keys():
    assert Krea2Model.detect(_FakeHandle(_KREA2_WRAPPED_KEYS)) is True


def test_anima_does_not_false_positive_on_wrapped_krea2_keys():
    assert AnimaModel.detect(_FakeHandle(_KREA2_WRAPPED_KEYS)) is False


def test_resolve_wrapped_krea2(tmp_path):
    p = tmp_path / "krea2-wrapped.safetensors"
    _write_safetensors(p, _KREA2_WRAPPED_KEYS)
    assert resolve(str(p)) is Krea2Model


def _write_safetensors(path: Path, keys: dict) -> None:
    import torch
    from safetensors.torch import save_file
    save_file({k: torch.zeros(1) for k in keys}, str(path))


def test_resolve_anima(tmp_path):
    p = tmp_path / "anima.safetensors"
    _write_safetensors(p, _ANIMA_KEYS)
    assert resolve(str(p)) is AnimaModel


def test_resolve_krea2(tmp_path):
    p = tmp_path / "krea2.safetensors"
    _write_safetensors(p, _KREA2_KEYS)
    assert resolve(str(p)) is Krea2Model


def test_resolve_zimage(tmp_path):
    p = tmp_path / "zimage.safetensors"
    _write_safetensors(p, _ZIMAGE_WRAPPED_KEYS)
    assert resolve(str(p)) is ZImageModel


def test_resolve_unknown_raises(tmp_path):
    p = tmp_path / "unknown.safetensors"
    _write_safetensors(p, {"some.random.key": 0})
    with pytest.raises(ValueError):
        resolve(str(p))


@pytest.mark.skipif(
    not os.path.exists("models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors"),
    reason="real Anima checkpoint not present",
)
def test_resolve_real_anima_checkpoint():
    """Sanity-check the detector against the real Anima DiT on disk."""
    p = "models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors"
    assert resolve(p) is AnimaModel
