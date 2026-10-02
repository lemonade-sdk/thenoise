"""Model-type detection: every adapter claims exactly its own checkpoints.

Detection reads safetensors header *names* only, so the matrix is checked against
synthetic key-sets (``conftest.MODEL_KEYSETS``) plus ``resolve()`` on throwaway
one-scalar files.
"""
from __future__ import annotations

import pytest

from conftest import CATALOG_IDS, MODEL_KEYSETS, FakeHandle, write_key_checkpoint
from thenoise.models import MODEL_CATALOG, resolve


@pytest.mark.parametrize("keyset", MODEL_KEYSETS)
@pytest.mark.parametrize("model", MODEL_CATALOG, ids=CATALOG_IDS)
def test_detection_matrix(model, keyset):
    """The key-sets include the ``model.diffusion_model.``-wrapped repackagings, so
    the prefix stripping and the Anima false-positive guard stay covered.
    """
    owner, keys = MODEL_KEYSETS[keyset]
    assert model.detect(FakeHandle(keys)) is (model is owner)


@pytest.mark.parametrize("keyset", ["zimage", "unknown"])
def test_resolve(keyset, tmp_path):
    """``resolve()`` opens the file, iterates the catalog and returns the claimant."""
    owner, keys = MODEL_KEYSETS[keyset]
    path = write_key_checkpoint(tmp_path / f"{keyset}.safetensors", keys)
    if owner is None:
        with pytest.raises(ValueError, match="could not determine model type"):
            resolve(path)
    else:
        assert resolve(path) is owner
