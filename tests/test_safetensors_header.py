"""The shared checkpoint header: one parse, read by everything.

Detection, geometry and the loaders all read the same parsed `Header`, so what matters
here is that a file is opened once and that a name test says what it looks like it says.
"""
from __future__ import annotations

import json

import pytest
import torch

from conftest import (
    TINY_DIM,
    W4A8_GROUP,
    quant_marker,
    w4a8_checkpoint,
    w4a8_marker,
    write_safetensors,
    write_wan_dit,
)
from thenoise.dit.wan_video.model import SIGNATURE as WAN_SIGNATURE, detect_params
from thenoise.utils.safetensors import (
    Header,
    MemoryEfficientSafeOpen,
    Signature,
    read_header,
)

WRAP = Signature("Wrapped thing", keys=("blocks.0.attn.weight",), absent=("audio.weight",))


# ------------------------------------------------------------------ the parse


def test_names_come_out_wrapper_normalized(tmp_path):
    """A repackaged file is read in the names the module tree uses, with the file's own
    spelling kept for the one caller that reads a tensor back by raw key."""
    path = write_safetensors(tmp_path / "dit.safetensors",
                             {"net.blocks.0.attn.weight": torch.zeros(2, 2)})
    header = read_header(path)
    assert header.names == ("blocks.0.attn.weight",)
    assert "net.blocks.0.attn.weight" in header.raw_entries
    assert header.shape("blocks.0.attn.weight") == (2, 2)
    assert header.dtypes["blocks.0.attn.weight"] == "F32"


def test_a_missing_shape_names_the_file_it_was_read_from(tmp_path):
    header = read_header(write_safetensors(tmp_path / "a.safetensors",
                                           {"x.weight": torch.zeros(1)}))
    with pytest.raises(KeyError, match="carries no 'y.weight'"):
        header.shape("y.weight")


def test_the_header_is_parsed_once_however_many_parts_ask_for_it(tmp_path, monkeypatch):
    from thenoise.models.config import ModelConfig

    path = write_safetensors(tmp_path / "dit.safetensors", {"x.weight": torch.zeros(1)})
    calls = []
    monkeypatch.setattr("thenoise.models.config.read_header",
                        lambda p: (calls.append(p), read_header(p))[1])
    config = ModelConfig(dit_paths={"": path}, vae_path="", text_encoder_path="",
                         device="cpu", offload_device="cpu")
    assert config.header() is config.header()
    assert config.all_headers() == {"": config.header()}
    assert calls == [path]


def test_a_geometry_read_handed_a_header_never_opens_the_file_again(tmp_path, monkeypatch):
    def opened(*_args, **_kwargs):
        raise AssertionError("a loader that was handed the header opened the file again")

    path = write_wan_dit(tmp_path / "dit.safetensors")
    header = read_header(path)
    monkeypatch.setattr(MemoryEfficientSafeOpen, "__init__", opened)
    assert detect_params(header).in_channels == 48


# ------------------------------------------------------------------ depth


def test_block_count_is_the_highest_index_plus_one(tmp_path):
    keys = {f"blocks.{i}.attn.weight": torch.zeros(1) for i in (0, 7, 41)}
    keys["blocks.ln.weight"] = torch.zeros(1)  # not an index
    keys["head.blocks.9.weight"] = torch.zeros(1)  # another prefix
    header = read_header(write_safetensors(tmp_path / "dit.safetensors", keys))
    assert header.block_count("blocks.") == 42
    assert header.block_count("head.blocks.") == 10
    assert header.block_count("nope.") == 0


def test_a_repackaged_file_counts_the_same_depth(tmp_path):
    keys = {f"model.diffusion_model.blocks.{i}.attn.weight": torch.zeros(1)
            for i in range(3)}
    header = read_header(write_safetensors(tmp_path / "dit.safetensors", keys))
    assert header.block_count("blocks.") == 3


# ------------------------------------------------------------------ identity


def test_an_empty_signature_claims_nothing(tmp_path):
    """A model that declared no names must not swallow every checkpoint."""
    header = read_header(write_safetensors(tmp_path / "dit.safetensors",
                                           {"x.weight": torch.zeros(1)}))
    assert not header.identifies(Signature("Nothing"))
    assert not Header("x", {}, {}, {}, {}).identifies(WAN_SIGNATURE)


def test_absent_names_are_what_a_superset_is_refused_by(tmp_path):
    """The same layout with an extra stream is a different model, not a match."""
    keys = {"blocks.0.attn.weight": torch.zeros(1)}
    assert read_header(write_safetensors(tmp_path / "image.safetensors",
                                         keys)).identifies(WRAP)
    video = dict(keys, **{"audio.weight": torch.zeros(1)})
    assert not read_header(write_safetensors(tmp_path / "video.safetensors",
                                             video)).identifies(WRAP)


def test_require_names_whichever_half_of_the_claim_failed(tmp_path):
    missing = read_header(write_safetensors(tmp_path / "a.safetensors",
                                            {"other.weight": torch.zeros(1)}))
    with pytest.raises(ValueError, match=r"does not match the Wrapped thing signature "
                                         r"\(missing \['blocks.0.attn.weight'\]\)"):
        missing.require(WRAP)

    unwanted = read_header(write_safetensors(tmp_path / "b.safetensors", {
        "blocks.0.attn.weight": torch.zeros(1), "audio.weight": torch.zeros(1)}))
    with pytest.raises(ValueError, match="it carries \\['audio.weight'\\]"):
        unwanted.require(WRAP)


# ------------------------------------------------------------------ metadata


def test_a_metadata_section_that_will_not_parse_only_costs_the_geometry(tmp_path, caplog):
    good = write_safetensors(tmp_path / "good.safetensors", {"x.weight": torch.zeros(1)},
                             metadata={"config": json.dumps({"dim": 8})})
    assert read_header(good).metadata_json("config") == {"dim": 8}

    broken = write_safetensors(tmp_path / "broken.safetensors",
                               {"x.weight": torch.zeros(1)}, metadata={"config": "{"})
    assert read_header(broken).metadata_json("config") == {}
    assert "Could not parse" in caplog.text

    absent = read_header(write_safetensors(tmp_path / "plain.safetensors",
                                           {"x.weight": torch.zeros(1)}))
    assert absent.metadata_json("config") == {}


# ------------------------------------------------------------------ renaming


def test_relabel_renames_without_reading_the_file_again(tmp_path, monkeypatch):
    header = read_header(write_safetensors(tmp_path / "dit.safetensors",
                                           {"norm.scale": torch.zeros(4)}))
    renamed = header.relabel(lambda key: key.replace(".scale", ".weight"))
    assert renamed.shape("norm.weight") == (4,)
    assert renamed.path == header.path  # same file, and no second parse needed


def test_two_names_resolving_to_one_is_refused(tmp_path):
    header = read_header(write_safetensors(tmp_path / "dit.safetensors", {
        "a.scale": torch.zeros(1), "a.weight": torch.zeros(1)}))
    with pytest.raises(RuntimeError, match="resolve to 'a.weight'"):
        header.relabel(lambda key: key.replace(".scale", ".weight"))


# ------------------------------------------------- a packed weight's true width


def test_a_packed_weight_reports_the_shape_it_fills(tmp_path):
    """Every geometry read goes through these shapes, so a packed width is resolved
    once here rather than in each model that would otherwise build a layer half as
    wide as the checkpoint means it to be.
    """
    path, _ = w4a8_checkpoint(tmp_path / "w4a8.safetensors")
    header = read_header(path)

    assert header.shape("q.weight") == TINY_DIM
    assert tuple(header.raw_entries["q.weight"]["shape"]) == (TINY_DIM[0], TINY_DIM[1] // 2)
    assert header.logical_shape("q.weight") == TINY_DIM
    # The scale siblings state the same tensor and keep the shape they are stored at.
    assert header.shape("q.weight_s_rel") == (TINY_DIM[0], TINY_DIM[1] // W4A8_GROUP)


def test_relabel_keeps_a_packed_weight_wide(tmp_path):
    """A loader that renames the checkpoint's leaves still reads the logical shape."""
    path, _ = w4a8_checkpoint(tmp_path / "w4a8.safetensors")
    renamed = read_header(path, key_map=lambda key: key.replace("q.", "layer."))
    assert renamed.shape("layer.weight") == TINY_DIM


@pytest.mark.parametrize(
    "marker,fault",
    [
        (None, "carries no"),
        (quant_marker({"format": "asym_w4a8_int8"}), "states no group size"),
        # A group size that does not divide the stored row into 4 or 6 bits cannot be
        # recovered, so it is refused rather than guessed at.
        (w4a8_marker(group_size=32), "neither 4- nor 6-bit"),
    ],
    ids=["no-marker", "no-group-size", "impossible-packing"],
)
def test_a_packed_weight_that_cannot_state_its_width_is_refused(tmp_path, marker, fault):
    path, _ = w4a8_checkpoint(tmp_path / "w4a8.safetensors", marker=marker)
    with pytest.raises(ValueError, match=fault):
        read_header(path)
