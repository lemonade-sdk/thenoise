"""The post-decode declaration is the single source of truth.

Every knob lives once in :class:`PostprocessOptions`; the CLI flags, the wire fields,
the PNG record, the chain order and the UI schema are all read off it. These tests are
what makes that true rather than merely intended: adding a knob to the declaration is
supposed to be enough, and dropping one is supposed to fail something here.
"""
from __future__ import annotations

import pytest

from thenoise.api import Text2ImageRequest
from thenoise.cli import build_parser
from thenoise.models.config import GenerateRequest
from thenoise.postprocess import CHAIN, FILTERS, POSTPROCESS_NAMES, PostprocessOptions
from thenoise.postprocess.options import DECODE, FINAL, filter_specs, flat_options
from thenoise.utils.options import option_meta, options_schema

# The checkpoints every generation subcommand needs.
PATHS = ["generate", "--dit", "d", "--vae", "v", "--text-encoder", "t", "--prompt", "x"]


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


#: The knobs as the web UI sees them, which is also the quickest way to a knob's type.
_SCHEMA = {entry["name"]: entry for entry in options_schema(PostprocessOptions)}


# ----------------------------------------------------------------- the declaration


def test_declaration_order_is_chain_order():
    """The filters run in field order, which is what makes a knob's place meaningful."""
    assert [spec.name for spec in CHAIN] == ["nyquist_notch", "color", "rcas", "film_grain"]


def test_the_colour_knobs_are_one_filter_reading_four_knobs():
    """Four knobs, one pass over the pixels, and one neutral-at-zero set."""
    color = next(spec for spec in CHAIN if spec.name == "color")
    assert color.options == ("saturation", "vibrance", "temperature", "tint")


def test_colour_runs_before_detail():
    """Sharpening sees the colours the image ships with, and grain sits on top."""
    stages = [spec.name for spec in filter_specs(FINAL)]
    assert stages.index("color") < stages.index("rcas") < stages.index("film_grain")


def test_only_the_notch_filter_runs_at_the_decode_stage():
    """The notch filter is the one that must see the raw decode, before upscaling."""
    assert [spec.name for spec in filter_specs(DECODE)] == ["nyquist_notch"]
    assert [spec.name for spec in filter_specs(FINAL)] == ["color", "rcas", "film_grain"]


def test_every_declared_filter_has_an_implementation():
    assert {spec.name for spec in CHAIN} == set(FILTERS)


def test_every_knob_defaults_to_off():
    """The chain skips a filter whose knobs all hold their default; a non-off default
    would mean pixels changed that nobody asked for."""
    options = PostprocessOptions()
    assert all(getattr(options, name) == 0 or getattr(options, name) is False
               for name in POSTPROCESS_NAMES)


def test_a_knob_outside_its_declared_bounds_is_rejected():
    with pytest.raises(ValueError, match="sharpening must be in"):
        PostprocessOptions(sharpening=1.5)


# ------------------------------------------------------------------------ the CLI


@pytest.mark.parametrize("name", POSTPROCESS_NAMES)
def test_every_knob_has_a_cli_flag(name):
    boolean = _SCHEMA[name]["type"] == "boolean"
    argv = [_flag(name)] if boolean else [_flag(name), "0.4"]
    value = getattr(build_parser().parse_args([*PATHS, *argv]), name)
    if boolean:
        assert value is True
    else:
        assert value == pytest.approx(0.4)


@pytest.mark.parametrize("name", POSTPROCESS_NAMES)
def test_an_unpassed_cli_flag_carries_the_declared_default(name):
    args = build_parser().parse_args(PATHS)
    assert getattr(args, name) == getattr(PostprocessOptions, name)


def test_cli_bounds_are_refused_where_they_are_parsed():
    with pytest.raises(SystemExit):
        build_parser().parse_args([*PATHS, "--sharpening", "2"])


def test_a_switch_knob_also_offers_its_negation():
    args = build_parser().parse_args([*PATHS, "--no-qwen-vae-enhance"])
    assert args.qwen_vae_enhance is False


def test_edit_takes_the_same_knobs_as_generate():
    edit = [
        "edit", "--dit", "d", "--vae", "v", "--text-encoder", "t",
        "--prompt", "x", "--image", "in.png",
    ]
    args = build_parser().parse_args([*edit, "--sharpening", "0.3"])
    assert args.sharpening == pytest.approx(0.3)


# ------------------------------------------------------------------ the two requests


@pytest.mark.parametrize("name", POSTPROCESS_NAMES)
def test_every_knob_is_on_the_wire_request(name):
    """The wire stays flat, so the knob must be a top-level field of its own."""
    field = Text2ImageRequest.model_fields[name]
    assert field.default == getattr(PostprocessOptions, name)


@pytest.mark.parametrize("name", POSTPROCESS_NAMES)
def test_every_knob_is_on_the_generate_request(name):
    assert getattr(GenerateRequest(prompt="p"), name) == getattr(PostprocessOptions, name)


def test_the_wire_request_carries_the_bounds_and_the_words_of_every_knob():
    """Pydantic keeps ``ge``/``le`` as constraint objects; compare them to the declaration."""
    for name in POSTPROCESS_NAMES:
        field = Text2ImageRequest.model_fields[name]
        declared = option_meta(PostprocessOptions, name)
        assert field.description == declared.get("help", "")

        bounds = {"ge": None, "le": None}
        for constraint in field.metadata:
            for key in bounds:
                if hasattr(constraint, key):
                    bounds[key] = getattr(constraint, key)
        assert bounds["ge"] == declared.get("ge")
        assert bounds["le"] == declared.get("le")


def test_the_wire_request_reaches_the_chain_unchanged():
    wire = Text2ImageRequest(prompt="a fox", film_grain=0.5, sharpening=0.2)
    assert wire.to_request().postprocess_options() == PostprocessOptions(
        film_grain=0.5, sharpening=0.2
    )


def test_the_generate_request_reaches_the_chain_unchanged():
    request = GenerateRequest(prompt="a fox", qwen_vae_enhance=True, film_grain=1.5)
    assert request.postprocess_options() == PostprocessOptions(
        qwen_vae_enhance=True, film_grain=1.5
    )


def test_flat_options_names_what_a_holder_is_missing():
    class Partial:
        film_grain = 0.5

    with pytest.raises(ValueError, match="carries no post-process field"):
        flat_options(Partial())


# ------------------------------------------------------------------ the UI schema


def test_the_ui_schema_describes_every_knob():
    schema = {entry["name"]: entry for entry in options_schema(PostprocessOptions)}
    assert set(schema) == set(POSTPROCESS_NAMES)

    grain = schema["film_grain"]
    assert grain["label"] == "Film grain"
    assert grain["type"] == "number"
    assert grain["default"] == 0.0
    # What the server accepts and what a slider is worth offering are not the same.
    assert (grain["min"], grain["max"]) == (0.0, 10.0)
    assert grain["ui_max"] == 1.0


def test_the_ui_schema_carries_nothing_a_control_cannot_use():
    for entry in options_schema(PostprocessOptions):
        assert set(entry) <= {
            "name", "label", "type", "default", "min", "max", "step",
            "choices", "pool", "hint", "ui_min", "ui_max",
        }
