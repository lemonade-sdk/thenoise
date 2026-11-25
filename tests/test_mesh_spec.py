import math

import pytest

from thenoise.mesh.finish.options import DEFAULTS as FINISH_DEFAULTS
from thenoise.mesh.spec import (
    DEFAULT_MESH_SCALE, OUTPUT_FORMATS, MeshSpec, knob_menu, limits, mode_of,
)
from thenoise.models.hunyuan3d21 import Hunyuan3D21ShapeModel
from thenoise.utils.attention import DEFAULT_ATTENTION

DEFAULTS = {
    "attention": DEFAULT_ATTENTION,
    "output_format": "glb",
    "num_latents": 4096,
    "steps": 30,
    "guidance_scale": 5.0,
    "sampler": "euler",
    "octree_resolution": 384,
    "bounds": 1.01,
    "mc_level": 0.0,
}


def resolve(**overrides):
    return MeshSpec(**overrides).resolve(DEFAULTS)


def test_a_picture_makes_it_an_image_to_shape_request():
    assert mode_of(()) == "t2d"
    assert mode_of((b"png",)) == "i2d"
    assert resolve(image=b"png").mode == "i2d"
    assert resolve(image=b"png").images == (b"png",)


def test_several_pictures_are_views_of_the_one_shape_asked_for():
    assert mode_of((b"front", b"left")) == "mv2d"
    resolved = resolve(image=[b"front", b"left", b"back"])
    assert (resolved.mode, resolved.images) == ("mv2d", (b"front", b"left", b"back"))


def test_the_models_defaults_fill_what_the_request_left_out():
    resolved = resolve()
    assert (resolved.latents, resolved.steps, resolved.sampler) == (4096, 30, "euler")
    assert (resolved.octree_resolution, resolved.bounds, resolved.mc_level) == (384, 1.01, 0.0)
    assert resolved.output_format == "glb"


def test_the_request_values_win_over_the_defaults():
    resolved = resolve(latents=1024, steps=4, guidance_scale=1.0, sampler="dpm++_3m_sde",
                       octree_resolution=96, bounds=1.5, mc_level=0.02, seed=7)
    assert resolved.latents == 1024
    assert resolved.steps == 4
    assert resolved.octree_resolution == 96
    assert resolved.bounds == 1.5
    assert resolved.mc_level == 0.02
    assert resolved.seed == 7


def test_the_latent_field_is_the_token_count_not_the_image_size():
    assert resolve(latents=8192).latents == 8192
    with pytest.raises(Exception, match="greater than"):
        MeshSpec(latents=0)


def test_a_seed_of_minus_one_asks_for_a_fresh_one():
    seeds = {resolve(seed=-1).seed for _ in range(12)}
    assert len(seeds) > 1
    assert all(0 <= seed < 2**32 for seed in seeds)


def test_an_explicit_seed_survives():
    assert resolve(seed=0).seed == 0


def test_the_container_is_one_of_the_two_writers():
    assert resolve(output_format="obj").output_format == "obj"
    with pytest.raises(Exception, match="Input should be"):
        MeshSpec(output_format="ply")
    with pytest.raises(ValueError, match="output_format must be one of"):
        MeshSpec().resolve({**DEFAULTS, "output_format": "ply"})


def test_the_volume_must_be_fine_enough_to_have_a_surface_in():
    with pytest.raises(ValueError, match="octree_resolution"):
        MeshSpec().resolve({**DEFAULTS, "octree_resolution": 4})


def test_the_numbers_have_to_be_numbers_the_solvers_can_use():
    for field, value in [("guidance_scale", -1.0), ("guidance_scale", math.nan),
                         ("bounds", 0.0), ("mc_level", math.inf)]:
        with pytest.raises(ValueError, match=field):
            MeshSpec(**{field: value}).resolve(DEFAULTS)


def test_the_seed_is_inside_torchs_range():
    with pytest.raises(Exception, match="torch.Generator"):
        MeshSpec(seed=2**64)


def test_the_echo_leaves_the_picture_out():
    public = resolve(image=b"png").public()
    assert "image" not in public
    assert public["mode"] == "i2d"
    assert set(public) <= {"mode", "from_shape", "latents", "steps", "guidance_scale",
                           "sampler", "seed", "attention", "output_format",
                           "octree_resolution", "bounds", "mc_level", "scale", "lora_specs",
                           "finish"}
    # The knobs travel back as the run resolved them, which is how a pane tells a knob that
    # arrived from one that its browser dropped on the floor.
    assert public["finish"] == FINISH_DEFAULTS


def test_a_request_names_no_loras_unless_asked():
    assert resolve().lora_specs == {}


def test_stl_is_one_of_the_answered_containers():
    assert resolve(output_format="stl").output_format == "stl"
    assert "stl" in OUTPUT_FORMATS


def test_every_promised_container_resolves_and_is_written():
    from thenoise.mesh.write import write_mesh
    from test_mesh_write import triangle

    for fmt in OUTPUT_FORMATS:
        assert resolve(output_format=fmt).output_format == fmt
    with pytest.raises(ValueError, match="no mesh writer for"):
        write_mesh(triangle(), "/tmp/nope.ply", "ply")


def named(menu, knob):
    return [(one["name"], one["value"]) for one in menu[knob]]


def test_a_shape_model_menus_words_over_its_own_numbers():
    """The pane's menus are the shipped model's trained values scaled, each keeping the word
    that step means: a user picks a word, the request carries the number."""
    menu = knob_menu(Hunyuan3D21ShapeModel.DEFAULT_PREFS)
    assert named(menu, "octree_resolution") == [("draft", 96), ("fine", 192),
                                                ("standard", 384), ("ultra", 768)]
    assert named(menu, "latents") == [("lean", 2048), ("standard", 4096),
                                      ("detailed", 8192)]
    assert named(menu, "bounds") == [("tight", 0.909), ("standard", 1.01),
                                     ("roomy", 1.212), ("wide", 1.515)]
    assert named(menu, "mc_level") == [("sharp", 1 / 1024), ("standard", 1 / 512),
                                       ("fatten", 1 / 256), ("extra fatten", 1 / 128)]
    assert named(menu, "scale") == [("\u00d70.5", 0.5), ("\u00d71", 1),
                                   ("\u00d72", 2), ("\u00d74", 4)]


def test_another_model_of_the_medium_answers_with_its_own_numbers():
    menu = knob_menu({"num_latents": 3072, "octree_resolution": 256, "bounds": 1.0,
                      "mc_level": 1 / 256, "scale": 400})
    assert named(menu, "latents") == [("lean", 1536), ("standard", 3072),
                                      ("detailed", 6144)]
    assert named(menu, "octree_resolution") == [("draft", 64), ("fine", 128),
                                                ("standard", 256), ("ultra", 512)]
    assert named(menu, "bounds") == [("tight", 0.9), ("standard", 1.0),
                                     ("roomy", 1.2), ("wide", 1.5)]
    assert named(menu, "mc_level") == [("sharp", 1 / 512), ("standard", 1 / 256),
                                       ("fatten", 1 / 128), ("extra fatten", 1 / 64)]
    assert named(menu, "scale") == [("\u00d70.5", 200), ("\u00d71", 400),
                                   ("\u00d72", 800)]


def test_a_menu_stops_where_a_request_would_be_refused():
    """A model trained near an edge must not be offered past it, and a step that falls away
    takes its word with it rather than renaming what is left."""
    menu = knob_menu({"num_latents": 12000, "octree_resolution": 700, "bounds": 7.0,
                      "mc_level": 0.4})
    assert named(menu, "latents") == [("lean", 6000), ("standard", 12000)]
    assert named(menu, "octree_resolution") == [("draft", 175), ("fine", 350),
                                                ("standard", 700)]
    assert named(menu, "bounds") == [("tight", 6.3), ("standard", 7.0)]
    assert named(menu, "mc_level") == [("sharp", 0.2), ("standard", 0.4)]


def test_a_knob_the_model_has_no_value_for_offers_no_menu():
    assert named(knob_menu({"num_latents": 4096}), "latents") == [
        ("lean", 2048), ("standard", 4096), ("detailed", 8192)]
    assert knob_menu({}) == {}


def test_every_offered_value_is_one_a_request_can_ask_for():
    for knob, steps in knob_menu(Hunyuan3D21ShapeModel.DEFAULT_PREFS).items():
        for step in steps:
            assert getattr(resolve(**{knob: step["value"]}), knob) == step["value"], step


def test_the_numbers_no_shape_can_use_are_refused_by_name():
    """Wide edges on purpose: refused where the request cannot work, not where it would
    look worse. These two are the ones a real request reached the octree with."""
    for knob, bad in [("bounds", 700.0), ("bounds", 0.001), ("mc_level", 3.0),
                      ("mc_level", -40.0), ("latents", 20_000),
                      ("octree_resolution", 900), ("octree_resolution", 4),
                      ("scale", 1e6), ("scale", 1e-6)]:
        with pytest.raises(ValueError, match=knob):
            resolve(**{knob: bad})


def test_a_refusal_names_the_edges_it_was_checked_against():
    for knob in ("latents", "bounds", "mc_level", "octree_resolution", "scale"):
        low, high, why = limits(knob)
        with pytest.raises(ValueError, match=f"{knob} must be between {low} and {high}"):
            resolve(**{knob: high * 1000 if low >= 0 else -high * 1000})


def test_an_unscaled_shape_spans_the_volume_it_was_sampled_from():
    assert resolve().scale == DEFAULT_MESH_SCALE


def test_units_are_nobody_s_business_until_someone_asks():
    assert resolve(scale=100.0).scale == 100.0
    assert MeshSpec().resolve({**DEFAULTS, "scale": 0.001}).scale == 0.001


def test_a_scale_that_is_not_a_positive_number_refuses_the_request():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            MeshSpec(scale=bad)
