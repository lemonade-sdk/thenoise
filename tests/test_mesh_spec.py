import math

import pytest

from thenoise.mesh.spec import (
    DEFAULT_MESH_SCALE, DENSITY_PRESETS, MAX_OCTREE_RESOLUTION, OUTPUT_FORMATS,
    MeshSpec, mode_of,
)
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
    assert set(public) <= {"mode", "latents", "steps", "guidance_scale", "sampler",
                           "seed", "attention", "output_format", "octree_resolution",
                           "bounds", "mc_level", "scale", "lora_specs"}


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


def test_the_density_presets_double_and_stop_where_the_grid_stops_fitting():
    steps = list(DENSITY_PRESETS)
    assert all(fine == 2 * coarse for coarse, fine in zip(steps, steps[1:])), steps
    assert min(steps) >= 8, steps
    assert max(steps) == MAX_OCTREE_RESOLUTION, "the pane would offer what the engine refuses"


def test_a_density_past_the_sampled_grid_is_refused_before_a_weight_moves():
    with pytest.raises(ValueError, match="at most"):
        resolve(octree_resolution=MAX_OCTREE_RESOLUTION + 1)


def test_an_unscaled_shape_spans_the_volume_it_was_sampled_from():
    assert resolve().scale == DEFAULT_MESH_SCALE


def test_units_are_nobody_s_business_until_someone_asks():
    assert resolve(scale=100.0).scale == 100.0
    assert MeshSpec().resolve({**DEFAULTS, "scale": 0.001}).scale == 0.001


def test_a_scale_that_is_not_a_positive_number_refuses_the_request():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            MeshSpec(scale=bad)
