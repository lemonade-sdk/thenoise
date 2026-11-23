import pytest
import torch
from PIL import Image, ImageDraw
from conftest import header_of
from test_dit_hunyuan3d import TINY
from thenoise.dit.hunyuan3d import Hunyuan3DShapeParams

from thenoise.models.base import Conditioning
from thenoise.models.config import EncodePromptArgs, MeshDecodeParams, MeshSamplingParams
from thenoise.models.hunyuan3d import Hunyuan3DShapeModel


class RecordingDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, latents, t, context):
        self.seen.append((t.clone(), context.shape))
        return torch.full_like(latents, 2.0 + context.mean())


class StubConditioner:
    def __init__(self):
        self.calls = []

    def __call__(self, images, dtype, views=False):
        self.calls.append((len(images), views))
        tokens = 5 * len(images) if views else 5
        return torch.zeros(1, tokens, 8, dtype=dtype)

    def unconditional(self, like):
        return torch.full_like(like, 7.0)


class StubCodec(torch.nn.Module):
    """Answers the way the shape codec does: an occupancy logit, positive inside."""

    def __init__(self, scale_factor=2.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.token_calls = 0
        self.sdf_calls = []

    def tokens(self, latents):
        self.token_calls += 1
        self.last = latents
        return torch.zeros(latents.shape[0], latents.shape[1], 4)

    def sdf(self, tokens, points):
        self.sdf_calls.append(tuple(points.shape))
        radius = points[0].norm(dim=-1)
        return (0.5 - radius)[None]


class LowPrecisionCodec(StubCodec):
    """The codec as served in bf16, which is how the checkpoint arrives."""

    def sdf(self, tokens, points):
        return (0.5 - points[0].norm(dim=-1))[None].to(torch.bfloat16)


class OneSignCodec(StubCodec):
    """A field with no surface in it, which is what a bad latent looks like from below."""

    def sdf(self, tokens, points):
        return torch.full((1, points.shape[1]), -3.0)


def bare(dtype=torch.float32) -> Hunyuan3DShapeModel:
    """An adapter with stubbed components: the kernels, without a checkpoint."""
    model = object.__new__(Hunyuan3DShapeModel)
    model.device = "cpu"
    model.dtype = dtype
    model.checkpoint_prefs = {}
    model.shape_params = Hunyuan3DShapeParams(**TINY)
    model.dit = RecordingDiT()
    model.conditioner = StubConditioner()
    model.codec = StubCodec()
    return model


def params(steps=4, num_latents=6, seed=3) -> MeshSamplingParams:
    return MeshSamplingParams(height=0, width=0, steps=steps, seed=seed,
                              guidance_scale=5.0, sampler="euler",
                              num_latents=num_latents)


def test_the_shared_shape_file_is_the_only_component():
    assert Hunyuan3DShapeModel.REQUIRED_COMPONENTS == ()
    assert Hunyuan3DShapeModel.MEDIUM == "mesh"
    assert Hunyuan3DShapeModel.REF_METHODS == ("vision",)


def test_the_dit_is_asked_at_the_inverted_clock_and_answers_inverted():
    """It counts up from noise and predicts ``x0 - noise``; the samplers take ``noise - x0``
    at a descending sigma, so neither the clock nor the flow can be handed over as they come."""
    model = bare()
    latents = torch.zeros(1, 3, TINY["in_channels"])
    with torch.no_grad():
        out = model._velocity(latents, torch.tensor(0.25), torch.zeros(1, 5, TINY["context_dim"]))
    t, _ = model.dit.seen[0]
    assert torch.equal(t, torch.tensor([0.75]))
    assert t.dtype is torch.float32
    assert torch.equal(out, torch.full_like(latents, -2.0))


def test_the_schedule_descends_and_sums_to_one():
    steps = bare().schedule(params(steps=5))
    assert len(steps) == 5
    assert float(steps[0].t) == 1.0
    assert all(float(step.delta) > 0 for step in steps)
    assert abs(sum(float(step.delta) for step in steps) - 1.0) < 1e-6


def test_a_requested_sigma_grid_replaces_the_uniform_one():
    model = bare()
    model.checkpoint_prefs = {"sigmas": [1.0, 0.5, 0.0]}
    steps = model.schedule(params(steps=9))
    assert len(steps) == 2
    assert [float(s.t) for s in steps] == [1.0, 0.5]


def test_latents_are_unstructured_tokens_and_seed_fixed():
    first = bare().init_latents(params(steps=1, num_latents=7, seed=11))
    again = bare().init_latents(params(steps=1, num_latents=7, seed=11))
    assert first.shape == (1, 7, TINY["in_channels"])
    assert torch.equal(first, again)


def test_cfg_mixes_the_two_branches():
    model = bare()
    cond = Conditioning(cond=torch.zeros(1, 5, TINY["context_dim"]),
                        null=torch.ones(1, 5, TINY["context_dim"]))
    latents = torch.zeros(1, 3, TINY["in_channels"])
    with torch.no_grad():
        guided = model.denoise_step(latents, torch.tensor(0.5), cond, 2.0, 0)
    assert torch.equal(guided, torch.full_like(latents, -1.0))
    assert len(model.dit.seen) == 2


def test_no_second_branch_without_guidance():
    model = bare()
    cond = Conditioning(cond=torch.zeros(1, 5, TINY["context_dim"]))
    with torch.no_grad():
        model.denoise_step(torch.zeros(1, 3, TINY["in_channels"]), torch.tensor(0.5),
                           cond, 1.0, 0)
    assert len(model.dit.seen) == 1


def test_the_picture_is_the_conditioning_and_the_null_branch_is_zeros_like_it():
    model = bare()
    out = model.encode_prompt(EncodePromptArgs(prompt="a chair", guidance_scale=5.0,
                                                  image=[Image.new("RGB", (32, 32))]))
    assert out.cond.shape == (1, 5, 8)
    assert torch.equal(out.null, torch.full(out.cond.shape, 7.0))
    assert model.conditioner.calls == [(1, False)]


def test_views_are_not_this_checkpoint_s_to_offer():
    """Its conditioner is the release's `DinoImageEncoder`, which concatenates nothing, so
    the offsets a multi-view tower adds are numbers this one has never met. The refusal of
    a second picture is the medium's, at the request boundary."""
    assert bare().capability("mesh").get("mv2d") is None


def test_a_prompt_without_a_picture_is_refused():
    with pytest.raises(ValueError, match="image is required"):
        bare().encode_prompt(EncodePromptArgs(prompt="a chair", guidance_scale=5.0))


def test_the_scale_factor_comes_off_before_the_codec_is_queried():
    model = bare()
    latents = torch.full((1, 4, TINY["in_channels"]), 4.0)
    model.decode_mesh(latents, MeshDecodeParams(octree_resolution=8, bounds=1.0,
                                                mc_level=0.0, chunk=64))
    assert model.codec.token_calls == 1
    assert torch.allclose(model.codec.last, torch.full((1, 4, TINY["in_channels"]), 2.0))


def test_the_tokens_are_refined_once_for_the_whole_volume():
    model = bare()
    model.decode_mesh(torch.zeros(1, 4, TINY["in_channels"]),
                      MeshDecodeParams(octree_resolution=16, bounds=1.0, mc_level=0.0,
                                       chunk=32))
    assert model.codec.token_calls == 1
    assert len(model.codec.sdf_calls) > 1


def test_a_mesh_comes_out_of_the_decode():
    model = bare()
    mesh = model.decode_mesh(torch.zeros(1, 4, TINY["in_channels"]),
                             MeshDecodeParams(octree_resolution=32, bounds=1.0, mc_level=0.0,
                                              chunk=512))
    assert not mesh.is_empty
    assert mesh.signed_volume() > 0


def test_a_bf16_codec_still_yields_a_mesh():
    model = bare()
    model.codec = LowPrecisionCodec()
    mesh = model.decode_mesh(torch.zeros(1, 4, TINY["in_channels"]),
                             MeshDecodeParams(octree_resolution=32, bounds=1.0, mc_level=0.0,
                                              chunk=512))
    assert not mesh.is_empty
    assert mesh.vertices.dtype is torch.float32


def test_a_shape_that_never_surfaces_blames_the_field_it_saw():
    model = bare()
    model.codec = OneSignCodec()
    with pytest.raises(RuntimeError) as caught:
        model.decode_mesh(torch.zeros(1, 4, TINY["in_channels"]),
                          MeshDecodeParams(octree_resolution=16, bounds=1.0, mc_level=0.0,
                                           chunk=64))
    assert "no corner changed sign" in str(caught.value)
    assert "3.0000..3.0000" in str(caught.value)
    assert "latents mean 0.000 std 0.000" in str(caught.value)


def test_the_volume_defaults_are_the_shape_pipelines():
    assert Hunyuan3DShapeModel.DEFAULT_PREFS["octree_resolution"] == 384
    assert Hunyuan3DShapeModel.DEFAULT_PREFS["bounds"] == 1.01
    assert Hunyuan3DShapeModel.DEFAULT_PREFS["mc_level"] == 1 / 512


def test_a_reference_arrives_as_the_square_the_tower_was_trained_on():
    cutout = Image.new("RGBA", (900, 300), (0, 0, 0, 0))
    ImageDraw.Draw(cutout).rectangle((300, 60, 600, 240), fill=(10, 20, 30, 255))
    images = bare().conditioned_images(
        EncodePromptArgs(prompt="", image=[cutout], guidance_scale=1.0))
    assert images[0].size == (518, 518)
    assert images[0].getpixel((0, 0)) == (255, 255, 255)
    assert sum(images[0].getpixel((259, 259))) < 120


def test_a_reference_is_covered_and_centre_cropped_rather_than_squashed():
    """The tower's own fit sends the short side to 518 and crops the overhang off the long
    one, so an off-centre band that leaves the frame does so symmetrically."""
    image = Image.new("RGB", (1000, 500), "white")
    ImageDraw.Draw(image).rectangle((0, 0, 199, 499), fill="black")
    framed = bare().prepare_reference(image)
    assert framed.size == (518, 518)
    assert framed.convert("L").getextrema() == (255, 255)


def test_a_small_reference_is_enlarged_to_the_towers_canvas():
    assert bare().prepare_reference(Image.new("RGB", (120, 90))).size == (518, 518)


def test_detection_uses_the_dits_own_fingerprint():
    assert Hunyuan3DShapeModel.detect(header_of([
        "model.t_embedder.mlp.2.weight", "model.blocks.0.attn1.k_norm.weight"]))
    assert not Hunyuan3DShapeModel.detect(header_of([
        "model.t_embedder.mlp.0.weight", "model.blocks.0.attn1.k_norm.weight"]))
    assert Hunyuan3DShapeModel.detect(header_of([
        "model.diffusion_model.model.t_embedder.mlp.2.weight",
        "model.diffusion_model.model.blocks.0.attn1.k_norm.weight"]))
    assert not Hunyuan3DShapeModel.detect(header_of([
        "model.diffusion_model.txt_in.weight", "model.diffusion_model.img_in.weight"]))
