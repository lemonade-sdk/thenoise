"""Shared reference-latent editing infrastructure tests (no weights / no GPU).

Covers the generic reference helpers (``thenoise.dit.reference``), the
``PipelineCache`` reference stage, and the pipeline's edit caching — re-editing
the same image with a different prompt must NOT re-encode it.
"""
from __future__ import annotations

import torch
from PIL import Image

from thenoise.dit.reference import (
    concat_reference,
    slice_reference_output,
)
from thenoise.models.base import Conditioning, DiffusionModel
from thenoise.models.config import GenerateRequest, ModelConfig, SamplingParams
from thenoise.pipeline import PipelineController
from thenoise.samplers import Step
from thenoise.upscale.pixel import PixelUpscalerManager
from thenoise.utils.pipeline_cache import PipelineCache


# ------------------------------------------------------------- reference helpers


def test_concat_reference_none_is_noop():
    img = torch.zeros(1, 4, 8)
    img_ids = torch.zeros(1, 4, 4, dtype=torch.long)
    out_img, out_ids = concat_reference(img, img_ids, None, None)
    assert out_img is img and out_ids is img_ids


def test_concat_and_slice_reference():
    img = torch.ones(1, 4, 8)
    img_ids = torch.zeros(1, 4, 4, dtype=torch.long)
    ref_tokens = torch.zeros(1, 2, 8)
    ref_ids = torch.ones(1, 2, 4, dtype=torch.long)

    cat_img, cat_ids = concat_reference(img, img_ids, ref_tokens, ref_ids)
    assert cat_img.shape == (1, 6, 8)
    assert cat_ids.shape == (1, 6, 4)
    # Reference ids are the trailing tokens.
    assert torch.all(cat_ids[0, 4:] == 1)
    assert torch.all(cat_ids[0, :4] == 0)

    # Slicing back drops the reference tokens -> original image tokens.
    out = slice_reference_output(cat_img, 4)
    assert out.shape == (1, 4, 8)
    assert torch.all(out == 1)


# ------------------------------------------------------------- cache cascade


def test_pipeline_cache_reference_cascade():
    c = PipelineCache()
    c.reference_store(("ref", 1), "R1")
    c.prompt_store(("p", 1), "C1")
    c.sampling_store(("s", 1), "S1")
    c.decode_store(("d", 1), "D1")

    # A reference miss clears all downstream stages.
    c.reference_store(("ref", 2), "R2")
    assert c.reference_hit(("ref", 2))
    assert c.prompt_key is None
    assert c.sampling_key is None
    assert c.decode_key is None


# ------------------------------------------------------------- edit caching


class _StubEditModel(DiffusionModel):
    """Minimal editing model: counts ``encode_reference`` calls."""

    name = "stub_edit"
    supports_edit = True
    DEFAULT_WIDTH = 64
    DEFAULT_HEIGHT = 64
    DEFAULT_STEPS = 4
    DEFAULT_GUIDANCE_SCALE = 1.0
    SAMPLER = "euler"

    def __init__(self):
        config = ModelConfig(
            dit_path="x", vae_path="x", text_encoder_path="x",
            device="cpu", dtype=torch.float32,
        )
        super().__init__(config=config)
        self.encode_calls = 0
        self.dit = torch.nn.Identity()

    @staticmethod
    def detect(f):
        return False

    def encode_reference(self, pixels):
        self.encode_calls += 1
        return pixels.unsqueeze(0).float()  # [1, C, H, W]

    def encode_prompt(self, args):
        return Conditioning(cond=torch.zeros(1, 8, 8), null=None)

    def init_latents(self, params):
        return torch.randn(1, 4, params.height // 8, params.width // 8)

    def schedule(self, params):
        return [
            Step(t=torch.tensor([1.0 - i / params.steps]),
                 delta=torch.tensor([1.0 / params.steps]))
            for i in range(params.steps)
        ]

    def denoise_step(self, latents, t, cond, guidance_scale, i):
        return torch.zeros_like(latents)

    def decode(self, latents):
        return torch.zeros(3, 64, 64)

    def _upscale_format(self):
        return "flux2"


def _edit_controller():
    return PipelineController(_StubEditModel(), PixelUpscalerManager(upscaler_dir="", device="cpu"))


class _SizeSpyModel(_StubEditModel):
    """Records the resolved ``SamplingParams`` passed to ``init_latents``."""

    def __init__(self):
        super().__init__()
        self.sizes = []

    def init_latents(self, params):
        self.sizes.append((params.width, params.height))
        return super().init_latents(params)


def test_edit_reuses_reference_for_same_image():
    controller = _edit_controller()
    img_a = Image.new("RGB", (64, 64), "red")

    controller.edit(GenerateRequest(prompt="p1", image=img_a, seed=1))
    assert controller.model.encode_calls == 1

    # Same image, different prompt -> reference reused (no re-encode).
    controller.edit(GenerateRequest(prompt="p2", image=img_a, seed=2))
    assert controller.model.encode_calls == 1


def test_edit_reencodes_on_different_image():
    controller = _edit_controller()
    img_a = Image.new("RGB", (64, 64), "red")
    img_b = Image.new("RGB", (64, 64), "blue")

    controller.edit(GenerateRequest(prompt="p1", image=img_a, seed=1))
    assert controller.model.encode_calls == 1
    controller.edit(GenerateRequest(prompt="p1", image=img_b, seed=2))
    assert controller.model.encode_calls == 2


def test_edit_rejects_unsupported_model():
    from thenoise.models.base import DiffusionModel
    from thenoise.models.anima import AnimaModel

    # Anima is a real (non-editing) model; construct a bare adapter check via
    # the class flag rather than instantiating (no weights needed).
    assert not AnimaModel.supports_edit
    assert DiffusionModel.supports_edit is False


def test_edit_derives_size_from_image_resized_to_1024():
    """No width/height -> first image resized to 1024 on its largest side.

    The caller's request is left unmutated; the derived size is applied to a
    local copy only.
    """
    controller = PipelineController(
        _SizeSpyModel(), PixelUpscalerManager(upscaler_dir="", device="cpu")
    )
    # 2:1 landscape image -> largest side 1024, height 512.
    img = Image.new("RGB", (100, 50), "red")
    request = GenerateRequest(prompt="p", image=img, seed=1)
    controller.edit(request)
    assert request.width is None and request.height is None
    assert controller.model.sizes == [(1024, 512)]


def test_edit_derives_size_from_image_portrait():
    """Portrait image: largest side (height) resized to 1024."""
    controller = PipelineController(
        _SizeSpyModel(), PixelUpscalerManager(upscaler_dir="", device="cpu")
    )
    img = Image.new("RGB", (50, 100), "blue")  # 1:2 portrait
    request = GenerateRequest(prompt="p", image=img, seed=1)
    controller.edit(request)
    assert request.width is None and request.height is None
    assert controller.model.sizes == [(512, 1024)]


def test_edit_explicit_width_height_are_used():
    """Explicit width/height override the image-derived size."""
    controller = PipelineController(
        _SizeSpyModel(), PixelUpscalerManager(upscaler_dir="", device="cpu")
    )
    img = Image.new("RGB", (100, 50), "red")
    request = GenerateRequest(prompt="p", image=img, seed=1,
                              width=128, height=64)
    controller.edit(request)
    assert controller.model.sizes == [(128, 64)]
    assert request.width == 128 and request.height == 64


def test_encode_prompt_receives_args_struct():
    """The pipeline passes an EncodePromptArgs struct to encode_prompt."""
    from thenoise.models.config import EncodePromptArgs

    received = {}

    class SpyModel(_StubEditModel):
        def encode_prompt(self, args):
            received["args"] = args
            return Conditioning(cond=torch.zeros(1, 8, 8), null=None)

    controller = PipelineController(SpyModel(), PixelUpscalerManager(upscaler_dir="", device="cpu"))
    img = Image.new("RGB", (64, 64), "red")
    controller.edit(GenerateRequest(prompt="p", negative_prompt="n",
                                    image=img, seed=1, guidance_scale=4.0))
    args = received["args"]
    assert isinstance(args, EncodePromptArgs)
    assert args.prompt == "p"
    assert args.negative_prompt == "n"
    assert args.guidance_scale == 4.0
    assert args.image is img


# ------------------------------------------------------------- multi-image


def test_multi_edit_derives_size_from_first_image():
    """Output resolution/aspect derive from the FIRST reference image (to 1024)."""
    controller = PipelineController(
        _SizeSpyModel(), PixelUpscalerManager(upscaler_dir="", device="cpu")
    )
    first = Image.new("RGB", (100, 50), "red")   # 2:1 landscape
    second = Image.new("RGB", (50, 100), "blue")  # portrait (ignored for size)
    request = GenerateRequest(prompt="p", image=[first, second], seed=1)
    controller.edit(request)
    # Largest side from the first image's aspect ratio (2:1 landscape) -> 1024.
    assert controller.model.sizes == [(1024, 512)]
    assert request.width is None and request.height is None


def test_multi_edit_encodes_each_reference_once():
    """Each distinct ref image is VAE-encoded once (cached by content hash)."""
    controller = _edit_controller()
    img_a = Image.new("RGB", (64, 64), "red")
    img_b = Image.new("RGB", (64, 64), "blue")

    controller.edit(GenerateRequest(prompt="p1", image=[img_a, img_b], seed=1))
    assert controller.model.encode_calls == 2

    # Same two images, different prompt -> refs reused (no re-encode).
    controller.edit(GenerateRequest(prompt="p2", image=[img_a, img_b], seed=2))
    assert controller.model.encode_calls == 2

    # Reordering the refs changes the reference -> re-encode.
    controller.edit(GenerateRequest(prompt="p3", image=[img_b, img_a], seed=3))
    assert controller.model.encode_calls == 4


def test_multi_edit_single_image_equals_one_element_list():
    """A bare single image and a one-element list are equivalent (both cache-hit)."""
    controller = _edit_controller()
    img = Image.new("RGB", (64, 64), "red")

    controller.edit(GenerateRequest(prompt="p1", image=img, seed=1))
    assert controller.model.encode_calls == 1

    # Same image as a one-element list -> cache hit (no re-encode).
    controller.edit(GenerateRequest(prompt="p2", image=[img], seed=2))
    assert controller.model.encode_calls == 1
