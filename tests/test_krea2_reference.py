"""Krea 2 reference-image ("edit") conditioning tests (no real weights).

Covers the adapter kernels (``pack_reference_latent`` / grounded ``encode_prompt``)
and the grounded-encode template/prep, using bare instances and mock encoders.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from thenoise.dit.krea2.encoder import Qwen3VLConditioner
from thenoise.dit.krea2.sampling import encode_prompts, gather_valid_text
from thenoise.models.config import EncodePromptArgs
from thenoise.models.krea2 import Krea2Model


def _bare_krea2(**attrs):
    model = object.__new__(Krea2Model)
    model.device = "cpu"
    model.dtype = torch.float32
    model.dit = SimpleNamespace(config=SimpleNamespace(patch=2))
    for k, v in attrs.items():
        setattr(model, k, v)
    return model


# --------------------------------------------------------------- grounded template


def test_grounded_template_places_vision_tokens():
    cond = object.__new__(Qwen3VLConditioner)
    cond.grounded_system = "SYS"
    cond.grounded_suffix = "SUF"
    tpl = cond._grounded_template(1)
    assert tpl == "SYS<|vision_start|><|image_pad|><|vision_end|>{}SUF"
    tpl2 = cond._grounded_template(2)
    assert tpl2.count("<|image_pad|>") == 2


def test_prep_image_caps_longest_side():
    from PIL import Image

    cond = object.__new__(Qwen3VLConditioner)
    img = Image.new("RGB", (2048, 1024))
    out = cond._prep_image(img, grounding_px=768)
    assert out.size == (768, 384)
    # Native when grounding_px is 0.
    assert cond._prep_image(img, grounding_px=0) is img
    # No-op when already under the cap.
    small = Image.new("RGB", (64, 64))
    assert cond._prep_image(small, grounding_px=768).size == (64, 64)


# --------------------------------------------------------------- grounded encode_prompts


class _GroundedRecorder:
    """Mock encoder that records the image/grounding kwargs it is handed."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompts, images=None, grounding_px=768):
        self.calls.append({"prompts": list(prompts), "images": images, "grounding_px": grounding_px})
        batch = len(prompts)
        txt = torch.arange(batch * 4 * 1 * 2, dtype=torch.float32).reshape(batch, 4, 1, 2)
        mask = torch.ones(batch, 4, dtype=torch.bool)
        return txt, mask


def test_encode_prompts_grounds_condition_and_negative():
    enc = _GroundedRecorder()
    img = "ref"
    txt, txtmask, untxt, untxtmask = encode_prompts(
        enc, ["a"], ["bad"], cfg=True, images=img, grounding_px=512
    )
    assert enc.calls == [
        {"prompts": ["a"], "images": img, "grounding_px": 512},
        {"prompts": ["bad"], "images": img, "grounding_px": 512},
    ]
    assert untxt is not None


def test_encode_prompts_text_only_does_not_pass_images():
    enc = _GroundedRecorder()
    txt, txtmask, untxt, untxtmask = encode_prompts(enc, ["a"], cfg=False)
    assert enc.calls == [{"prompts": ["a"], "images": None, "grounding_px": 768}]


def test_gather_valid_text_keeps_vision_tokens():
    # A grounded sequence is [system, vision, prompt, suffix], all valid (batch 1).
    txt = torch.arange(6 * 1 * 3, dtype=torch.float32).reshape(1, 6, 1, 3)
    mask = torch.ones(1, 6, dtype=torch.bool)
    out, out_mask = gather_valid_text(txt, mask)
    assert out.shape == (1, 6, 1, 3)
    assert out_mask.tolist() == [[True] * 6]


# --------------------------------------------------------------- adapter kernels


def test_pack_reference_latent_rejects_non_fit():
    model = _bare_krea2()
    with pytest.raises(ValueError):
        model.pack_reference_latent(torch.randn(1, 16, 4, 4), method="crop")


def test_pack_reference_latent_returns_tokens_and_pos():
    model = _bare_krea2()
    tokens, pos = model.pack_reference_latent(torch.randn(1, 16, 4, 4), ref_index=1)
    assert tokens.shape == (1, 4, 16 * 2 * 2)
    assert pos.shape == (1, 4, 3)
    assert (pos[0, :, 0] == 1.0).all()


def test_encode_prompt_routes_grounded_path_with_image():
    """With an image the encoder is called grounded; text-only stays the fast path."""
    class _Enc:
        def __init__(self):
            self.calls = []

        def __call__(self, prompts, images=None, grounding_px=768):
            self.calls.append((images, grounding_px))
            batch = len(prompts)
            txt = torch.arange(batch * 4 * 1 * 2, dtype=torch.float32).reshape(batch, 4, 1, 2)
            mask = torch.ones(batch, 4, dtype=torch.bool)
            return txt, mask

    model = _bare_krea2()
    model.encoder = _Enc()
    model.DEFAULT_GROUNDING_PX = 768

    # Text-only: no image -> fast path.
    model.encode_prompt(EncodePromptArgs(prompt="p", negative_prompt="", guidance_scale=1.0, image=None))
    assert model.encoder.calls == [(None, 768)]

    # Edit: image present -> grounded path (grounded_px default). cfg off (guidance 1.0)
    # means only the conditional is encoded, so a single grounded call. A single image is
    # normalized to a list for the grounded template.
    model.encoder.calls.clear()
    model.encode_prompt(EncodePromptArgs(prompt="p", negative_prompt="", guidance_scale=1.0, image="ref"))
    assert model.encoder.calls == [(["ref"], 768)]


def test_identity_edit_lora_keys_map_to_krea2_dit_schema():
    """Every identity-edit LoRA module maps 1:1 onto a Krea2 DiT parameter.

    Validates the ``switch_loras`` key match (keys only, no full weights) so the
    LoRA loads without an unknown-key error.
    """
    from pathlib import Path

    from accelerate import init_empty_weights
    from safetensors import safe_open

    from thenoise.dit.krea2.mmdit import SingleStreamDiT
    from thenoise.dit.krea2.utils import single_mmdit_large_wide

    lora_path = Path(__file__).parent.parent / "models/loras/krea2_identity_edit_v1_2.safetensors"
    if not lora_path.exists():
        pytest.skip("identity-edit LoRA not present")

    with init_empty_weights():
        dit = SingleStreamDiT(single_mmdit_large_wide)
    model_params = {n for n, _ in dit.named_parameters()}

    with safe_open(str(lora_path), framework="pt") as f:
        lora_keys = list(f.keys())

    assert lora_keys, "expected LoRA keys"
    for key in lora_keys:
        module = key.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
        # ``diffusion_model.`` is a ComfyUI wrapper stripped by the loader.
        model_key = module.removeprefix("diffusion_model.") + ".weight"
        assert model_key in model_params, f"LoRA key {key!r} -> {model_key!r} not in schema"


class _FakePosemb:
    def clear(self):
        pass

    def store(self, name, pos, dtype=None):
        self._v = pos

    def __getitem__(self, name):
        return self._v


class _FakeDit:
    """Records the ``img``/``ref_len`` it is handed and returns a target-sized output."""

    def __init__(self):
        self.config = SimpleNamespace(patch=2)
        self.posemb = _FakePosemb()
        self.calls = []

    def __call__(self, img, context, t, mask, freqs, ref_len=0):
        self.calls.append({"img": img, "ref_len": ref_len})
        # The DiT slices the leading refs; return only the target tokens.
        return img[:, ref_len:]


def test_prepare_latent_returns_target_only_and_denoise_prepends_refs():
    from thenoise.models.config import SamplingParams

    model = _bare_krea2()
    model.dit = _FakeDit()
    model._compression = 8

    # Target latent [1,16,4,4] -> 2x2 target tokens; one ref -> 2x2 ref tokens.
    latents = torch.randn(1, 16, 4, 4)
    ref = [torch.randn(1, 16, 4, 4)]
    cond = type("C", (), {"cond": torch.zeros(1, 3, 12, 2560), "cond_mask": torch.ones(1, 3, dtype=torch.bool),
                          "null": None, "null_mask": None})()
    params = SamplingParams(height=32, width=32, steps=2, seed=0, guidance_scale=1.0, sampler="euler")

    out = model.prepare_latent(latents, cond, params, ref=ref, ref_method="fit")
    # ``prepare_latent`` returns the TARGET tokens only (2x2 = 4 tokens), so the sampler
    # integrates a shape that matches ``denoise_step``'s output.
    assert out.shape == (1, 4, 16 * 2 * 2)
    assert model._ref_tokens.shape == (1, 4, 16 * 2 * 2)

    v = model.denoise_step(out, torch.tensor(1.0), cond, 1.0, 0)
    # ``denoise_step`` handed the DiT the combined [refs|target] and sliced back to target.
    assert model.dit.calls[-1]["img"].shape == (1, 8, 16 * 2 * 2)
    assert model.dit.calls[-1]["ref_len"] == 4
    assert v.shape == (1, 4, 16 * 2 * 2)
