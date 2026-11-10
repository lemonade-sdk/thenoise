"""Qwen-Image 2.1 adapter specifics that need no weights."""
from __future__ import annotations

import torch
from PIL import Image

from comfy_kitchen.tensor import QuantizedTensor, TensorWiseINT8Layout
from conftest import comfy_quant, write_safetensors
from thenoise.dit.quantized import QuantizedLinear
from thenoise.dit.qwen_image21.utils import _fuse_qkv_state
from thenoise.utils.loader import restore_quantized_layer
from thenoise.models import QwenImage21Model
from thenoise.models.base import DiffusionModel
from thenoise.models.config import EncodePromptArgs
from thenoise.upscale import LatentTranscodeNet, Qwen21TranscodeUpscaler
from thenoise.upscale import qwen21_transcode


class FakeVAE:
    """Just the surface the transcoder is allowed to reach for."""

    z_dim = 4

    def decode_features(self, latents, *, blocks=1):
        b, _, h, w = latents.shape
        return torch.zeros(b, 6, 2 * h, 2 * w, dtype=latents.dtype)


def _bare(**attrs):
    """An adapter instance without ``__init__`` (no weights, no device moves)."""
    model = object.__new__(QwenImage21Model)
    model.device = "cpu"
    model.dtype = torch.float32
    for key, value in attrs.items():
        setattr(model, key, value)
    return model


def test_the_vision_tokens_get_an_rgb_composite_not_a_dropped_alpha():
    """A transparent reference must not reach Qwen3-VL as its raw RGB."""
    transparent_red = Image.new("RGBA", (32, 32), (255, 0, 0, 0))

    images = _bare()._encoder_images(
        EncodePromptArgs(prompt="p", image=transparent_red)
    )

    assert [img.mode for img in images] == ["RGB"]
    assert images[0].getpixel((0, 0)) == (255, 255, 255)


def test_every_reference_in_a_list_is_composited():
    images = _bare()._encoder_images(
        EncodePromptArgs(
            prompt="p",
            image=[Image.new("RGBA", (32, 32), (0, 0, 255, 0)), Image.new("RGB", (32, 32), "red")],
        )
    )

    assert images[0].getpixel((0, 0)) == (255, 255, 255)  # transparent -> white
    assert images[1].getpixel((0, 0)) == (255, 0, 0)  # opaque -> untouched


def test_the_vision_half_and_the_latent_half_are_fitted_by_the_same_rule():
    """The reference latent is spliced into the vision tokens' place, so both halves
    have to agree on shape."""
    refs = [Image.new("RGB", (200, 100), "gray"), Image.new("RGB", (60, 300), "gray")]

    images = _bare()._encoder_images(EncodePromptArgs(prompt="p", image=refs))

    assert [img.size for img in images] == [
        QwenImage21Model.REFERENCE_SIZING.target_size(*img.size) for img in refs
    ]


def test_the_adapter_does_not_override_the_shared_decode():
    """The alpha drop-out used to be a ``decode`` override; it must not come back."""
    assert "decode" not in QwenImage21Model.__dict__
    assert QwenImage21Model.decode is DiffusionModel.decode


def test_the_adapter_upscales_with_the_trained_transcoder(monkeypatch):
    """The advertised 2x is the bridge, conditioned on this model's own VAE."""
    loaded = []

    def fake_load(device, dtype):
        loaded.append((device, dtype))
        return LatentTranscodeNet(feature_channels=6, latent_channels=4, width=4, depth=1)

    monkeypatch.setattr(qwen21_transcode, "_load_net", fake_load)
    model = _bare(_upscaler=None, vae=FakeVAE())

    upscaler = model.get_upscaler()

    assert isinstance(upscaler, Qwen21TranscodeUpscaler)
    assert upscaler.vae is model.vae  # conditioned on the model's own decoder
    assert upscaler.scale == QwenImage21Model.UPSCALE_SCALE
    assert loaded == [(torch.device("cpu"), torch.float32)]
    assert model.get_upscaler() is upscaler  # built once, then cached


def test_qkv_checkpoint_fold_and_quantized_undo(tmp_path):
    prefix = "transformer_blocks.0.attn."
    state = {}
    for i, part in enumerate("qkv", 1):
        name = f"{prefix}to_{part}"
        state[name + ".weight"] = torch.full((256, 256), i, dtype=torch.int8)
        state[name + ".weight_scale"] = torch.full((256, 1), i / 100)
        state[name + ".comfy_quant"] = comfy_quant()

    merged = _fuse_qkv_state(state)
    key = prefix + "qkv"
    assert merged[key + ".weight"].shape == (768, 256)
    assert [int(x) for x in merged[key + ".weight"][:, 0][::256]] == [1, 2, 3]
    assert merged[key + ".weight_scale"].shape == (768, 1)
    assert not any(".to_q." in k or ".to_k." in k or ".to_v." in k for k in merged)

    path = write_safetensors(tmp_path / "qkv.safetensors", state)
    params = TensorWiseINT8Layout.Params(
        scale=merged[key + ".weight_scale"], orig_dtype=torch.bfloat16,
        orig_shape=(768, 256), is_weight=True, convrot=True, convrot_groupsize=256,
    )
    layer = QuantizedLinear(256, 768, bias=False)
    layer.load_quantized(QuantizedTensor(merged[key + ".weight"], "TensorWiseINT8Layout", params))
    expected = merged[key + ".weight"].clone()
    layer.weight._qdata.zero_()
    restore_quantized_layer(layer, path,
                            tuple(f"{prefix}to_{p}.weight" for p in "qkv"))
    assert torch.equal(layer.weight._qdata, expected)
    assert torch.equal(layer.weight.params.scale, merged[key + ".weight_scale"])
