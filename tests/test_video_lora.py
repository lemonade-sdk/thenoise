"""Video LoRA switching reuses the common image-model LoRA machinery."""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from thenoise.dit.quantized import QuantizedLinear
from thenoise.models.wan_video import WanVideo
from thenoise.models.base import DiffusionModel


def test_video_and_image_share_lora_switch_implementation():
    assert WanVideo.switch_loras is DiffusionModel.switch_loras
    assert WanVideo.list_loras is DiffusionModel.list_loras


@pytest.mark.parametrize("raw,canonical", [
    ("blocks.0.self_attn.q", "blocks.0.attn1.to_q"),
    ("diffusion_model.blocks.0.cross_attn.v", "blocks.0.attn2.to_v"),
    ("lora_unet_blocks_0_self_attn_q", "blocks.0.attn1.to_q"),
    ("blocks.0.ffn.0", "blocks.0.ffn_in"),
    ("blocks.0.ffn.2", "blocks.0.ffn_out"),
])
def test_wan_lora_key_map(raw, canonical):
    video = object.__new__(WanVideo)
    assert video._lora_key_map(raw + ".lora_A.weight") == canonical + ".lora_A.weight"
    assert video._lora_key_map(raw + ".lora_B.weight") == canonical + ".lora_B.weight"


def test_wan_applies_switches_and_undoes_native_lora(tmp_path):
    cfg = SimpleNamespace(dit_path="unused.safetensors", vae_path="unused",
                          text_encoder_path="unused", device="cpu",
                          offload_device="cpu", lora_dir=str(tmp_path))
    video = WanVideo(config=cfg)
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([torch.nn.Module()])
    dit.blocks[0].attn1 = torch.nn.Module()
    dit.blocks[0].attn1.to_q = QuantizedLinear(4, 4, bias=False)
    original = dit.blocks[0].attn1.to_q.weight.detach().clone()
    save_file({
        "blocks.0.self_attn.q.lora_A.weight": torch.ones(1, 4),
        "blocks.0.self_attn.q.lora_B.weight": torch.ones(4, 1),
    }, str(tmp_path / "style.safetensors"))

    assert video.list_loras() == ["style"]
    video.switch_loras(["style:0.5"], dit)
    torch.testing.assert_close(dit.blocks[0].attn1.to_q.weight, original + 0.5,
                               atol=1e-6, rtol=1e-6)
    video.switch_loras(["style:0.5"], dit)  # idempotent
    video.switch_loras(None, dit)
    torch.testing.assert_close(dit.blocks[0].attn1.to_q.weight, original,
                               atol=1e-6, rtol=1e-6)


def test_video_lora_names_cannot_escape_model_lora_directory(tmp_path):
    cfg = SimpleNamespace(dit_path="unused", vae_path="unused", text_encoder_path="unused",
                          device="cpu", offload_device="cpu", lora_dir=str(tmp_path))
    video = WanVideo(config=cfg)
    with pytest.raises(ValueError, match="escapes base directory"):
        video.switch_loras(["../escape"], torch.nn.Module())
