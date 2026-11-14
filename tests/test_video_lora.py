"""Video LoRA switching reuses the common image-model LoRA machinery."""
import pytest
import torch
from safetensors.torch import save_file

from conftest import stub_wan_components, wan_adapter, write_wan_dit, write_wan_vae
from thenoise.dit.quantized import QuantizedLinear
from thenoise.models.config import ModelConfig
from thenoise.models.wan_video import Wan22Model
from thenoise.models.base import DiffusionModel


def _config(tmp_path):
    """A real (tiny) DiT, because the adapter reads the variant out of its header.

    Kept out of ``lora_dir``, which is scanned recursively and would otherwise offer
    the DiT itself as a LoRA.
    """
    (tmp_path / "loras").mkdir(exist_ok=True)
    return ModelConfig(
        dit_paths={"": write_wan_dit(tmp_path / "dit.safetensors")},
        vae_path=str(write_wan_vae(tmp_path / "vae.safetensors")),
        text_encoder_path="te",
        device="cpu", offload_device="cpu", lora_dir=str(tmp_path / "loras"))


@pytest.fixture(autouse=True)
def stub_components(monkeypatch):
    """LoRA bookkeeping never reaches the codec or the encoder, but the adapter loads
    both as it is built."""
    vae, text_encoder = stub_wan_components()
    monkeypatch.setattr("thenoise.models.wan_video.load_wan_video_vae", vae)
    monkeypatch.setattr("thenoise.models.wan_video.load_wan_text_encoder", text_encoder)


def test_video_loras_use_the_shared_switch_and_decline_two_experts():
    """Listing and switching come from ``LoRASupport`` — the single-expert case is proved
    by the apply/undo test below. All Wan adds is refusing a LoRA it could only apply to
    one of its two DiTs (plan C14 lifts that by role.)
    """
    assert Wan22Model.list_loras is DiffusionModel.list_loras
    assert Wan22Model._lora_key_map is not DiffusionModel._lora_key_map
    with pytest.raises(ValueError, match="expert-specific"):
        wan_adapter("t2v_14b").switch_loras(["style"], torch.nn.Module())


@pytest.mark.parametrize("raw,canonical", [
    ("blocks.0.self_attn.q", "blocks.0.attn1.to_q"),
    ("diffusion_model.blocks.0.cross_attn.v", "blocks.0.attn2.to_v"),
    ("lora_unet_blocks_0_self_attn_q", "blocks.0.attn1.to_q"),
    ("blocks.0.ffn.0", "blocks.0.ffn_in"),
    ("blocks.0.ffn.2", "blocks.0.ffn_out"),
])
def test_wan_lora_key_map(raw, canonical):
    video = wan_adapter()
    assert video._lora_key_map(raw + ".lora_A.weight") == canonical + ".lora_A.weight"
    assert video._lora_key_map(raw + ".lora_B.weight") == canonical + ".lora_B.weight"


def test_wan_applies_switches_and_undoes_native_lora(tmp_path):
    video = Wan22Model(config=_config(tmp_path))
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([torch.nn.Module()])
    dit.blocks[0].attn1 = torch.nn.Module()
    dit.blocks[0].attn1.to_q = QuantizedLinear(4, 4, bias=False)
    original = dit.blocks[0].attn1.to_q.weight.detach().clone()
    save_file({
        "blocks.0.self_attn.q.lora_A.weight": torch.ones(1, 4),
        "blocks.0.self_attn.q.lora_B.weight": torch.ones(4, 1),
    }, str(tmp_path / "loras" / "style.safetensors"))

    assert video.list_loras() == ["style"]
    video.switch_loras(["style:0.5"], dit)
    torch.testing.assert_close(dit.blocks[0].attn1.to_q.weight, original + 0.5,
                               atol=1e-6, rtol=1e-6)
    video.switch_loras(["style:0.5"], dit)  # idempotent
    video.switch_loras(None, dit)
    torch.testing.assert_close(dit.blocks[0].attn1.to_q.weight, original,
                               atol=1e-6, rtol=1e-6)


def test_video_lora_names_cannot_escape_model_lora_directory(tmp_path):
    video = Wan22Model(config=_config(tmp_path))
    with pytest.raises(ValueError, match="escapes base directory"):
        video.switch_loras(["../escape"], torch.nn.Module())
