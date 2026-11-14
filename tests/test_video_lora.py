"""Video LoRA switching reuses the common image-model LoRA machinery."""
import pytest
import torch
from safetensors.torch import save_file

from conftest import stub_wan_components, wan_adapter, write_wan_dit, write_wan_vae
from thenoise.dit.quantized import QuantizedLinear
from thenoise.models.config import ModelConfig
from thenoise.models.wan_video import Wan22Model
from thenoise.errors import ModelReloadNeeded
from thenoise.models.base import DiffusionModel


def _config(tmp_path, *, experts: bool = False):
    """A real (tiny) DiT, because the adapter reads the variant out of its header.

    Kept out of ``lora_dir``, which is scanned recursively and would otherwise offer
    the DiT itself as a LoRA.
    """
    (tmp_path / "loras").mkdir(exist_ok=True)
    channels = 16 if experts else 48
    dit = write_wan_dit(tmp_path / "dit.safetensors", channels=channels)
    return ModelConfig(
        dit_paths={"": dit, "low": dit} if experts else {"": dit},
        vae_path=str(write_wan_vae(tmp_path / "vae.safetensors", channels=channels)),
        text_encoder_path="te",
        device="cpu", offload_device="cpu", lora_dir=str(tmp_path / "loras"))


def _expert_pair(tmp_path):
    """Two LoRA-able modules, and two LoRAs whose deltas are 1x and 2x the weight."""
    def target():
        dit = torch.nn.Module()
        dit.blocks = torch.nn.ModuleList([torch.nn.Module()])
        dit.blocks[0].attn1 = torch.nn.Module()
        dit.blocks[0].attn1.to_q = QuantizedLinear(4, 4, bias=False)
        return dit, dit.blocks[0].attn1.to_q.weight.detach().clone()

    for name, magnitude in (("style", 1.0), ("pose", 2.0)):
        save_file({
            "blocks.0.self_attn.q.lora_A.weight": torch.ones(1, 4),
            "blocks.0.self_attn.q.lora_B.weight": torch.full((4, 1), magnitude),
        }, str(tmp_path / "loras" / f"{name}.safetensors"))
    return target(), target()


@pytest.fixture(autouse=True)
def stub_components(monkeypatch):
    """LoRA bookkeeping never reaches the codec or the encoder, but the adapter loads
    both as it is built."""
    vae, text_encoder = stub_wan_components()
    monkeypatch.setattr("thenoise.models.wan_video.load_wan_video_vae", vae)
    monkeypatch.setattr("thenoise.models.wan_video.load_wan_text_encoder", text_encoder)


def test_video_loras_use_the_shared_switch():
    """Listing and switching come from ``LoRASupport``; all the adapter adds is the
    trainer's key names and a second LoRA-able component for its low-noise expert.
    """
    assert Wan22Model.list_loras is DiffusionModel.list_loras
    assert Wan22Model._lora_key_map is not DiffusionModel._lora_key_map
    assert sorted(wan_adapter().lora_components()) == ["dit"]
    assert sorted(wan_adapter("t2v_14b").lora_components()) == ["dit", "dit_low"]


def test_each_expert_takes_its_own_loras(tmp_path):
    """The role is the component: what one expert is asked for never reaches the other,
    and switching one leaves the other's LoRA in place.
    """
    video = Wan22Model(config=_config(tmp_path, experts=True))
    (dit, dit_original), (low, low_original) = _expert_pair(tmp_path)

    video.switch_loras(["style"], dit, "dit")
    video.switch_loras(["pose:0.5"], low, "dit_low")
    torch.testing.assert_close(dit.blocks[0].attn1.to_q.weight, dit_original + 1.0,
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(low.blocks[0].attn1.to_q.weight, low_original + 1.0,
                               atol=1e-6, rtol=1e-6)

    video.switch_loras(None, dit, "dit")
    torch.testing.assert_close(dit.blocks[0].attn1.to_q.weight, dit_original,
                               atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(low.blocks[0].attn1.to_q.weight, low_original + 1.0,
                               atol=1e-6, rtol=1e-6)


def test_a_role_the_model_does_not_offer_is_refused_at_resolve():
    """/videos answers a bad role at submit, and the two-expert model offers both.

    A bare list is the single component's, which is every request written before roles.
    """
    from thenoise.video.spec import VideoSpec

    single = wan_adapter()
    with pytest.raises(ValueError, match="does not offer"):
        single.resolve_video(VideoSpec(prompt="fox", lora_specs={"dit_low": ["style"]}))
    assert single.resolve_video(
        VideoSpec(prompt="fox", lora_specs=["style"])).lora_specs == {"dit": ["style"]}
    both = wan_adapter("t2v_14b")
    assert both.resolve_video(VideoSpec(
        prompt="fox",
        lora_specs={"dit": ["a"], "dit_low": ["b"]})).lora_specs == {
        "dit": ["a"], "dit_low": ["b"]}
    # A bare list is the default role's on a two-expert model too: the high-noise
    # expert, not both of them.
    assert both.resolve_video(VideoSpec(prompt="fox", lora_specs=["a"])).lora_specs == {
        "dit": ["a"]}


def test_an_out_of_memory_switch_refuses_only_that_component(tmp_path, monkeypatch):
    """Half-merged weights cannot be repaired, so that component is finished on this
    instance. The other one was never touched, and keeps taking LoRAs.
    """
    video = Wan22Model(config=_config(tmp_path, experts=True))
    (dit, _), (low, _) = _expert_pair(tmp_path)

    def maybe_oom(module, _lora_sds, _multipliers, **_kwargs):
        if module is low:
            raise torch.cuda.OutOfMemoryError

    monkeypatch.setattr("thenoise.models.lora_support.apply_lora_to_model", maybe_oom)
    with pytest.raises(ModelReloadNeeded, match="'dit_low'"):
        video.switch_loras(["pose"], low, "dit_low")
    with pytest.raises(ModelReloadNeeded, match="'dit_low'"):
        video.switch_loras(None, low, "dit_low")
    video.switch_loras(["style"], dit, "dit")


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
    video.switch_loras(["style:0.5"], dit)  # a bare role: the single component's
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
