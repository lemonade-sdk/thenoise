"""Video architecture seams: shared weights, model paths and adapter lifecycle."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from thenoise.models.video_base import VideoModel
from thenoise.models.wan_video import WanVideo
from thenoise.utils.checkpoint_weights import load_mapped_weights
from thenoise.utils.loader import load_mapped_safetensors
from thenoise.utils.model_dir import find_weight_file
from thenoise.vae.wan22 import WAN22_LATENTS_MEAN, WAN22_LATENTS_STD
from thenoise.vae.wan_video import _comfy_vae_config
from thenoise.video.spec import VideoSpec


def test_video_model_contract_and_shared_latent_statistics():
    assert issubclass(WanVideo, VideoModel)
    assert WanVideo.resolve_video.__func__ is not VideoModel.resolve_video.__func__
    config = _comfy_vae_config()
    assert config.latents_mean == WAN22_LATENTS_MEAN
    assert config.latents_std == WAN22_LATENTS_STD


def test_video_base_is_usable_without_wan(tmp_path, monkeypatch):
    """A future adapter reuses resolution, arbitrary components and teardown."""
    class AnotherVideo(VideoModel):
        name = "other-video"
        CAPABILITIES = {**VideoModel.CAPABILITIES, "t2v": True}
        DEFAULT_PREFS = {
            "negative_prompt": "", "width": 640, "height": 384,
            "frames": 17, "fps": 25, "steps": 8, "guidance_scale": 1.0,
            "output_format": "mp4", "sampler": "euler",
        }

        @staticmethod
        def detect(_checkpoint):
            return False

        def load_weights(self):
            raise AssertionError("not a server startup test")

        def generate(self, spec, token, progress, output):
            raise AssertionError("not an inference test")

        def _clear_weights(self):
            self.extra_component = None

    config = SimpleNamespace(
        dit_path=str(tmp_path / "dit"), vae_path=str(tmp_path / "vae"),
        text_encoder_path=str(tmp_path / "text"), device="cpu", offload_device="cpu",
    )
    model = AnotherVideo(config=config)
    model.extra_component = object()
    resolved = model.resolve_video(VideoSpec(prompt="another model", width=800))
    assert (resolved.width, resolved.height, resolved.steps, resolved.frame_count) == (
        800, 384, 8, 17,
    )

    model.memory.register("audio_vae", torch.nn.Linear(1, 1))
    assert model.memory.weight_bytes("audio_vae") > 0
    old_manager = model.memory
    released = []
    monkeypatch.setattr("thenoise.models.video_base.release_device_memory", released.append)
    model.unload()
    assert model.extra_component is None
    assert model.memory is not old_manager
    assert model.memory.weight_bytes("audio_vae") == 0
    assert released == [model.device]


def test_streaming_checkpoint_uses_shared_key_mapping(tmp_path):
    path = tmp_path / "float.safetensors"
    save_file({"raw.weight": torch.ones(3, 2, dtype=torch.float16),
               "raw.bias": torch.zeros(3, dtype=torch.float16)}, str(path))
    with torch.device("meta"):
        model = torch.nn.Linear(2, 3)
    load_mapped_safetensors(
        model, path, lambda key: key.removeprefix("raw."),
        dtype_for=lambda _: torch.float32, label="test",
    )
    assert model.weight.dtype == torch.float32
    assert model.weight.shape == (3, 2)
    assert not any(p.is_meta for p in model.parameters())


def test_streaming_checkpoint_validates_before_assigning(tmp_path):
    path = tmp_path / "bad.safetensors"
    save_file({"weight": torch.ones(3, 2, dtype=torch.float16),
               "bias": torch.zeros(4, dtype=torch.float16)}, str(path))
    with torch.device("meta"):
        model = torch.nn.Linear(2, 3)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_mapped_safetensors(model, path, lambda key: key,
                                dtype_for=lambda _: torch.float16, label="test")
    assert all(p.is_meta for p in model.parameters())


def test_streaming_loader_keeps_cpu_weights_memory_mapped(tmp_path, monkeypatch):
    from safetensors import safe_open as open_safetensors
    import thenoise.utils.loader as loader

    path = tmp_path / "mapped.safetensors"
    save_file({"weight": torch.full((4, 512), 1.5, dtype=torch.float16),
               "bias": torch.zeros(4, dtype=torch.float16)}, str(path))

    opens = []

    def capture_mmap(*args, **kwargs):
        opens.append((args, kwargs))
        return open_safetensors(*args, **kwargs)

    monkeypatch.setattr(loader, "safe_open", capture_mmap)
    with torch.device("meta"):
        model = torch.nn.Linear(512, 4)
    loader.load_mapped_safetensors(
        model, path, lambda key: key,
        dtype_for=lambda _: torch.float16, label="test",
    )
    assert len(opens) == 1
    assert opens[0][1] == {"framework": "pt", "device": "cpu"}
    # File ownership/lifetime is handled by safetensors; the parameter remains
    # valid after the file context has been closed, without a copy on read.
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 1.5))


def test_find_weight_file_supports_comfy_layout_and_explicit_file(tmp_path):
    path = tmp_path / "split_files" / "vae" / "wan2.2_vae.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"dummy")
    assert find_weight_file(tmp_path, "vae", ("wan2.2_vae.safetensors",)) == path
    assert find_weight_file(path, "text_encoders", ("missing.safetensors",)) == path
    with pytest.raises(FileNotFoundError, match="checkpoint missing"):
        find_weight_file(tmp_path, "text_encoders", ("missing.safetensors",))
    # The path resolver is layout-based, not hardcoded to Wan filenames.
    other = tmp_path / "another_model" / "diffusion_models" / "other_dit.safetensors"
    other.parent.mkdir(parents=True)
    other.touch()
    assert find_weight_file(
        tmp_path / "another_model", "diffusion_models", ("other_dit.safetensors",)
    ) == other


def test_mapped_checkpoint_loading_validates_then_assigns(tmp_path):
    checkpoint = tmp_path / "small.safetensors"
    weights = {
        "wrapped.weight": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "wrapped.bias": torch.ones(3, dtype=torch.float32),
    }
    save_file(weights, str(checkpoint))
    with torch.device("meta"):
        model = torch.nn.Linear(2, 3)
    checks = []
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        load_mapped_weights(
            source, model, lambda key: key.removeprefix("wrapped."),
            dtype_for=lambda key: torch.float16 if key == "weight" else torch.float32,
            excluded=frozenset(), check_cancel=lambda: checks.append(True),
            label="test",
        )
    assert checks
    assert model.weight.dtype is torch.float16
    assert model.bias.dtype is torch.float32
    assert not any(p.is_meta for p in model.parameters())
    assert torch.equal(model.weight.float(), weights["wrapped.weight"])


def test_mapped_checkpoint_rejects_wrong_shape_before_assignment(tmp_path):
    checkpoint = tmp_path / "wrong.safetensors"
    save_file({
        "weight": torch.zeros(5, 2),
        "bias": torch.zeros(3),
    }, str(checkpoint))
    with torch.device("meta"):
        model = torch.nn.Linear(2, 3)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        with pytest.raises(ValueError, match="shape mismatch"):
            load_mapped_weights(
                source, model, lambda key: key,
                dtype_for=lambda _: torch.float32, excluded=frozenset(),
                check_cancel=None, label="test",
            )
    assert all(p.is_meta for p in model.parameters())
