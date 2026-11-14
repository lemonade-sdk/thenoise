"""The video controller driving an adapter: lifecycle, streaming, progress, cancellation.

The three component loaders are the only stubs. Staging, the sampler, the kernel contract,
the streamed decode and the job's progress clock all run for real, because that is what
this file is about.
"""
from __future__ import annotations

import io
import types

import pytest
import torch
from PIL import Image

from conftest import write_wan_dit, write_wan_vae
from thenoise.models.config import ModelConfig
from thenoise.models.wan_video import Wan22Model
from thenoise.video.controller import VideoPipelineController
from thenoise.video.jobs import Cancellation, VideoCancelled
from thenoise.video.spec import VideoSpec

FRAMES_PER_CHUNK = (1, 4, 4)  # a causal codec's first latent frame decodes alone


def _config(root):
    """The paths `fake_video` stands its three loaders in for."""
    return ModelConfig(
        dit_paths={"": write_wan_dit(root / "dit.safetensors")},
        vae_path=str(write_wan_vae(root / "vae.safetensors")),
        text_encoder_path=str(root), device="cpu", offload_device="cpu")


@pytest.fixture
def fake_video(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    calls = []

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
            self.cache_clears = 0

        @property
        def dtype(self):
            return self.weight.dtype

        def clear_cache(self):
            self.cache_clears += 1

        z_dim = 48
        pixel_channels = 3
        spatial_compression = 16
        temporal_compression = 4

        def encode_pixels_to_latents(self, pixels):
            calls.append("vae_encode")
            return torch.zeros(pixels.shape[0], 48, 1,
                               pixels.shape[-2] // 16, pixels.shape[-1] // 16)

        def decode_to_pixel_frames(self, z):
            calls.append("vae_decode")
            try:
                for i in range(z.shape[2]):
                    yield torch.zeros(z.shape[0], 3, FRAMES_PER_CHUNK[i],
                                      z.shape[3] * 16, z.shape[4] * 16)
            finally:
                self.clear_cache()

    class DiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float16))
            self.params = types.SimpleNamespace(patch_size=(1, 2, 2))

        @property
        def dtype(self):
            return self.weight.dtype

        def forward(self, x, timestep, encoder_hidden_states):
            calls.append("dit")
            return torch.zeros_like(x)

    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

        def forward(self, ids, mask):
            return torch.zeros(1, 16, 32, dtype=self.weight.dtype)

    class Tokenizer:
        def encode(self, prompt):
            return torch.ones(1, 16, dtype=torch.long), torch.ones(1, 16, dtype=torch.bool)

    monkeypatch.setattr("thenoise.models.wan_video.load_wan_video_vae",
                        lambda _path, _device: calls.append("load_vae") or VAE())
    monkeypatch.setattr("thenoise.models.wan_video.load_wan_native_transformer",
                        lambda _path, _device: calls.append("load_dit") or DiT())
    monkeypatch.setattr("thenoise.models.wan_video.load_wan_text_encoder",
                        lambda _path, _device: calls.append("load_text") or (Text(), Tokenizer()))

    def stub_encode_video(frames, _spec, _token, dest, _progress):
        for _ in frames:  # a real muxer pulls frames lazily; so must this stub
            pass
        dest.write_bytes(b"video")

    monkeypatch.setattr("thenoise.video.controller.encode_video", stub_encode_video)

    # The adapter reads the DiT and codec headers for its variant and its pairing even
    # though every loader here is a stub, so both files have to be real.
    video = Wan22Model(config=_config(root))
    return video, VideoPipelineController(video), calls


def _spec(video, *, image=None, **kwargs):
    return video.resolve_video(VideoSpec(
        prompt="fox", width=480, height=320, frames=9, steps=2, image=image, **kwargs))


def test_construction_loads_every_component_once(fake_video):
    """There is no second entry point: an adapter is loaded when it is built."""
    video, _, calls = fake_video
    assert video.dit is not None and video.text_encoder is not None
    assert [calls.count(name) for name in
            ("load_dit", "load_text", "load_vae")] == [1, 1, 1]


def test_native_t2v_i2v_t2v_reuses_models(fake_video, tmp_path):
    video, controller, calls = fake_video
    image = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(image, format="PNG")
    out = tmp_path / "out.mp4"
    for spec in (_spec(video), _spec(video, image=image.getvalue()), _spec(video)):
        controller.render(spec, Cancellation(), lambda *_: None, out)
    assert calls.count("load_dit") == 1  # a render never loads for itself
    # Only the I2V request encodes a first frame.
    assert calls.count("vae_encode") == 1
    assert video.vae.dtype is torch.float32
    assert video.dit.dtype is torch.float16
    assert video.text_encoder.weight.dtype is torch.bfloat16
    assert out.read_bytes() == b"video"


def test_unload_releases_weights_and_allows_reloading(fake_video, tmp_path):
    video, controller, calls = fake_video
    controller.render(_spec(video), Cancellation(), lambda *_: None, tmp_path / "clip.mp4")
    old_manager = video.memory

    video.unload()
    assert (video.vae, video.text_encoder, video.tokenizer, video.dit) == (None,) * 4
    assert video.memory is not old_manager
    assert video.memory.weight_bytes("dit") == 0

    # A reload is a rebuild, not a second call into the adapter that was unloaded.
    fresh = Wan22Model(config=_config(tmp_path / "model"))
    controller2 = VideoPipelineController(fresh)
    controller2.render(_spec(fresh), Cancellation(), lambda *_: None,
                       tmp_path / "again.mp4")
    assert (tmp_path / "again.mp4").read_bytes() == b"video"
    assert calls.count("load_dit") == calls.count("load_vae") == 2


def test_cancel_sampling_recovers_for_next_job(fake_video, tmp_path):
    video, controller, calls = fake_video
    token = Cancellation()

    def cancel_at_one(stage, step, _total):
        if stage == "sampling" and step == 1:
            token.cancel()

    with pytest.raises(VideoCancelled):
        controller.render(_spec(video), token, cancel_at_one, tmp_path / "clip.mp4")
    controller.render(_spec(video), Cancellation(), lambda *_: None, tmp_path / "clip.mp4")
    assert calls.count("load_dit") == 1  # a cancelled job leaves the components usable


def test_decode_progress_uses_shared_tqdm_style(fake_video, tmp_path, monkeypatch):
    seen = []

    class Bar:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, value):
            seen.append(("update", value))

    monkeypatch.setattr("thenoise.video.controller.tqdm",
                        lambda *, total, desc: seen.append((desc, total)) or Bar())
    video, controller, _calls = fake_video
    controller.render(_spec(video), Cancellation(), lambda *_: None, tmp_path / "clip.mp4")
    assert seen == [("decoding", 9)] + [("update", 1)] * 9


def test_decode_progress_and_recovery(fake_video, tmp_path):
    video, controller, _calls = fake_video
    events = []
    controller.render(_spec(video), Cancellation(), lambda *e: events.append(e),
                      tmp_path / "clip.mp4")
    assert ("sampling", 1, 2) in events
    assert [e for e in events if e[0] == "decoding"] == [
        ("decoding", i, 9) for i in range(10)
    ]
    assert ("encoding", 0, 9) in events
    assert video.vae.cache_clears >= 1


def test_an_unresolved_size_and_frame_count_reach_the_kernels(fake_video):
    """The controller resolves nothing itself; the adapter's grid decides."""
    video, controller, _calls = fake_video
    spec = controller.resolve(VideoSpec(prompt="fox", width=500, height=330, duration=1.2))
    # 1.2 s at the variant's 24 fps, then onto the codec's 4n+1 grid.
    assert (spec.width, spec.height, spec.frame_count) == (480, 320, 29)
    assert spec.explicit_frames is False
