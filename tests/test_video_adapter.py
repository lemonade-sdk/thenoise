"""Native video lifecycle and cancellation checks."""
from __future__ import annotations

import io
import sys
import threading
import types

import pytest
import torch
from PIL import Image

from thenoise.video.jobs import Cancellation, VideoCancelled
from thenoise.video.spec import VideoSpec
from thenoise.models.wan_video import WanVideo


@pytest.fixture
def fake_video(tmp_path, monkeypatch):
    root = tmp_path / "model"
    root.mkdir()
    calls = []

    class Decoder(torch.nn.Module):
        def forward(self, z):
            return z

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0012345], dtype=torch.float32))
            self.config = types.SimpleNamespace(
                z_dim=48, latents_mean=[0.0] * 48, latents_std=[1.0] * 48,
            )
            self.decoder = Decoder()
            self.cache_clears = 0

        @property
        def dtype(self):
            return self.weight.dtype

        def clear_cache(self):
            self.cache_clears += 1

        def encode(self, x, return_dict=False):
            calls.append("vae_encode")
            latent = torch.zeros(x.shape[0], 48, 1, x.shape[-2] // 16, x.shape[-1] // 16)
            return (types.SimpleNamespace(mode=lambda: latent),)

        def decode_frames(self, z):
            calls.append("vae_decode")
            try:
                for i in range(z.shape[2]):
                    self.decoder(z[:, :, i:i + 1])
                    yield torch.zeros(z.shape[0], 3, 1, z.shape[3] * 16, z.shape[4] * 16)
            finally:
                self.clear_cache()

    class DiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float16))
            self.config = types.SimpleNamespace(patch_size=(1, 2, 2))

        @property
        def dtype(self):
            return self.weight.dtype

    class Text(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

        def forward(self, ids, mask):
            return torch.zeros(1, 16, 32, dtype=self.weight.dtype)

    class Tokenizer:
        def encode(self, prompt):
            return torch.ones(1, 16, dtype=torch.long), torch.ones(1, 16, dtype=torch.bool)

    def load_vae(_root, *, check_cancel):
        check_cancel()
        calls.append("load_vae")
        return VAE()

    def load_dit(_root, *, dtype, check_cancel):
        check_cancel()
        assert dtype is torch.float16
        calls.append("load_dit")
        return DiT()

    def load_text(_root, *, dtype, check_cancel):
        check_cancel()
        assert dtype is torch.bfloat16
        calls.append("load_text")
        return Text(), Tokenizer()

    def sample(_self, latents, spec, positive, negative, condition, token, progress):
        calls.append("sample_" + ("i2v" if condition is not None else "t2v"))
        for i in range(spec.steps):
            token.check()
            progress("sampling", i + 1, spec.steps)
        return latents

    class FakeStage:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def run(func, *args):
            return func(*args)

    class FakeMemory:
        def __init__(self, *args):
            self.items = {}

        def register(self, key, value):
            self.items[key] = value

        def stage(self, *args, **kwargs):
            return FakeStage()

        def offload(self, *args):
            pass

    import thenoise.models.wan_video as wan
    monkeypatch.setattr(wan, "load_wan_native_video_vae", load_vae)
    monkeypatch.setattr(wan, "load_wan_native_transformer", load_dit)
    monkeypatch.setattr(wan, "load_wan_text_encoder", load_text)
    monkeypatch.setattr(wan.WanVideo, "_sample", sample)
    import thenoise.models.video_base as video_base
    monkeypatch.setattr(video_base, "MemoryManager", FakeMemory)

    def stub_encode_video(frames, _spec, _token, dest, _progress):
        for _ in frames:  # a real muxer pulls frames lazily; so must this stub
            pass
        dest.write_bytes(b"video")

    monkeypatch.setattr(wan, "encode_video", stub_encode_video)

    inference = types.ModuleType("thenoise.inference")
    from contextlib import nullcontext
    inference.inference = nullcontext
    inference.inference_lock = threading.Lock()
    monkeypatch.setitem(sys.modules, "thenoise.inference", inference)
    config = types.SimpleNamespace(dit_path=str(root), vae_path=str(root),
                                   text_encoder_path=str(root), device="cpu", offload_device="cpu")
    return WanVideo(config=config), calls


def _spec(*, image=None):
    return VideoSpec(prompt="fox", width=480, height=320, frames=9,
                     steps=2, image=image).resolve(WanVideo.DEFAULT_PREFS)



def test_native_model_can_be_initialized_before_first_request(fake_video):
    video, calls = fake_video
    video.load_weights()
    video.load_weights()
    assert calls == ["load_vae", "load_text", "load_dit"]

def test_native_t2v_i2v_t2v_reuses_models(fake_video, tmp_path):
    video, calls = fake_video
    image = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(image, format="PNG")
    out = tmp_path / "out.mp4"
    for spec in (_spec(), _spec(image=image.getvalue()), _spec()):
        video.generate(spec, Cancellation(), lambda *_: None, out)
    assert calls[:3] == ["load_vae", "load_text", "load_dit"]
    assert calls.count("load_dit") == calls.count("load_text") == 1
    assert [s for s in calls if s.startswith("sample_")] == ["sample_t2v", "sample_i2v", "sample_t2v"]
    assert video.vae.dtype is torch.float32
    assert video.transformer.dtype is torch.float16
    assert video.text_encoder.weight.dtype is torch.bfloat16
    assert out.read_bytes() == b"video"


def test_unload_releases_weights_and_allows_reloading(fake_video, tmp_path, monkeypatch):
    video, calls = fake_video
    video.generate(_spec(), Cancellation(), lambda *_: None, tmp_path / "clip.mp4")
    import thenoise.models.video_base as video_base
    released = []
    monkeypatch.setattr(video_base, "release_device_memory", released.append)

    video.unload()
    assert video.vae is video.text_encoder is video.transformer is video.tokenizer is None
    assert video.memory.items == {}
    assert released == [video.device]

    video.generate(_spec(), Cancellation(), lambda *_: None, tmp_path / "clip.mp4")
    assert calls.count("load_dit") == calls.count("load_vae") == 2


def test_cancel_sampling_recovers_for_next_job(fake_video, tmp_path):
    video, calls = fake_video
    token = Cancellation()

    def cancel_after_one(stage, *_):
        if stage == "sampling":
            token.cancel()

    with pytest.raises(VideoCancelled):
        video.generate(_spec(), token, cancel_after_one, tmp_path / "clip.mp4")
    video.generate(_spec(), Cancellation(), lambda *_: None, tmp_path / "clip.mp4")
    assert calls.count("load_dit") == 1


def test_decode_progress_uses_shared_tqdm_style(fake_video, tmp_path, monkeypatch):
    import thenoise.models.wan_video as wan

    seen = []

    class Bar:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def update(self, value):
            seen.append(("update", value))

    def fake_tqdm(*, total, desc):
        seen.append((desc, total))
        return Bar()

    monkeypatch.setattr(wan, "tqdm", fake_tqdm)
    video, _calls = fake_video
    video.generate(_spec(), Cancellation(), lambda *_: None, tmp_path / "clip.mp4")
    assert seen == [("decoding", 3), ("update", 1), ("update", 1), ("update", 1)]


def test_decode_progress_and_recovery(fake_video, tmp_path):
    video, calls = fake_video
    events = []
    video.generate(_spec(), Cancellation(), lambda *e: events.append(e), tmp_path / "clip.mp4")
    assert ("decoding", 0, 3) in events
    assert [e for e in events if e[0] == "decoding" and e[1] > 0] == [
        ("decoding", 1, 3), ("decoding", 2, 3), ("decoding", 3, 3)
    ]
    assert ("encoding", 0, 9) in events
    assert video.vae.cache_clears >= 1
