"""Video pipeline lifecycle, cancellation and precision checks."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import threading
import types

import pytest
import torch
from PIL import Image

from thenoise.video.jobs import Cancellation, VideoCancelled
from thenoise.video.spec import VideoSpec
from thenoise.video.wan import WanVideo


@pytest.fixture
def adapter_env(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text(json.dumps({
        "_class_name": "WanPipeline", "expand_timesteps": True,
        "transformer_2": [None, None],
        "scheduler": ["diffusers", "UniPCMultistepScheduler"],
    }))
    loads = []
    runs = []

    class Scheduler:
        def __init__(self, **config):
            self.config = config or {
                "use_flow_sigmas": True,
                "prediction_type": "flow_prediction",
                "num_train_timesteps": 1000,
            }

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        @classmethod
        def from_config(cls, config, **overrides):
            return cls(**(dict(config) | overrides))

    class Euler(Scheduler):
        pass

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0012345], dtype=torch.float32))
            self.config = types.SimpleNamespace(z_dim=48)
            self.encoder = torch.nn.Identity()
            self.decoder = torch.nn.Identity()
            self.cache_clears = 0

        def clear_cache(self):
            self.cache_clears += 1

    class DiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
            self.config = types.SimpleNamespace(in_channels=48)

        def forward(self, x):
            return x

    class Pipeline:
        def __init__(self, *, tokenizer, text_encoder, vae, scheduler, transformer,
                     transformer_2=None, boundary_ratio=None, expand_timesteps=False):
            self.vae, self.transformer = vae, transformer
            self.text_encoder, self.tokenizer = text_encoder, tokenizer
            self.scheduler, self.transformer_2 = scheduler, transformer_2
            self.config = {"boundary_ratio": boundary_ratio, "expand_timesteps": expand_timesteps}
            self.offload_calls = 0
            self.hooks_removed = 0

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise AssertionError("video must use native DiT/VAE loaders")

        def to(self, device):
            assert device == "cpu"
            return self

        def enable_model_cpu_offload(self, *, device):
            assert device == "cpu"
            self.offload_calls += 1

        def remove_all_hooks(self):
            self.hooks_removed += 1

        def __call__(self, **params):
            mode = "i2v" if isinstance(self, ImagePipeline) else "t2v"
            runs.append((mode, params, dict(self.scheduler.config)))
            self.text_encoder(torch.ones(1))
            if mode == "i2v":
                self.vae.encoder(torch.ones(1))
            for step in range(params["num_inference_steps"]):
                self.transformer(torch.ones(1))
                params["callback_on_step_end"](self, step, 0, {"latents": None})
            for _ in range((params["num_frames"] - 1) // 4 + 1):
                self.vae.decoder(torch.ones(1))
            return types.SimpleNamespace(frames=[[
                Image.new("RGB", (params["width"], params["height"]))
                for _ in range(params["num_frames"])
            ]])

    class ImagePipeline(Pipeline):
        pass

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

    class TextEncoder(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def forward(self, x):
            return x

    diffusers = types.ModuleType("diffusers")
    diffusers.WanPipeline = Pipeline
    diffusers.WanImageToVideoPipeline = ImagePipeline
    diffusers.UniPCMultistepScheduler = Scheduler
    diffusers.FlowMatchEulerDiscreteScheduler = Euler
    schedulers = types.ModuleType("diffusers.schedulers")
    unipc = types.ModuleType("diffusers.schedulers.scheduling_unipc_multistep")
    unipc.UniPCMultistepScheduler = Scheduler
    transformers = types.ModuleType("transformers")
    transformers.T5TokenizerFast = Tokenizer
    transformers.UMT5EncoderModel = TextEncoder

    inference = types.ModuleType("thenoise.inference")
    inference.inference = contextlib.nullcontext
    inference.inference_lock = threading.RLock()
    for name, module in {
        "diffusers": diffusers,
        "diffusers.schedulers": schedulers,
        "diffusers.schedulers.scheduling_unipc_multistep": unipc,
        "transformers": transformers,
        "thenoise.inference": inference,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    from thenoise.video import native_vae, native_checkpoint

    def load_vae(path, *, check_cancel):
        check_cancel()
        loads.append("vae")
        return VAE()

    def load_dit(path, *, dtype, check_cancel):
        check_cancel()
        assert dtype is torch.bfloat16
        loads.append("dit")
        return DiT()

    monkeypatch.setattr(native_vae, "load_wan_native_video_vae", load_vae)
    monkeypatch.setattr(native_checkpoint, "load_wan_native_transformer", load_dit)
    monkeypatch.setattr("thenoise.video.wan.encode_video", lambda frames, spec, token, out, progress: out.write_bytes(b"video"))
    return model, loads, runs


def _spec(*, image=None, sampling_method="unipc"):
    return VideoSpec(prompt="fox", width=480, height=320, video_frames=9,
                     steps=2, image=image, sampling_method=sampling_method).resolve()


@pytest.mark.parametrize("offload", [False, True])
def test_native_t2v_i2v_t2v_reuses_weights_and_scheduler(adapter_env, tmp_path, offload):
    model, loads, runs = adapter_env
    video = WanVideo(str(model), device="cpu", offload=offload)
    out = tmp_path / "clip.webm"
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buffer, format="PNG")

    for spec in (_spec(), _spec(image=buffer.getvalue(), sampling_method="euler"), _spec()):
        video.generate(spec, Cancellation(), lambda *_: None, out)

    assert loads == ["vae", "dit"]
    assert [r[0] for r in runs] == ["t2v", "i2v", "t2v"]
    assert runs[1][1]["image"].size == (480, 320)
    assert runs[1][2]["shift"] == 5.0
    assert runs[2][2]["use_flow_sigmas"] is True
    assert runs[2][2]["prediction_type"] == "flow_prediction"
    assert video._pipe.vae.weight.dtype is torch.float32
    assert video._pipe.vae.weight.item() == pytest.approx(1.0012345, abs=1e-7)
    assert video._pipe.transformer.weight.dtype is torch.bfloat16
    if offload:
        assert video._pipe.offload_calls == 1
    assert out.read_bytes() == b"video"


def test_cancel_sampling_does_not_poison_next_job(adapter_env, tmp_path):
    model, loads, runs = adapter_env
    video = WanVideo(str(model), device="cpu")
    token = Cancellation()

    def cancel_at_step(stage, *_):
        if stage == "sampling":
            token.cancel()

    with pytest.raises(VideoCancelled):
        video.generate(_spec(), token, cancel_at_step, tmp_path / "clip.mp4")
    video.generate(_spec(), Cancellation(), lambda *_: None, tmp_path / "clip.mp4")
    assert loads == ["vae", "dit"]
    assert len(runs) == 2


def test_cancel_decoding_removes_hooks_and_clears_cache(adapter_env, tmp_path):
    model, loads, runs = adapter_env
    video = WanVideo(str(model), device="cpu")
    token = Cancellation()

    def cancel_during_decode(stage, *_):
        if stage == "decoding":
            token.cancel()

    with pytest.raises(VideoCancelled):
        video.generate(_spec(), token, cancel_during_decode, tmp_path / "clip.mp4")
    video.generate(_spec(), Cancellation(), lambda *_: None, tmp_path / "clip.mp4")
    assert loads == ["vae", "dit"]
    assert len(runs) == 2
    assert video._pipe.vae.cache_clears >= 2


def test_progress_reports_decode_and_encoding(adapter_env, tmp_path):
    model, _, _ = adapter_env
    video = WanVideo(str(model), device="cpu")
    events = []
    video.generate(_spec(), Cancellation(), lambda *args: events.append(args), tmp_path / "clip.mp4")
    assert ("decoding", 0, 3) in events
    assert [e for e in events if e[0] == "decoding" and e[1] > 0] == [
        ("decoding", 1, 3), ("decoding", 2, 3), ("decoding", 3, 3)
    ]
    assert ("encoding", 0, 9) in events
