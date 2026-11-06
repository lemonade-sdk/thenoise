#!/usr/bin/env python3
"""Profile Qwen-Image 2.1 sampling on the loaded GPU.

Run from the repository root with the ROCm venv. Model load and first-run compile
are timed separately from steady-state runs. `--kv-cache` / `--no-kv-cache` make
same-seed A/B runs straightforward for text-to-image and edits.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.profiler import ProfilerActivity, profile

from thenoise.runtime import ModelPaths, Runtime, Settings
from thenoise.samplers import euler as euler_module
from thenoise.models.config import GenerateRequest
from thenoise.utils import attention as attention_module
from thenoise.utils.device import synchronize_device


class SamplingInstrumentation:
    """Measure each synchronized Euler step and optionally save comparison tensors."""

    def __init__(self, synchronize_each_step: bool = True) -> None:
        self.step_times: list[tuple[int, float]] = []
        self.captured: dict[str, torch.Tensor] = {}
        self.capture = False
        self.synchronize_each_step = synchronize_each_step
        self._step_start: float | None = None
        self._step_index: int | None = None
        self._model = None
        self._denoise = None
        self._finalize = None
        self._sync = None
        self.kv_cache_bytes = 0

    def install(self, model) -> None:
        self._model = model
        self._denoise = model.denoise_step
        self._finalize = model.finalize_latent
        self._sync = euler_module.synchronize_device

        def denoise_step(latents, t, cond, guidance_scale, i):
            self._step_index = i
            self._step_start = time.perf_counter() if self.synchronize_each_step else None
            if self.capture:
                self.captured[f"step_{i:02d}_input"] = latents.detach().cpu()
            velocity = self._denoise(latents, t, cond, guidance_scale, i)
            caches = getattr(self._model, "_kv_caches", None)
            if caches:
                self.kv_cache_bytes = max(
                    self.kv_cache_bytes,
                    sum(cache.nbytes() for cache in caches.values()),
                )
            if self.capture:
                self.captured[f"step_{i:02d}_velocity"] = velocity.detach().cpu()
            return velocity

        def finalize_latent(latents, params):
            result = self._finalize(latents, params)
            if self.capture:
                self.captured["final_latents"] = result.detach().cpu()
            return result

        def synchronize_and_record(device):
            if not self.synchronize_each_step:
                return
            self._sync(device)
            if self._step_start is not None and self._step_index is not None:
                self.step_times.append(
                    (self._step_index, time.perf_counter() - self._step_start)
                )
                self._step_start = None
                self._step_index = None

        model.denoise_step = denoise_step
        model.finalize_latent = finalize_latent
        euler_module.synchronize_device = synchronize_and_record

    def restore(self) -> None:
        if self._model is not None:
            self._model.denoise_step = self._denoise
            self._model.finalize_latent = self._finalize
        if self._sync is not None:
            euler_module.synchronize_device = self._sync


def _reset_sampling_cache(pipeline) -> None:
    """Keep prompt/reference encodings warm while forcing a new denoise pass."""
    pipeline._cache._sampling.clear()
    pipeline._cache._decode.clear()


def _run(runtime: Runtime, request: GenerateRequest) -> Image.Image:
    def action(pipeline):
        _reset_sampling_cache(pipeline)
        if request.image is None:
            return pipeline.generate(request)
        return pipeline.edit(request)

    return runtime.run(action)


def _ssim(reference: Image.Image, result: Image.Image) -> float:
    """Mean 11x11 Gaussian-window SSIM, without an extra image-metrics dependency."""
    a = np.asarray(reference.convert("RGB"), dtype=np.float32).copy()
    b = np.asarray(result.convert("RGB"), dtype=np.float32).copy()
    if a.shape != b.shape:
        raise ValueError(f"image dimensions differ: {a.shape} != {b.shape}")
    a = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
    b = torch.from_numpy(b).permute(2, 0, 1).unsqueeze(0)
    coords = torch.arange(11, dtype=torch.float32) - 5
    one_d = torch.exp(-(coords * coords) / (2 * 1.5**2))
    one_d /= one_d.sum()
    kernel = (one_d[:, None] * one_d[None, :]).expand(3, 1, 11, 11)

    def blur(x):
        x = F.pad(x, (5, 5, 5, 5), mode="reflect")
        return F.conv2d(x, kernel, groups=3)

    mu_a, mu_b = blur(a), blur(b)
    var_a = (blur(a * a) - mu_a * mu_a).clamp_min(0)
    var_b = (blur(b * b) - mu_b * mu_b).clamp_min(0)
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a.square() + mu_b.square() + c1) * (var_a + var_b + c2)
    )
    return float(score.mean())


def _tensor_metrics(reference: torch.Tensor, result: torch.Tensor) -> dict[str, float]:
    if reference.shape != result.shape:
        raise ValueError(f"latent shapes differ: {reference.shape} != {result.shape}")
    delta = result.float() - reference.float()
    return {
        "max_abs": float(delta.abs().max()),
        "rms": float(delta.square().mean().sqrt()),
        "relative_rms": float(
            delta.square().mean().sqrt()
            / reference.float().square().mean().sqrt().clamp_min(1e-12)
        ),
    }


def _profile_summary(profiler, trace_path: Path) -> dict[str, Any]:
    profiler.export_chrome_trace(str(trace_path))
    events = profiler.events()
    cuda_events = [
        event for event in events
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    transfers = [
        event.name for event in events
        if any(tag in event.name.lower() for tag in ("memcpy", "memset"))
    ]
    return {
        "trace": str(trace_path),
        "gpu_kernel_events": len(cuda_events),
        "memory_transfer_events": len(transfers),
        "transfer_names": transfers[:40],
        "top_device_ops": profiler.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=35
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", default="A detailed still life with small printed text")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--configuration", default=None,
                        help="label recorded in the JSON report")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--latents-out", type=Path)
    parser.add_argument("--compare-image", type=Path)
    parser.add_argument("--compare-latents", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--disable-compile", action="store_true",
        help="diagnostic eager-block run; default uses the repository's compiled blocks",
    )
    parser.add_argument(
        "--compile-dit", action="store_true",
        help="diagnostic outer DiT compile boundary around the compiled blocks",
    )
    parser.add_argument(
        "--no-step-sync", action="store_true",
        help="diagnostic run without Euler's per-step device synchronization; total "
             "time is still synchronized and per-step timings are omitted",
    )
    parser.add_argument(
        "--skip-attention-layout-copy", action="store_true",
        help="diagnostic run that passes Q/K/V strides directly to SDPA, bypassing "
             "the normal L-major layout copies",
    )
    qkv = parser.add_mutually_exclusive_group()
    qkv.add_argument(
        "--fuse-qkv", dest="fuse_qkv", action="store_true",
        help="force joined Q/K/V projections (the quantized Qwen adapter does this by default)",
    )
    qkv.add_argument(
        "--no-fuse-qkv", dest="fuse_qkv", action="store_false",
        help="use the separate checkpoint projections for A/B comparison",
    )
    parser.set_defaults(fuse_qkv=None)
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--kv-cache", dest="kv_cache", action="store_true")
    cache.add_argument("--no-kv-cache", dest="kv_cache", action="store_false")
    parser.set_defaults(kv_cache=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.iterations < 1 or args.steps < 1:
        raise SystemExit("--iterations and --steps must be positive")
    if not args.device.startswith("cuda"):
        raise SystemExit("this profiler is intended for the ROCm/CUDA device")
    if not torch.cuda.is_available():
        raise SystemExit("PyTorch reports no ROCm/CUDA device")
    if args.disable_compile:
        torch._dynamo.config.disable = True
    if args.disable_compile and args.compile_dit:
        raise SystemExit("--disable-compile and --compile-dit cannot be combined")

    device = torch.device(args.device)
    prop = torch.cuda.get_device_properties(device)
    if getattr(prop, "gcnArchName", "").split(":")[0] != "gfx1151":
        print(f"warning: expected gfx1151, found {getattr(prop, 'gcnArchName', prop.name)}")

    reference = Image.open(args.reference_image) if args.reference_image else None
    request = GenerateRequest(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        sampler="euler",
        kv_cache=args.kv_cache,
        image=reference,
    )

    runtime = Runtime(Settings(device=args.device))
    t0 = time.perf_counter()
    runtime.load(ModelPaths(args.dit, args.vae, args.text_encoder))
    load_seconds = time.perf_counter() - t0
    model = runtime.model
    original_uniform_layout = attention_module.uniform_layout
    if args.skip_attention_layout_copy:
        attention_module.uniform_layout = lambda q, k, v: (q, k, v)
    if args.fuse_qkv is True and not model.dit.fuse_qkv_projections():
        raise SystemExit("the loaded Q/K/V weights do not support exact fusion")
    if args.fuse_qkv is False and hasattr(model.dit, "unfuse_qkv_projections"):
        model.dit.unfuse_qkv_projections()
    if args.compile_dit:
        model.dit = torch.compile(model.dit, fullgraph=False)
    instrumentation = SamplingInstrumentation(
        synchronize_each_step=not args.no_step_sync
    )
    instrumentation.install(model)
    timings: list[float] = []
    steady_steps: list[list[float]] = []
    output: Image.Image | None = None

    try:
        # First run includes lazy compilation and is reported separately.
        synchronize_device(device)
        t0 = time.perf_counter()
        output = _run(runtime, request)
        synchronize_device(device)
        warmup_seconds = time.perf_counter() - t0
        warmup_steps = instrumentation.step_times[:]
        instrumentation.step_times.clear()

        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.iterations):
            instrumentation.step_times.clear()
            synchronize_device(device)
            t0 = time.perf_counter()
            output = _run(runtime, request)
            synchronize_device(device)
            timings.append(time.perf_counter() - t0)
            steady_steps.append([seconds for _, seconds in instrumentation.step_times])
        steady_peak_allocated_bytes = torch.cuda.max_memory_allocated(device)

        profile_summary = None
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            instrumentation.step_times.clear()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ) as profiler:
                output = _run(runtime, request)
                synchronize_device(device)
            profile_summary = _profile_summary(profiler, args.trace)

        if args.latents_out or args.compare_latents:
            instrumentation.captured.clear()
            instrumentation.capture = True
            output = _run(runtime, request)
            synchronize_device(device)
            instrumentation.capture = False
            if args.latents_out:
                args.latents_out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(instrumentation.captured, args.latents_out)

        if args.output and output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output.save(args.output)

        image_comparison = None
        if args.compare_image and output is not None:
            baseline = Image.open(args.compare_image).convert("RGB")
            candidate = output.convert("RGB")
            if baseline.size != candidate.size:
                raise ValueError(f"image dimensions differ: {baseline.size} != {candidate.size}")
            diff = (
                np.asarray(candidate, dtype=np.float32)
                - np.asarray(baseline, dtype=np.float32)
            )
            image_comparison = {
                "max_abs_pixel": float(np.abs(diff).max()),
                "rms_pixel": float(np.square(diff).mean() ** 0.5),
                "ssim": _ssim(baseline, candidate),
            }

        latent_comparison = None
        if args.compare_latents:
            baseline = torch.load(args.compare_latents, map_location="cpu", weights_only=True)
            latent_comparison = {
                key: _tensor_metrics(baseline[key], value)
                for key, value in instrumentation.captured.items()
                if key in baseline
            }

        def per_step_median(index: int) -> float | None:
            values = [run[index] for run in steady_steps if index < len(run)]
            return statistics.median(values) if values else None

        report = {
            "device": prop.name,
            "architecture": getattr(prop, "gcnArchName", "unknown"),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "configuration": args.configuration,
            "checkpoint": args.dit,
            "kv_cache": args.kv_cache,
            "fuse_qkv": bool(model.dit.transformer_blocks) and all(
                getattr(block.attn, "to_qkv", None) is not None
                for block in model.dit.transformer_blocks
            ),
            "reference_image": str(args.reference_image) if args.reference_image else None,
            "resolution": [args.width, args.height],
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "iterations": args.iterations,
            "step_synchronization": not args.no_step_sync,
            "skip_attention_layout_copy": args.skip_attention_layout_copy,
            "model_load_seconds": load_seconds,
            "first_run_seconds_including_compile": warmup_seconds,
            "first_run_step_seconds": [seconds for _, seconds in warmup_steps],
            "steady_total_seconds": timings,
            "steady_total_median_seconds": statistics.median(timings),
            "steady_sampling_seconds": [sum(run) for run in steady_steps],
            "steady_sampling_median_seconds": statistics.median(
                sum(run) for run in steady_steps
            ),
            "steady_step_median_seconds": [
                per_step_median(i) for i in range(args.steps)
            ],
            "kv_cache_used": instrumentation.kv_cache_bytes > 0,
            "kv_cache_buffer_bytes": instrumentation.kv_cache_bytes,
            "peak_allocated_bytes": steady_peak_allocated_bytes,
            "image_comparison": image_comparison,
            "latent_comparison": latent_comparison,
            "profiler": profile_summary,
        }
        rendered = json.dumps(report, indent=2)
        print(rendered)
        if profile_summary:
            print("\nTop PyTorch profiler operations (self CUDA/HIP time):")
            print(profile_summary["top_device_ops"])
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(rendered + "\n")
    finally:
        instrumentation.restore()
        attention_module.uniform_layout = original_uniform_layout


if __name__ == "__main__":
    main()
