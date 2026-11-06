# Qwen-Image 2.1 sampling on gfx1151

This note records the sampling optimization investigation run on 2026-11-06. The
measured machine was an AMD Radeon 8060S (`gfx1151`) with PyTorch
`2.14.0+rocm10.1.0rc2` / HIP `7.16.26382`. Both Turbo INT8-ConvRot and BF16 DiT
checkpoints were measured. INT8-ConvRot is the production optimization target here;
BF16 was additionally tested for safe transfer of the cache and projection-fusion
ideas.

The optional text-to-image prefix cache is now enabled only when the loaded DiT has
the measured INT8-ConvRot projection profile and the active runtime reports gfx1151,
PyTorch 2.14.0 with ROCm 10.1 / HIP 7.16, and comfy-kitchen 0.2.40's HIP
`int8_linear` backend. Other runtimes safely use the uncached T2I path. If a
quantized LoRA uses a runtime projection branch, only T2I prefix caching is
disabled; reference editing continues using its existing cache. The measurements
below were collected on the qualified stack and remain unchanged by this
eligibility gate. They do not validate other backends or runtime-LoRA cache
equivalence.

## Measurements

The corrected A/B measurements were run on 2026-11-06. Each pair used the same
Turbo INT8-ConvRot checkpoint, prompt, seed (12345), Euler schedule, 8 steps,
guidance 1.0, batch 1, resolution, and warm prompt embeddings. Each steady number is
the median of 3 generations after load and first-run compilation. The generation
time includes VAE decode; sampling time is the sum of 8 individually synchronized
DiT calls and Euler updates, excluding prompt encoding and decode. The profiler
records model loading and first-run/compile time separately. Outputs were captured
for the same-seed numerical comparison. See the reproduction section for the exact
benchmark harness and baseline worktree procedure.

These are three separate configurations: the original HEAD with its defaults; the
optimized code with its defaults (Q/K/V fused, T2I cache off); and the optimized
code with T2I prefix caching explicitly enabled. This separates Q/K/V fusion gains
from the optional cache gain.

| Workload | Configuration | Total generation | Sampling, 8 steps | Change vs original (total / sampling) | Cache allocation |
|---|---|---:|---:|---:|---:|
| T2I 768x1024 | Original defaults | 14.715 s | 13.873 s | reference | 0 |
| T2I 768x1024 | Optimized defaults | 14.578 s | 13.736 s | 0.93% / 0.99% faster | 0 |
| T2I 768x1024 | Optimized + explicit T2I cache | 13.780 s | 12.932 s | 6.35% / 6.79% faster | 1.62 GB (1.51 GiB) |
| T2I 1024x1024 | Original defaults | 20.213 s | 19.062 s | reference | 0 |
| T2I 1024x1024 | Optimized defaults | 19.900 s | 18.749 s | 1.55% / 1.64% faster | 0 |
| T2I 1024x1024 | Optimized + explicit T2I cache | 19.035 s | 17.890 s | 5.83% / 6.15% faster | 2.16 GB (2.01 GiB) |

With the optimized default as the reference, explicitly enabling the T2I cache
improved total/sampling time by 5.48%/5.86% at 768x1024 and 4.34%/4.58% at
1024x1024. Cache is not on by default for T2I. The default-only gains above are the
Q/K/V fusion result.

For editing, both implementations used their existing prefix cache. This isolates
the Q/K/V change instead of attributing cache-on versus cache-off gains to the patch:

| Workload | Configuration | Total generation | Sampling, 8 steps | Change vs original | Cache allocation |
|---|---|---:|---:|---:|---:|
| Edit 768x1024 | Original, reference cache enabled | 18.013 s | 17.161 s | reference | 3.67 GB (3.42 GiB) |
| Edit 768x1024 | Optimized, same cache enabled | 17.731 s | 16.875 s | 1.56% / 1.66% faster | 3.67 GB (3.42 GiB) |

The edit pair used the same synthetic observatory reference image and prompt in both
runs. The cache allocation was identical. Peak allocated memory for T2I was 18.34 GB
at 768x1024 and 18.61 GB at 1024x1024; the optimized fused weights did not increase
the peak. The earlier report's cache-off versus cache-on edit comparison is not a
patch speedup and has been removed from the comparison above.

The measured model-load and first generation (including lazy compile) seconds were,
respectively, 8.17 / 23.55, 9.15 / 20.11, and 9.18 / 22.50 for original, optimized
default, and optimized-cache T2I at 768x1024; 5.12 / 28.23, 9.38 / 25.37, and
9.31 / 22.83 at 1024x1024. For edit they were 4.84 / 32.70 and 9.11 / 26.42.
Loading and cold first-run values vary with process/cache state and are reported
separately from steady sampling; they are not part of the speedup claims.

BF16 Turbo, 768x1024, was measured separately. The cache is shown here as a rejected
experiment, not as a supported T2I optimization:

| Prompt / seed | Cache | Total generation | Mean step time | Change vs uncached | Result |
|---|---:|---:|---:|---:|---|
| Detailed still life / 12345 | off | 16.188 s | 1.915 s | reference | baseline |
| Detailed still life / 12345 | on | 15.354 s | 1.815 s | 5.15% faster | rejected: final-latent RMS 0.00872; image RMS 2.05/255, SSIM 0.99635 |
| Brass observatory sign reading NORTH / 67890 | off | 16.439 s | 1.949 s | reference | baseline |
| Brass observatory sign reading NORTH / 67890 | on | 15.410 s | 1.821 s | 6.26% faster | rejected: final-latent RMS 0.0320; image RMS 5.38/255, SSIM 0.98345 |

The INT8 T2I cache adds K/V buffers for each cached conditioning branch. BF16 target
projections use a different token-row dimension after prefix removal; gfx1151's BF16
GEMM kernels then round differently.
The text-heavy BF16 case has material pixel and latent deltas, so that nominal speed
gain does not meet the quality constraint. With INT8 prefix caching, the first step
still computes the full sequence and fills K/V; subsequent steps can skip the fixed
text/reference prefix. An earlier cache-toggle edit experiment measured a 4.088 s
fill step and 1.829 s average for later cached steps, versus 4.289 s per uncached
step. That cache-toggle experiment explains the mechanism but is not used as the
Q/K/V patch speedup claim. End-to-end totals include the same decode stage in each
current A/B pair.

An earlier projection microbenchmark found the joined INT8 Q/K/V GEMM bitwise equal
and 11.6% faster at 3,072 tokens and 9.1% faster at 4,096 tokens. The end-to-end
sampling gain is smaller because this projection is only part of each block's total
INT8 GEMMs, and the split output still needs layout normalization before SDPA.

## Profile findings and experiments

The 32-block DiT, rather than the Python sampler loop, dominates. A synchronized
768x1024 optimized-default trace recorded 11,602 GPU kernel events and 26
memcpy/memset events. In that trace, `comfy_kitchen::int8_linear` accounted for
70.96% of device time (10.353 s summed over the profiled sampling pass), with the HIP
WMMA GEMM kernel itself at 68.01% (9.923 s). Fused ROCm SDPA (`attn_fwd.kd`) was
9.68% (1.413 s), `aten::copy_` was 6.90% (1.006 s), and ConvRot quantization was
2.95% (0.430 s). These percentages are from the profiler trace and are not the
unprofiled generation wall time. The INT8 GEMM remains the clearest optimization
target; MLP and residual projections contribute to the same dominant path.

The installed comfy-kitchen 0.2.40 exposes HIP WMMA, Triton, and eager INT8
implementations. I benchmarked the alternate backends directly with the actual fused
Q/K/V checkpoint weight at target token counts, synchronizing every measurement:

| Backend / input rows | HIP steady median | Candidate steady median | Candidate change | Output delta vs HIP |
|---|---:|---:|---:|---:|
| Triton, 3,072 | 8.784 ms | 11.271 ms | 28.3% slower | max abs 0.4375, RMS 0.02393 |
| Triton, 4,096 | 11.890 ms | 15.336 ms | 29.0% slower | max abs 0.3750, RMS 0.02398 |
| Eager, 3,072 | 8.933 ms | 101.671 ms | 11.4x slower | max abs 0.3750, RMS 0.02395 |

The Triton first calls also took 4.57 s and 1.52 s to compile for the two shapes.
Neither alternative matches the current HIP output bitwise, and both are slower, so
neither is retained. No full image run was needed for these rejected operator
candidates. The installed HIP extension is a compiled binary without tile-selection
controls or its kernel sources in this environment; changing its WMMA tiling or
accumulation path would require a separately maintained kernel and fresh numerical
validation. No unvalidated custom kernel is included.

I also forced comfy-kitchen's alternate tiled HIP WMMA schedule, currently selected
for gfx1150 but not gfx1151. With the actual fused Qwen Q/K/V weight, it was 3.08%
faster at 3,072 rows and 2.68% faster at 4,096 rows, with bitwise-equal projection
outputs. A full 768x1024 T2I run was 2.32% faster end to end and 2.49% faster in
sampling than the untiled optimized default, and its velocities, latents, and image
were bitwise equal to the original. However, a cache-enabled reference-edit run
diverged: final latent max absolute error 0.078125 / RMS 0.01301 / relative RMS
0.01313; decoded RGB max difference 41 / RMS 1.742 and SSIM 0.99255. The first edit
velocity already differed (RMS 0.02905). The source of this edit-only change was not
isolated, so the architecture-wide tile override is rejected and gfx1151 keeps
comfy-kitchen's default schedule. This preserves editing semantics; the T2I-only
speedup is not used.

The BF16 768x1024 trace showed the same broad bottleneck with a different GEMM path:
`aten::mm` accounted for 75.35% of device time (12.185 s in the profiled sampling
run), while fused ROCm SDPA (`attn_fwd.kd`) accounted for 8.83%. The profiled run
recorded 10,579 GPU kernel events and 26 copy/memset events. The top individual
elementwise and normalization kernels were each below 1%, so block-level elementwise
rewrites are unlikely to produce a substantial BF16 gain without changing compilation
or GEMM behavior.

| Experiment | Finding | Decision |
|---|---|---|
| Original compiled blocks, T2I, no cache | Reference timing and profiler baseline | Keep as baseline |
| Exact fixed-text-prefix K/V reuse | Current opt-in T2I comparison adds 5.48% / 4.34% total-time improvement over optimized defaults at 768x1024 / 1024x1024; cache-equivalent edit A/B is reported separately | Keep for qualified INT8; T2I opt-in, edit behavior preserved |
| BF16 fixed-text-prefix K/V reuse | 5–6% faster, but final latent relative RMS reached 2.63% and text-rendering image SSIM fell to 0.98345 | Reject for T2I; gate it off for BF16 checkpoints |
| Join the three INT8 Q/K/V projection rows | Current defaults improve total generation 0.93% / 1.55% and sampling 0.99% / 1.64% at 768x1024 / 1024x1024; cache-equivalent edit improves total 1.56% and sampling 1.66%; all captured outputs bitwise equal | Keep for qualified INT8; preserve separate Q/K/V module names and LoRA targets |
| Join BF16 Q/K/V projection rows | Same-seed outputs were bitwise equal, but 768x1024 median time increased from 16.188 s to 16.249 s (0.38% slower) | Reject for BF16; production fusion is INT8-only |
| Disable Dynamo / run eager blocks | 1024x1024 steady generation 24.896 s vs 20.070 s compiled (24.1% slower) | Reject; retain per-block compilation |
| Remove per-step device synchronization | 768x1024 total 13.701 s vs 13.678 s synchronized (within 0.2% run variation); outputs remained bitwise equal | Reject; no measurable throughput gain, retain accurate per-step progress |
| Add an outer `torch.compile` around the already compiled DiT blocks | One 768x1024 run measured 14.671 s vs 14.705 s baseline (within run noise); first velocity RMS error 0.00830 grew to final-latent RMS 0.31679, and image SSIM fell to 0.832 | Reject; no stable speedup and unacceptable numerical/image regression |
| Bypass Q/K/V L-major layout normalization | 768x1024 total 13.714 s vs 13.678 s standard (within run variation); all outputs bitwise equal | Reject; no stable gain, keep SDPA layout normalization |
| Replace SDPA backend | Existing trace selects a fused ROCm SDPA kernel; it is secondary to quantized GEMMs | No change |
| Route INT8 linear through comfy-kitchen Triton | 28–29% slower and max absolute BF16 projection delta 0.375–0.438 | Reject; not bitwise equivalent |
| Route INT8 linear through comfy-kitchen eager | 11.4x slower and max absolute BF16 projection delta 0.375 | Reject; not numerically equivalent |
| Force tiled HIP WMMA schedule on gfx1151 | 2.32% faster on 768x1024 T2I with exact outputs, but a cache-enabled edit produced final latent RMS delta 0.01301 and decoded-image SSIM 0.99255 | Reject; edit difference is unexplained |
| Change HIP WMMA tiling or add a custom block kernel | No source or tile controls are present in the installed HIP extension; would need a new kernel and independent validation | No unvalidated kernel retained |

Prefix modulation is fixed at `t=0` and causal attention prevents target tokens from
changing prefix hidden states. The first step runs the ordinary full-sequence pass
and fills K/V; later steps write target K/V at the end of the buffer and read the
unchanged prefix from its front. INT8-ConvRot matched the uncached outputs bitwise in
the measured workloads. BF16's algebraically equivalent target-only GEMMs can produce
different rounded results on gfx1151, which is why text-to-image cache is disabled
for BF16. Euler, schedule, guidance, attention precision, model weights, and block
count are unchanged.

The wider compile experiment diverged on the first DiT velocity despite identical
initial latents. Its full-latent error then amplified over the unchanged Euler schedule.
The precise source inside the wider graph was not isolated; the candidate was discarded
because it provided no repeatable speed benefit and failed the strict output comparison.

## Numerical validation

For the seed-12345 INT8 T2I comparisons at 768x1024 and 1024x1024 (original versus
optimized defaults, and original versus optimized with explicit cache), plus the
cache-equivalent INT8 reference edit, every captured denoising input, velocity, and
final latent had `max_abs = 0`, `RMS = 0`, and `relative_RMS = 0`. Decoded RGB output
had maximum pixel difference 0, RMS 0, and SSIM 1.0. The optimized-default to
optimized-cache pairs also matched the same way. Focused weight-free tests compare
cached and uncached prefix sampling with reference images and separate conditional
and unconditional branches under CFG.

The two BF16 T2I cache comparisons were not bitwise equal. On the detailed still-life
prompt, final latent delta was max 0.2031 / RMS 0.00872 / relative RMS 0.00736; image
delta was max 148 / RMS 2.05 per 8-bit channel / SSIM 0.99635. On the explicit
text-rendering prompt, final latent delta was max 0.9307 / RMS 0.0320 / relative RMS
0.0263; image delta was max 216 / RMS 5.38 / SSIM 0.98345. Both changed the rendered
lettering region, so those runs fail the quality requirement and BF16 T2I cache is
rejected. These comparisons report exact metrics rather than inferring quality from
thumbnail similarity; they are still a limited two-prompt sample, not a general
image-quality benchmark.

## Reproduction

The measurements above used a detached `e24d6f5` baseline worktree for the original
implementation, and the current worktree for the optimized implementation. Both used
the same profiler script and local checkpoint files. To recreate the original run,
make the baseline worktree and copy the current harness into it:

```bash
git worktree add --detach /tmp/thenoise-qwen-original e24d6f5
cp scripts/profile_qwen_image21.py /tmp/thenoise-qwen-original/scripts/
(cd /tmp/thenoise-qwen-original && \
  PYTHONPATH=/tmp/thenoise-qwen-original \
  /home/florian/git/thenoise/.venv/bin/python scripts/profile_qwen_image21.py \
  --dit /home/florian/git/thenoise/models/qwen_image21/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors \
  --vae /home/florian/git/thenoise/models/qwen_image21/vae/qwen_image_2.1_vae_bf16.safetensors \
  --text-encoder /home/florian/git/thenoise/models/qwen_image21/text_encoders/qwen3vl_8b_int8_convrot.safetensors \
  --prompt "An intricate still life of glass bottles on a wooden table, tiny printed labels, warm window light" \
  --width 1024 --height 1024 --steps 8 --guidance-scale 1 --seed 12345 \
  --iterations 3 --no-kv-cache --no-fuse-qkv --configuration original-default \
  --output /tmp/qwen-original.png --latents-out /tmp/qwen-original.pt \
  --report /tmp/qwen-original.json)
```

For each size, run the optimized defaults without either cache/fusion override, then
the optimized opt-in case with `--kv-cache --fuse-qkv`. Save the reference output and
latents and compare each candidate against the original. The harness reports
synchronized step times, steady total time, peak allocation, and optional profiler
trace. For example, the optimized default and opt-in commands are:

```bash
.venv/bin/python scripts/profile_qwen_image21.py \
  --dit models/qwen_image21/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors \
  --vae models/qwen_image21/vae/qwen_image_2.1_vae_bf16.safetensors \
  --text-encoder models/qwen_image21/text_encoders/qwen3vl_8b_int8_convrot.safetensors \
  --prompt "An intricate still life of glass bottles on a wooden table, tiny printed labels, warm window light" \
  --width 1024 --height 1024 --steps 8 --guidance-scale 1 --seed 12345 \
  --iterations 3 --configuration optimized-default \
  --compare-image /tmp/qwen-original.png --compare-latents /tmp/qwen-original.pt \
  --output /tmp/qwen-default.png \
  --latents-out /tmp/qwen-default.pt --trace /tmp/qwen-default-trace.json \
  --report /tmp/qwen-default.json

.venv/bin/python scripts/profile_qwen_image21.py \
  --dit models/qwen_image21/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors \
  --vae models/qwen_image21/vae/qwen_image_2.1_vae_bf16.safetensors \
  --text-encoder models/qwen_image21/text_encoders/qwen3vl_8b_int8_convrot.safetensors \
  --prompt "An intricate still life of glass bottles on a wooden table, tiny printed labels, warm window light" \
  --width 1024 --height 1024 --steps 8 --guidance-scale 1 --seed 12345 \
  --iterations 3 --kv-cache --fuse-qkv --configuration optimized-cache \
  --compare-image /tmp/qwen-original.png --compare-latents /tmp/qwen-original.pt \
  --output /tmp/qwen-cached.png --latents-out /tmp/qwen-cached.pt \
  --trace /tmp/qwen-cached-trace.json --report /tmp/qwen-cached.json
```

To run editing, add the same `--reference-image /path/to/reference.png` argument to
both implementations and compare against the original edit output. Leave `--kv-cache`
unset for this case so each implementation applies its existing edit-cache default;
do not use a cache-off edit run as the original comparison.
