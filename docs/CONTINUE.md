# CONTINUE — VAE performance tooling session

## Round 2 (this session) — probes are now actually runnable; one measurement bug found

`pytest tests/ -q` still 901 passed. Everything below is CPU-verified only; no GPU was run.

1. **`probes/vae_layout.py` was rewritten — it had never once run.** It died on the first
   case: `Tensor.memory_format` does not exist in torch 2.14. While fixing it the useful
   parts were added too: a per-category table underneath the headline (conv, matmul,
   norm, activation, elementwise, resample, copy, reduce — time *and* TFLOPS for each
   layout, plus Δ), so a GroupNorm that gets slower in NHWC cannot hide behind the
   wall-time number; `conv TF` is now conv FLOPs over conv time *net of* the staged
   copies; peak/reserved are measured after `reset_peak_memory_stats()` post-warmup
   (before, the channels_last arm inherited the nchw arm's peak, which would have made
   the im2col memory difference — the deployment risk — invisible); the output difference
   is printed as worst element *and* relative L2; staged kernel names are printed
   unconditionally, `--kernels` adds the full list. All 15 codec×op cases now build and
   run at 64x64 on CPU. CPU-only observations worth carrying to the GPU run: with
   `channels_last`, `wan22/decode` and `flux2/encode` come back out **nchw** (the layout
   did not survive the graph), and `flux/decode`'s norm got ~2x slower in NHWC.
2. **Fixed a real bug in the shared classifier** (`probes/vae_kernels.py`): it matched op
   names as *substrings* and its metadata-only list contained `"t"` — so `convolution`,
   `native_group_norm`, `_softmax`, `upsample_nearest2d`, `constant_pad_nd`, `cat` all
   matched, and every conv in every table was filed under "layout (free)" and thrown
   away. Categories are now whole-name tables (`FREE`, `OPS`) with unknown ops landing in
   `other`, plus `--check`, which builds all 7 codecs on CPU and prints the category of
   every op they dispatch (49 ops today, all classified) and exits 1 on any unknown.
   *Consequence*: the **category rows** of the earlier GPU tables are not trustworthy —
   conv, norm, softmax and resample time sat in "free". The kernel-name findings
   (`batched_transpose_*` 88.5 ms, `Im2d2Col_v2` 218/879 ms) and the elementwise GB/s
   numbers stand, but "conv runs at 17 TFLOPS while bmm runs at 24–30" must be re-measured
   before it is argued from.
3. **New `probes/vae_attn.py`** (the unexplained `aten::bmm` anomaly, item 5). It records
   the shape *and stride* of every q/k/v each codec hands `single_head_attention` — once
   NCHW, once `channels_last` — then replays each recorded call with nothing, one, or all
   three operands forced row-major, timing the copies separately so a copy budget cannot
   pose as a win. Two predictions from reading the code, one of which the CPU already
   confirms: NCHW `flux2` hands over `chan/chan/chan` (channel axis outermost, from
   `view(b,c,hw).transpose(1,2)`) while `flux` hands over `row/row/row` (it calls
   `.contiguous()`); the CPU also suggested that a `channels_last` arm would make every codec
   come out row-major — **that part did not survive the GPU** (see round 3).

## Round 3 — the user's GPU run: `channels_last` is not a lever on this stack

**`aten::miopen_convolution` returns NCHW whatever it is handed.** Verified on the
container's 890M in a 20-line micro-test (`nn.Conv2d(128, 256, 3)`, bf16): NCHW in /
NCHW weight → NCHW out; **NHWC in / NHWC weight → NCHW out** — with `cudnn.benchmark`
both off and on, and `Im2d2Col_v2` + `Cijk_*` chosen either way. MIOpen's NHWC path is
not exposed through ATen's conv, so a VAE cannot *stay* channels_last: the first conv
drops the format and everything after it reallocates NCHW. That accounts for the whole
run — all four codecs printed `output nchw -> nchw`, `0.00e+00` worst *and* relative-L2
difference, same `staged` ms, same kernel counts and counts-per-kernel. The two arms were
never two experiments, so "channels_last is -0.4%" is not a measurement. For the record:
`channels_last` *is* just a stride pattern (transpose + contiguous + transpose makes the
same bytes); its entire value is that it propagates through allocation, and here it
cannot. **Nothing layout-shaped went into the engine and nothing should.**

Two probe bugs found on the way, both fixed in `probes/vae_layout.py`:
* `CONV_STAGE` said `im2col`; the kernel is `Im2d2Col_v2`, so it never matched. That is
  why `staged` read `0.0` for flux/flux2/qi21 while the kernel table of the same session
  shows 218–263 ms of it (26–28% of a flux decode). The `batched_transpose_*` figure for
  qwen_image (89.4 ms, 39% of its conv) was real — that one matches `transpose`.
* `report()` now refuses to print a null result: bit-identical output *and* identical
  output layout prints "this is no experiment at all".

### What did go into the engine: one line, `thenoise/vae/flux2.py`

`swish(x) = x * torch.sigmoid(x)` → `F.silu(x)`: same function, one kernel instead of two
full read+write passes. From the user's own 1024² profile flux2 pays 31.1 ms `sigmoid` +
48.0 ms `mul`, while `flux` — identical conv geometry (583 ms of conv in both), and it
already uses `F.silu` — pays 31.7 ms. Expect ≈ 5% on both flux2 ops; the signal is
`sigmoid` disappearing from `vae_kernels.py --models flux2`'s op list. fp32 check over
[-8, 8]: max absolute difference 4.8e-07. 901 tests pass.

### The two levers the data does support

1. **`MIOPEN_FIND_MODE` sweep, zero code.** `Im2d2Col_v2` is 28% of `flux/decode`, 26% of
   `flux2/decode`, 22% of `qwen_image21/decode` — it stages a 9x-expanded copy for the
   GEMM conv path, so a find mode that picks a CK implicit-GEMM solution is pure win.
2. **Extra passes, not slow convs.**
   * The qwen-family norm is `F.normalize(x, dim=1) * scale * gamma` — a reduce over **C**
     (a strided axis in NCHW) plus up to three more full-tensor passes: `mul` 41.8 +
     `div` 20.2 + `linalg_vector_norm` 13.7 = **75.7 ms = 19% of `qwen_image/decode`**. It
     reduces over channels, so `F.rms_norm` (trailing dims) is *not* a drop-in — fusing it
     wants one kernel over an `(b*c, hw)` view, and its own probe.
   * `native_group_norm` 69.7 ms at **98 GiB/s** in both flux codecs, against the ~220
     GiB/s this machine streams at: 8.4–8.9% of the decode, at less than half the roofline.

## Round 4 — the layout problem is a GEMM-orientation problem, and tiling is the variable that is left

`pytest tests/ -q` = 902 passed (one test added, one policy test rewritten). Engine diff:
`thenoise/utils/attention.py`, `thenoise/vae/flux.py`, `thenoise/vae/qwen_image.py`.
New probes: `probes/attn_nocopy_probe.py` (orientation variants of one attention call +
copy-kernel counter), `probes/ck_conv_probe.py` (comfy_kitchen conv gate and throughput),
`probes/layout_probe.py` (conv output layout, SDPA backend availability, bmm orientation).

### Closed, with numbers, so nobody re-opens them

1. **MIOpen conv** never returns channels_last (re-confirmed), and *asking* for it is ruin:
   NHWC-in conv2d measures **1.2–1.5 TF vs 20.7–21.3 TF** for the same conv in NCHW. That is
   the `batched_transpose_*` tax, quantified.
2. **comfy_kitchen 0.2.37 is not the answer**, though it contains the only conv on this stack
   that writes channels_last: `fp16_conv3d` is a WMMA implicit conv, NDHWC in *and* out, fuses
   bias+residual, no im2col, can write into a view — and it runs **8.7–9.4 TF against MIOpen's
   20.7–21.3**, fp16-accumulate only (rel L2 3.7e-2–5.2e-2 vs an fp32-accumulate resblock), with
   a gate of spatial ≥128×128 and C,K multiples of 8 (K=3 declines, so conv_in/out falls back).
   It cannot pay for the staging it removes. `group_norm_silu_pad3d` is a genuinely good fused
   GN+SiLU+pad, but it outputs channels_last_3d *and transposes its input*.
   Its attention kernels cannot fit either: `sol_attn` head_dim must be exactly 128,
   `int8_attention` ∈ {64,128,256}, `na2d` ≤64 — VAE attention is head_dim = C (512 flux/flux2,
   768 qwen). And a split-D flash is not profitable at this shape: re-streaming K/V costs
   `(N/Br)·N·C·4` bytes ≈ 2× the whole score-matrix round trip.
3. **Fused SDPA does not exist for this shape.** On this build FLASH is runtime-disabled, cuDNN
   attention is not compiled, EFFICIENT is behind `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.
   With that env var AOTriton's `attn_fwd.kd` serves head_dim 64/128/256 (relL2 2.2e-3) and
   **returns NaN at head_dim 512 while reporting success** — that is round 1's "accepts it and
   returns wrong values", explained and closed.

### What went in

A GEMM wants a leading dimension, not an orientation. `single_head_attention` now consumes the
conv outputs exactly as they are born and **writes the answer back in that same orientation**
(`vᵀ @ pᵀ` into a (C, L) buffer) so the projection conv reads a view; the `1/√C` rides in the
score GEMM's `alpha` (`baddbmm(beta=0, alpha=…)`, applied to the fp32 accumulator), which deletes
the `div_` pass over N² *and* the extra rounding it caused; and the probabilities are written
over the scores (`aten._softmax.out` aliased), so one score matrix is live instead of two.
Codecs: `flux._Attention` lost its four `.contiguous()`, `QwenImageAttentionBlock` lost the 3C
`permute + contiguous`; flux2/wan22/mage_flow needed no edit (already views) and now exit
copy-free for free. `l_major`/`uniform_layout` are untouched — DiT side.

Verified: flux `_Attention` **bit-identical** to the old block in fp32 *and* bf16 at the test
size; qwen block max|d| 1.2e-7 with a bit-identical, view-only q; a TorchDispatchMode spy sees
**no** `copy_`/`contiguous`/`clone` in the helper or in the projection conv. 8060S attention in
isolation (C=512): **12.1 → 17.5 TF** at L=4096, **14.7 → 19.8 TF** at L=16384 (1.34–1.45×), zero
copy kernels.

Two landmines, both now fenced in code comments:
* `aten._softmax.out` with `input is out` is **exact on a contiguous buffer and garbage on a
  strided view** (max|d| 4.2 on CPU). Full tiles alias the scratch; the ragged tail gets its own
  buffer. This is a private overload — the tiling tests are what catch a change.
* `baddbmm(beta=0)` was verified not to read `C` (a NaN-filled scratch comes back finite).

### The 8060S codec diff, bucketed by whether the policy tiles the score matrix

| bucket | cases | before → after | Δ |
|---|---|---|---|
| scored whole | 25 | 20709 → 19696 ms | **−4.9%** |
| mage_flow windows | 6 | 1439 → 1432 | −0.5% |
| tiled, N=49152 (1536×2048) | 7 | 10704 → 10755 | +0.5% |
| tiled, N=65536 (2048²) | 6 | 14483 → 14916 | **+3.0%** |
| all 44 | | 47335 → 46798 | −1.1% |

By rung: 1024² **−6.0%**, 1536×2048 −1.4%, 2048² +0.1%. Peak memory moved −28%/−26% on the
`features` rows and −11.6% on the 1024² encodes — that is the aliased softmax.

**Un-tiling experiment (limit raised to 16 GiB) and why it was walked back.** Whole-matrix at
N=49152 (4.3 GiB) *fixed* the tiled regression: ming/qwen 1536×2048 encoders went from +6.3/+7.0%
to off the slower list. Whole-matrix at N=65536 (8 GiB) went the other way: +6.8% on qwen/ming
encodes, +4.7% on their decodes, while flux/flux2 improved 0.7–1.9%. So the limit is now
**5 GiB** (whole up to ~51k positions = the whole current ladder except 2048² on 8× codecs) and
the tile size is back to **256 MiB** (the 16 GiB change had silently made still-tiling sites 16×
bigger). Separately, the `(C, L)` result store is a *column-slice per tile*, so tiled now writes
token-major and the caller's existing `transpose + reshape` does one N·C copy (0.6 ms at 2048²);
the copy-free store is kept where the matrix is scored whole, which is where the −4.9% lives.
That last change is **unmeasured**.

And the number that explains flux: raising the limit put an 8 GiB score buffer into
`flux/decode@2048²` and its peak **did not move off 23.68 GiB**. So all of that 23 GiB is MIOpen
staging (the 9× `Im2d2Col` workspace — flux allocates 5× what every other 8× codec does), and
flux's 2048² timings are mostly a measure of that, not of attention.

### Next steps, live list (all on the user's 8060S)

1. **Re-run the A/B of the current tree.** Baseline snapshot is
   `bench-scripts/snapshots/vae-gfx1151-20261006-1443.json` (pre-round-4); `…-1454.json` is
   rewrite+tiled, useful for separating the two later changes:

       .venv/bin/python bench-scripts/vae_bench.py --res 1024x1024,1536x2048,2048x2048 \
           --ops decode,encode --out /tmp/vae-r4.json
       .venv/bin/python bench-scripts/diff_bench.py bench-scripts/snapshots/vae-gfx1151-20261006-1443.json /tmp/vae-r4.json

   Success = 2048² rows no longer slower, nothing lost at 1024².
2. **Sweep the tile size at the real (C, N) sites, attention only, and then fit the policy to
   it** — the two experiments so far moved the limit and the tile size together, which is why the
   conclusion is still soft. Rows ∈ {whole, 32768, 16384, 8192, 4096, 2048, 1024, 256} at
   (C=512, N=65536) and (C=768, N=65536): monkeypatch `attn.SCORE_TILE_BYTES`, or pass `rows=`
   straight into `single_head_attention` with synthetic operands (that is all
   `probes/attn_nocopy_probe.py` does; it needs argv sizes).
3. **Get the real attention geometry** — we still do not have every site's (C, N) per codec/op:
   `.venv/bin/python bench-scripts/probes/vae_attn.py --nchw-only --models qwen_image,ming_image,flux --res 2048x2048`
   (it records shape *and* stride of every q/k/v handed over; `--nchw-only` halves the run).
4. **flux's 23.68 GiB / `Im2d2Col_v2` is now the biggest thing on the table**, and it is MIOpen's,
   not ours: `MIOPEN_DEBUG_CONV_GEMM=0` (the im2col+GEMM solver that runs `Im2d2Col_v2`),
   `MIOPEN_DEBUG_FIND_ONLY_SOLVER=<ConvHipImplicitGemmGroupFwdXdlops|ConvHipIgemmGroupFwdXdlops>`,
   `MIOPEN_DEBUG_CONV_IMPLICIT_GEMM_ASM_FWD_GTC_XDLOPS_NHWC=0` (the `_NHWC` solvers are the ones
   needing the transposes — that is qwen's 88.5 ms), winograd via
   `MIOPEN_DEBUG_AMD_MP_BD_WINOGRAD_F3X3` / `MIOPEN_DEBUG_AMD_WINOGRAD_MPASS_F3X3`, plus
   `MIOPEN_FIND_MODE`. Names were read out of `_rocm_sdk_libraries/lib/libMIOpen.so.1`. Success =
   the kernel gone from `vae_kernels.py --kernels`, and the peak down.
5. Still owed from round 3 and unchanged: the qwen-family C-axis RMS norm (19% of that decode, 3
   passes) and `native_group_norm` at 98 GiB/s. Both move pixels; both need a probe first.

## What was built (all committed-free, working tree only; `pytest tests/ -q` = 902 passed)

1. **`bench-scripts/vae_bench.py`** (new) — a "camera" for VAE cost, in the same style as
   `block_bench.py`: seeded random weights, no checkpoints, no A/B switches, table + JSON
   snapshot. Registry of 7 production codecs (`qwen_image`, `ming_image`, `qwen_image21`,
   `wan22`, `flux`, `flux2`, `mage_flow`) × ops (`encode`, `decode`, `features`) × a pixel
   ladder (default `1024x1024,1536x2048,2048x2048`, `WxH`, snapped up to each codec's
   compression). 45 cases. Columns: latent, ms, ±%, Mpx/s, TFLOPS, GiB allocated, GiB
   reserved, `1st` (untimed first call). FLOPs are *measured* on the untimed first call
   with `torch.utils.flop_counter.FlopCounterMode`. Defaults `--iters 1 --repeats 3
   --warmup 1`, `--dtype bf16`. Fixtures use the loaders' own constants
   (`_WAN21_FAMILY_ARCH`; QI21 = `dim=96, dec_dim=144, z_dim=64, dim_mult=[1,2,4,8,8],
   temperal=[F,T,T,T], image_channels=4, patch_size=1`, read off the checkpoint header).
   Sesqui/upscale nets deliberately excluded; `decode_features` (QI21/Wan22 transcoder
   path) deliberately included.
2. **`bench-scripts/diff_bench.py`** (edited) — now accepts `tool: "vae_bench"` as well as
   `block_bench`, refuses a cross-tool diff, and tolerates a case whose flop count failed
   (`tflops: null`).
3. **`bench-scripts/probes/vae_kernels.py`** (new) — per-op cost of one case: kineto
   profiler for time per aten op + a `TorchDispatchMode` for bytes touched +
   `FlopCounterMode` for FLOPs, joined **per category** (op names disagree between the
   three instruments, categories don't); also spies on `score_tile_rows` to print whether
   attention ran whole or TILED. Joins had to be normalised with `op_key()`
   (`aten.convolution.default` → `aten::convolution`), and metadata-only layout ops are
   excluded from bytes. Verified by CPU smoke run only — **the GPU run is the user's**.
4. **`bench-scripts/probes/vae_layout.py`** (new, **NOT YET RUN AT ALL**) — runs the same
   case in NCHW vs `channels_last` and prints ms, conv TFLOPS, "staging" kernel time
   (transpose/im2col/col2im/copy by kernel-name regex), peak/reserved, and the max
   relative output difference. Wraps the bench's `new_vae`/`rand_pixels`/`randn` to apply
   the memory format. Intended to be re-run under different `MIOPEN_FIND_MODE` (env var is
   read at ROCm-lib load, so it must be set on the command line, not in-process).

## Key findings from the user's GPU data (Strix Halo / 8060S, bf16, batch 1)

* Cross-codec `Mpx/s` is architecture, not speed: analytic work ranges 0.35 TFLOP
  (mage_flow/encode) to 17 TFLOP (qwen_image21/decode) per 1024² image — 48×. My analytic
  decomposition reproduced the bench's `TFLOPS × ms` to ~1% (qwen dec 4.71 vs 4.70).
* Decode > encode in every codec: 3 resblocks/level vs 2, decoder *ends* at full
  resolution, and the nearest-neighbour upsampler's conv runs at the **finer** resolution
  (~25% of the Flux AE decoder's FLOPs).
* `qwen_image21/decode` (17 TFLOP, 1.25 s @1024²) vs `wan22/decode` (10 TFLOP, 0.60 s) is
  a codec-shape difference: QI21 has 5 conv stages so its decoder runs convs at /1 and /2,
  while Wan 2.2's patchify-2 hides the last factor of 2 in a shuffle (no full-res convs).
* Flux-family decoders ≈ qi21/2: SD-style 8× AE, base 128 (⇒ (128/96)² = 1.78× the Qwen
  VAE, which matches flux2/enc vs qwen/enc exactly).
* Roofline fit over the 1024² bench rows: ~17 TFLOPS + ~154 GB/s explains every conv codec
  within ±20%; the `features` rows (21.5 / 27.9 TFLOPS) prove convs can reach ~28 TFLOPS,
  and `mage_flow/decode` (3.2 TFLOPS) is kernel-count bound, not FLOP/bandwidth bound.
* **The profiler results (1024² and 2048², decode) showed "layout + elementwise" — the
  layout half is dead (round 3: the conv will not take NHWC). What survives of this
  evidence is the elementwise mass and the `Im2d2Col` staging:**
  - conv category runs at 17 TFLOPS while `aten::bmm` in the same case runs at 24–30.
  - `qwen_image/decode` @1024²: 250.8 ms "conv" contains **88.5 ms of
    `batched_transpose_*`** (MIOpen NCHW↔NHWC churn) ⇒ ~35% of conv time, 21% of the
    decode; strip it and convs are ~26 TFLOPS, i.e. kernel quality is fine.
  - `flux/decode` + `flux2/decode`: **`Im2d2Col_v2.kd` = 218 ms of 781 ms @1024² (28%) and
    879 ms of 4263 @2048² (21%)** — MIOpen staging a 9×-expanded im2col copy for GEMM-path
    convs. This is also why flux conv TFLOPS *falls* 17.0 → 13.1 with resolution and why
    peak jumps 6.05 → 23.68 GiB allocated / 38.5 reserved @2048² (deployment risk on
    smaller cards).
  - 25–33% of every decode is unfused elementwise at 165–224 GiB/s (the machine's effective
    streaming ceiling): `mul`/`div`/`add_`/`silu`/`linalg_vector_norm`. Qwen's VAE norm is
    RMS-style (`linalg_vector_norm` + `div` + `mul`, 3 kernels) where Flux uses fused
    `native_group_norm`; `flux2.swish` is `x * sigmoid(x)` (2 kernels + mul) where the other
    codecs use `F.silu`.
  - Prediction confirmed: `score_tile_rows` tiles only when `4n² > 2 GiB`, so at 2048² the
    8× codecs (`qwen_image`, `ming_image`, `flux`, `flux2`) print **TILED, rows=1024** and
    their attention share goes 4.5% → 19% of the decode, while `qwen_image21`/`wan22`
    (attention at /16, N=16384) stay whole.

## Next steps, in priority order — SUPERSEDED by the round 4 list above (kept for the reasoning)

1. **`MIOPEN_FIND_MODE` sweep — zero code, biggest single number on the table.**
   `Im2d2Col_v2` is 28% of `flux/decode`, 26% of `flux2/decode`, 22% of
   `qwen_image21/decode`, and it only *stages* a 9x-expanded copy for a GEMM conv path. If
   a find mode picks a CK implicit-GEMM solution instead, that fraction is simply gone:

       .venv/bin/python bench-scripts/vae_bench.py --models flux,flux2,qwen_image21 --ops decode --out snap-fast.json
       MIOPEN_FIND_MODE=1 .venv/bin/python bench-scripts/vae_bench.py --models flux,flux2,qwen_image21 --ops decode --out snap-normal.json
       MIOPEN_FIND_MODE=3 .venv/bin/python bench-scripts/vae_bench.py --models flux,flux2,qwen_image21 --ops decode --out snap-hybrid.json
       .venv/bin/python bench-scripts/diff_bench.py snap-fast.json snap-normal.json

   Read `1st` too: a real find costs cold-shape seconds, which is a server-side cost of its
   own (`MIOPEN_FIND_MAX_FINDS_PER_SEARCH` is the knob). Success = `Im2d2Col_v2` gone from
   `vae_kernels.py --models flux2 --kernels`, not just ms moving.
2. **The qwen-family norm: 19% of `qwen_image/decode` is 3 passes over the pixels.**
   `QwenImageRMS_norm.forward` is `F.normalize(x, dim=1) * self.scale * self.gamma`, which
   is `linalg_vector_norm` 13.7 + `div` 20.2 + `mul` 41.8 = 75.7 ms of a 394 ms decode, all
   at 166–223 GiB/s (the machine's streaming ceiling) and the reduce is over **C**, a
   strided axis in NCHW. It is not `F.rms_norm` (that normalises trailing dims), so the
   fusion is either one kernel over an `(b*c, hw)` view with the gamma folded into the
   scale, or `torch.compile` of that one module — and compiled-per-resolution needs
   measuring before it is trusted, because requests change shape every time. `ming_image`
   and `wan22`/`qwen_image21` have the same shape of norm (`mul` 79.9 + `div` 34.5 +
   `linalg_vector_norm` 26.9 = 141 ms of `qwen_image21/decode`) so one fix pays four
   codecs. Ask the user first: it moves pixels.
3. **`native_group_norm` runs at 98 GiB/s** in both flux codecs — 69.7 ms, 8.4–8.9% of the
   decode, under half the roofline the same case's `silu` reaches (215 GiB/s). Worth a
   probe of `nn.GroupNorm` vs `F.group_norm` vs norm-over-`(b,c,hw)` before concluding it
   is a MIOpen/at::native limit.
4. **The attention operand anomaly is real but small** — `vae_attn.py` on the GPU: NCHW
   `flux2` hands over `chan/chan/chan` and pays 37.8 ms / 14.6 TF, `flux` hands over
   `row/row/row` and pays 32.5 ms / 16.9 TF for the same N=16384, C=512 call; forcing all
   three row-major is +14% of the call, +9.2% net of the 1.8 ms of copies. That is 5 ms of
   an 835 ms decode (**0.6%**) — a `.contiguous()` in `flux2._AttnBlock.attention`, or the
   `chan` column staying `chan` in the `channels_last` arm, which is another reminder that
   the arm did nothing. Take it after the two above, not before.
5. **Re-run the op profile** now that the classifier is fixed — the conv/norm/softmax rows
   of the earlier tables were filed as "free" and are worthless:

       .venv/bin/python bench-scripts/probes/vae_kernels.py --res 1024x1024 --kernels
       .venv/bin/python bench-scripts/probes/vae_kernels.py --res 2048x2048 --models flux,flux2,qwen_image --kernels

   (`--check` re-verifies the classifier itself, on CPU, in ~20 s: it prints the category
   of every op the seven codecs dispatch and exits 1 if any is unclassified.)
6. **Do not re-open layout.** `vae_layout.py`/`vae_attn.py` still have `channels_last` arms
   because they are the instrument that proved the conv refuses it; on this stack they can
   only ever print the "no experiment at all" warning. If MIOpen ever grows an NHWC conv
   path in ATen, the probes are ready and nothing else needs writing.
7. Still owed: the 1536×2048 rung of `vae_bench` (the tiling switch should show up as a
   TFLOPS dip for `qwen_image`/`ming_image`/`flux`/`flux2`), and `--ops encode,features`
   profiles.

## Constraints to remember

Never `uv sync`; never run the server/generate/load real weights from here — build probes
and ask the user to run them (the dev container does expose a gfx1150/890M and a CPU
only good for smoke runs; measurements belong to the 8060S box). Run tests with
`.venv/bin/python -m pytest tests/ -q`; smoke probes with `--device cpu --res 64x64
--groups 1`; do not inspect `bench-scripts/probes/__pycache__` (the user deleted those
probe sources on purpose); ask rather than guess when unsure.

A GEMM does not want a particular orientation, it wants a leading dimension: before copying a
conv output to make a matmul happy, check whether the matmul can take it as it is and *write its
result transposed*. It nearly always can, and the copy is the whole cost.

`aten._softmax.out` with `input is out` is exact on a contiguous buffer and silently wrong on a
strided view. Alias only whole buffers. `baddbmm(beta=0)` does not read `C`, verified.

Do not derive the score-tiling policy; measure it per (C, N). Two rounds of deriving it got it
wrong, and one round of moving the limit and the tile size together got two opposite answers.
`flux/decode@2048²` is im2col-dominated (23.68 GiB peak, unmoved by an 8 GiB score buffer): read
any attention result there with suspicion.

Matching aten ops by substring is not classification — `"t" in "convolution"` cost this
session a whole round of measurements. Use the whole-name tables in `vae_kernels.py` and
run `--check` after touching them.
