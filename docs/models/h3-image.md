# H3-Image

**MiniMax's 20 B model: prompt, references and the picture being made, all in one sequence.**

H3-Image has no cross-attention and no separate image stream. `condition_proj` pulls the
language model's output into the DiT's width, and from there the prompt's tokens, the
reference images' patches and the noisy target's patches are one sequence that all 50
blocks read. References are told apart from the target by where they sit on the position
grid — one time unit earlier — and by a modulation tag, not by a different projection.

Two things about it are not the usual shape of a diffusion transformer:

**Time is a curve, not an MLP.** The checkpoint carries a 1025×8 table of modulation
factors; a step's sigma is read off it by interpolation and each block projects that row
into its own shift/scale/gate set. Because the table is FP32 and is read *between* rows,
the engine keeps it FP32 whatever the compute dtype is — rounding it would move the
conditioning of every block, not just one tensor.

**The output head is a bank, not a head.** `final_layer` holds one set of output weights
per interval of a distilled sigma grid, and a step that spans more than one interval gets
their dt-weighted mean. The grid is in the checkpoint (`pdd_sigmas`, 6 intervals), so the
schedule walks the grid's *index* rather than its sigma values: ask for 6 steps and you
step on the grid itself, ask for 4 and each step averages the intervals it covers.

The conditioner is **Qwen3-VL-32B cut to its first 50 layers**, and the cut took the final
RMSNorm with it — the file has no `norm.weight`, and the DiT expects the raw output of the
last surviving layer. An RMSNorm would rescale every token to a fixed magnitude, and the
prompt's magnitude carries information the DiT reads, so the norm is an identity rather
than a weight that happens to be all ones. What the conditioner is fed is also bare: the
prompt with no system turn and no user turn around it, and each reference spliced in as
`Picture 1: ` followed by its vision blocks, in the order the prompt refers to them.

- Text-to-image **and editing** (one or more reference images + instruction)
- Built-in defaults: **2048×2048, 6 steps, guidance 1.0, euler** — guidance is distilled
  in, so a negative prompt is an extrapolation off the training distribution
- 20.1 B DiT: 21 GB int8-convrot, 40 GB bf16 — the largest model here by a distance
- Its own codec: 24-channel latent at 16 px per cell, with a ViT decoder
- Tokenizer and image preprocessor are vendored in this repository; nothing is fetched

## Specs

| Spec | Data |
|---|---|
| Architecture | packed single-stream DiT (flow matching), 50 blocks + a 2-block token refiner, dim 5376, 56 heads × 128 (attention inner 7168 — wider than the trunk), SwiGLU 28672 |
| Parameters | 20.1 B |
| Download size | ~49.3 GB int8-convrot (21 GB DiT + 27.1 GB text encoder + 1.2 GB VAE); ~68.5 GB with the bf16 DiT |
| VAE | H3-Image AE, 24-ch latent at 16 px per cell |
| Sizes | any size you ask for, rounded up to a multiple of 32 px |
| Sequence | a 2048×2048 picture is 4096 target rows, plus the prompt and each reference's patches |
| Text encoder | Qwen3-VL 32B truncated to 50 layers, output unnormalized |
| Editing | ✓ |
| KV cache | — (batch-1, guidance distilled) |
| Default settings | 2048×2048, 6 steps, guidance 1.0, euler |

## Checkpoints

| DiT precision | File |
|---|---|
| bf16 *(default)* | `minimax_h3_image_pruned_bf16.safetensors` |
| int8-convrot | `minimax_h3_image_pruned_int8_convrot.safetensors` |

The repo also publishes a `w4a8` DiT and an `nvfp4-awq` text encoder. Neither quantization
is one this engine dequantizes, so both are left alone; the text encoder's int8-convrot
file is fetched either way, since it is the only one of its three that runs here.

A `w4a8` file will not resolve to this model — detection requires the packed DiT's
`video_patch_proj` and `condition_proj` and refuses anything carrying an
`audio_patch_proj`, which is how the H3 *video* checkpoint is recognizable. Video files are
not supported: this engine has no audio stream to give them.

## Lemonade recipes

Ready-made [Lemonade](https://lemonade-server.ai/docs/dev/backends-reference/#backends)
recipes, each pinning the three checkpoints and the 6-step 2048×2048 defaults:

- [H3-Image-INT8.json](../../recipes/h3image/H3-Image-INT8.json) - int8-convrot DiT, 49 GB
- [H3-Image.json](../../recipes/h3image/H3-Image.json) - bf16 DiT, 69 GB

## Download

```bash
# bf16 DiT (40 GB) + the int8-convrot Qwen3-VL-32B conditioner + the AE
.venv/bin/python scripts/download.py --model h3-image

# int8-convrot DiT (21 GB instead of 40)
.venv/bin/python scripts/download.py --model h3-image --int8-convrot
```

The three files land under `./models/h3-image/` in `diffusion_models/`, `text_encoders/`
and `vae/`. Plan the download without fetching with `--dry-run`.

Expect roughly 48 GB resident with the int8 pair — this is the model on which 128 GB of
unified RAM earns its keep. If you are short, run the DiT int8 and nothing else changes.

## Usage

Generate:

```bash
./thenoise.sh generate \
  --dit ./models/h3-image/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors \
  --vae ./models/h3-image/vae/minimax_h3_image_vae_fp16.safetensors \
  --text-encoder ./models/h3-image/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --prompt "a half-eaten croissant on a ceramic plate, window light" \
  --out croissant.png
```

Edit. References are seen twice, both times aspect-preserved and on the codec's 32 px grid:
the conditioner gets one whose short edge is exactly 1024 px, the DiT one whose short edge
lands between 1024 and 2048 px. The prompt refers to them as `Picture 1`, `Picture 2`, … in
the order they are passed:

```bash
./thenoise.sh edit \
  --dit ./models/h3-image/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors \
  --vae ./models/h3-image/vae/minimax_h3_image_vae_fp16.safetensors \
  --text-encoder ./models/h3-image/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --image /tmp/fox.png \
  --prompt "put the fox in snow, keep its pose" --out /tmp/fox_snow.png
```

Step counts other than 6 are supported and do the interval averaging described above;
`--steps 6` is what the head bank was distilled for. Guidance above 1.0 costs a second
forward per step — the unconditional branch carries the same pictures, since they are the
edit's subject rather than something to negate.

`--upscale` on this model is a codec round trip (decode, resize, re-encode): no latent
upscaler has been trained for a 24-channel codec, so there is nothing weight-based to offer
instead.

## Notes

- The modulation curve, the patch projection and the head bank are FP32 in the file and
  stay FP32 when the rest of the model moves to BF16. The per-block modulation projections
  are the odd ones out — FP16 weights, BF16 biases — and both are read in FP32, which is
  how they were exported to be used.
- The three rope axes (time, height, width) share one 16-entry frequency vector, so 96 of
  each head's 128 dimensions rotate and the rest pass through.
- A reference is fed to the DiT 99.9% signal — its own noise is drawn from a stream that
  restarts for each reference, so a second reference does not inherit the first one's draws.
