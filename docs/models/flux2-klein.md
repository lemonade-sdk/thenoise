# Flux.2 Klein

**Small model, four steps, real editing.**

Flux.2 Klein is a distilled flow-matching MMDiT in two sizes (4B and 9B) that
generates in **4 steps** — and, unlike the Turbo models, it can **edit**: give
it one or more reference images plus an instruction and it returns the edited
image. The 9B is the quality pick, the 4B the speed pick; both are fast on
Strix Halo, and both have int8-convrot checkpoints.

- Text-to-image **and editing** (reference image(s) + instruction)
- Reference-latent **KV cache** for faster editing (see note below)
- Built-in defaults: **4 steps, guidance 1.0** (CFG off)

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

| Spec | Data |
|---|---|
| Architecture | Distilled flow-matching MMDiT (Flux.2 packed 128-ch latent) |
| Download size | 4B: ~12 GB · 9B: ~25 GB (bf16) |
| VAE | Flux.2 VAE (shared by both sizes) |
| Text encoder | Qwen3-4B (4B models) · Qwen3-8B (9B models) |
| Editing | ✓ |
| KV cache | ✓ (see note) |
| Default settings | 1024×1024, 4 steps, guidance 1.0 |

> **KV cache note:** the reference-latent KV cache needs a special "KV"
> checkpoint for Flux.2 Klein to be valid. Without it, editing still works
> (without `--kv-cache`).

## Variants

| Variant | Checkpoint | Steps |
|---|---|---|
| `4b` *(default)* | distilled 4B | 4 |
| `4b-base` | base 4B (CFG) | more (guidance > 1.0) |
| `9b` | distilled 9B | 4 |
| `9b-base` | base 9B (CFG) | more (guidance > 1.0) |

Base variants have no int8-convrot release.

## Lemonade recipes

Ready-made [Lemonade](https://lemonade-server.ai/docs/dev/backends-reference/#backends)
recipes for this model - each pins the checkpoints and the generation defaults:

- [Flux-2-Klein-4B-TheNoise.json](../../recipes/flux/Flux-2-Klein-4B-TheNoise.json) - 4B BF16, 4 steps
- [Flux-2-Klein-4B-INT8.json](../../recipes/flux/Flux-2-Klein-4B-INT8.json) - 4B INT8-ConvRot, 4 steps
- [Flux-2-Klein-9B-TheNoise.json](../../recipes/flux/Flux-2-Klein-9B-TheNoise.json) - 9B BF16, 4 steps
- [Flux-2-Klein-9B-INT8.json](../../recipes/flux/Flux-2-Klein-9B-INT8.json) - 9B INT8-ConvRot, 4 steps
- [Flux-2-Klein-9B-KV.json](../../recipes/flux/Flux-2-Klein-9B-KV.json) - 9B KV-cache BF16, 4 steps
- [Flux-2-Klein-9B-KV-INT8.json](../../recipes/flux/Flux-2-Klein-9B-KV-INT8.json) - 9B KV-cache INT8-ConvRot, 4 steps

## Performance (Strix Halo)

9B INT8-ConvRot, 1024×768:

- Generate @ 4 steps: **~9 s**
- Edit @ 4 steps: ~15 s

## Examples

*Generated - 9B @ 4 steps:*

<img width="45%" alt="thenoise_269346642" src="https://github.com/user-attachments/assets/7784ad89-12f6-4838-a412-508cc23ecf31" />

*Edited - instruction: "Color this image":*

| Before | After |
|---|---|
| <img width="1536" height="2048" alt="thenoise_571971704" src="https://github.com/user-attachments/assets/71c8397f-00ac-45e2-b4d0-2c391e256c40" /> | <img width="1536" height="2048" alt="thenoise_edit_3340995256" src="https://github.com/user-attachments/assets/58557f2e-6591-40b9-84ab-5521bab7f2ca" />|

## Download

```bash
# 4B distilled (default)
.venv/bin/python scripts/download.py --model klein

# 9B distilled
.venv/bin/python scripts/download.py --model klein --variant 9b

# int8-convrot DiT instead of bf16 (distilled variants only)
.venv/bin/python scripts/download.py --model klein --variant 9b --int8-convrot
```

Each fetch lands in `./models/klein/` with the matching text encoder and the
shared Flux.2 VAE.

## Usage

Generate (9B):

```bash
./thenoise.sh generate \
  --dit ./models/klein/flux-2-klein-9b.safetensors \
  --vae ./models/klein/split_files/vae/flux2-vae.safetensors \
  --text-encoder ./models/klein/split_files/text_encoders/qwen_3_8b.safetensors \
  --prompt "a red panda drinking tea" \
  --out panda.png
```

(4B paths: `split_files/diffusion_models/flux-2-klein-4b.safetensors` and
`split_files/text_encoders/qwen_3_4b.safetensors`.)

Edit (4B - `--image` is repeatable; the first image sets the output size):

```bash
./thenoise.sh edit \
  --dit ./models/klein/split_files/diffusion_models/flux-2-klein-4b.safetensors \
  --vae ./models/klein/split_files/vae/flux2-vae.safetensors \
  --text-encoder ./models/klein/split_files/text_encoders/qwen_3_4b.safetensors \
  --image fox.png \
  --prompt "a fox wearing a red scarf" \
  --out fox_edited.png
```

Serve over HTTP with the web UI (open <http://localhost:8000/>) — the Edit tab
appears because this model supports editing:

```bash
./thenoise.sh serve \
  --dit ./models/klein/flux-2-klein-9b.safetensors \
  --vae ./models/klein/split_files/vae/flux2-vae.safetensors \
  --text-encoder ./models/klein/split_files/text_encoders/qwen_3_8b.safetensors \
  --host 127.0.0.1 --port 8000
```

For every flag (size, steps, seed, multi-image references, KV cache, LoRAs,
upscaling, post-processing), see the [CLI reference](../cli.md) and the
[HTTP API reference](../api.md).
