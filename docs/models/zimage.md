# Z-Image

**8-step quality without the bloat.**

Z-Image is an S3-DiT with a Qwen3-4B caption encoder. The distilled
**Z-Image-Turbo** checkpoint generates in 8 steps with CFG off and lands between
Anima and Krea 2 in both speed and detail. The non-distilled **Z-Image** base
checkpoint uses the same engine and the same VAE/text encoder, but wants more
steps.

- Text-to-image only (no editing)
- Turbo: built-in defaults **8 steps, guidance 1.0** (CFG off)
- Base: pass `--steps` / `--guidance-scale` explicitly (see below)
- int8-convrot variant available for both

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

|---|---|
| Architecture | S3-DiT (flow matching) |
| Download size | ~21 GB (bf16 DiT + Flux VAE + Qwen3-4B) |
| VAE | Flux VAE |
| Text encoder | Qwen3-4B (caption encoder) |
| Editing | — |
| KV cache | — |
| Default settings | 1024×1024, 8 steps, guidance 1.0, euler sampler |

## Checkpoints

| Checkpoint | Repo | Steps |
|---|---|---|
| `--variant turbo` *(default)* | `Comfy-Org/z_image_turbo` | 8 (built-in default) |
| `--variant base` | `Comfy-Org/z_image` | your choice (not distilled) |

For the **base** checkpoint the engine's built-in defaults target Turbo, so pass
explicit `--steps` and `--guidance-scale` on every run; the upstream
[Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) page documents the
reference settings.

## Lemonade recipes

Ready-made [Lemonade](https://lemonade-server.ai/docs/dev/backends-reference/#backends)
recipes for this model - each pins the checkpoints and the generation defaults:

- [Z-Image-Turbo.json](../../recipes/zimage/Z-Image-Turbo.json) - Turbo BF16, 8 steps
- [Z-Image-Turbo-INT8.json](../../recipes/zimage/Z-Image-Turbo-INT8.json) - Turbo INT8-ConvRot, 8 steps
- [Z-Image.json](../../recipes/zimage/Z-Image.json) - Base BF16, 28 steps, CFG 3
- [Z-Image-INT8.json](../../recipes/zimage/Z-Image-INT8.json) - Base INT8-ConvRot, 28 steps, CFG 3

## Performance (Strix Halo)

- Turbo @ 8 steps: **~10 s** (1024×768)

## Examples

*Generated - Turbo:*

<img width="45%" alt="thenoise_1484041240" src="https://github.com/user-attachments/assets/ffc2ef78-51d3-415b-a263-372353c9eb76" />
<img width="45%" alt="thenoise_3826314304" src="https://github.com/user-attachments/assets/00298384-47a0-469e-bea7-fb7a809a1830" />

## Download

```bash
# Turbo (default)
.venv/bin/python scripts/download.py --model zimage

# non-distilled base
.venv/bin/python scripts/download.py --model zimage --variant base

# int8-convrot DiT instead of bf16 (either variant)
.venv/bin/python scripts/download.py --model zimage --int8-convrot
```

Each fetch lands in `./models/zimage/` with the DiT under
`split_files/diffusion_models/`; the Flux VAE and Qwen3-4B encoder are shared
between both variants.

## Usage

Generate (Turbo):

```bash
./thenoise.sh generate \
  --dit ./models/zimage/split_files/diffusion_models/z_image_turbo_bf16.safetensors \
  --vae ./models/zimage/split_files/vae/ae.safetensors \
  --text-encoder ./models/zimage/split_files/text_encoders/qwen_3_4b.safetensors \
  --prompt "a snowy mountain village at night" \
  --out village.png
```

Generate (base — explicit schedule):

```bash
./thenoise.sh generate \
  --dit ./models/zimage/split_files/diffusion_models/z_image_bf16.safetensors \
  --vae ./models/zimage/split_files/vae/ae.safetensors \
  --text-encoder ./models/zimage/split_files/text_encoders/qwen_3_4b.safetensors \
  --prompt "a snowy mountain village at night" \
  --steps 28 --guidance-scale 3.0 \
  --out village_base.png
```

Serve over HTTP with the web UI (open <http://localhost:8000/>):

```bash
./thenoise.sh serve \
  --dit ./models/zimage/split_files/diffusion_models/z_image_turbo_bf16.safetensors \
  --vae ./models/zimage/split_files/vae/ae.safetensors \
  --text-encoder ./models/zimage/split_files/text_encoders/qwen_3_4b.safetensors \
  --host 127.0.0.1 --port 8000
```

For every flag (size, steps, seed, LoRAs, upscaling, post-processing), see the
[CLI reference](../cli.md) and the [HTTP API reference](../api.md).
