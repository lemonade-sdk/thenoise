# Krea 2

**The quality anchor.**

Krea 2 is a large Turbo MMDiT — the reference for image quality in TheNoise's
model lineup. It trades speed for detail: ~19–26 seconds per image on Strix
Halo depending on quantization, and a ~35 GB download, but the results are the
reason people run TheNoise at all.

- Text-to-image (Turbo and RAW checkpoints)
- Built-in defaults: **8 steps, guidance 1.0** (CFG off)
- int8-convrot variant available (faster generation, lower memory)

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

| | |
|---|---|
| Architecture | Turbo MMDiT (flow matching) |
| Download size | ~35 GB (Turbo + VAE + text encoder, bf16) |
| VAE | Qwen-Image VAE |
| Text encoder | Qwen3-VL-4B |
| Editing | — |
| KV cache | — |
| Default settings | 1024×1024, 8 steps, guidance 1.0 |

## Checkpoints

| Checkpoint | Notes |
|---|---|
| Turbo *(default)* | distilled, 8 steps |
| RAW (`--include-raw`) | non-distilled, for research/tuning |

## Lemonade recipes

Ready-made [Lemonade](https://lemonade-server.ai/docs/dev/backends-reference/#backends)
recipes for this model - each pins the checkpoints and the generation defaults:

- [Krea-2-Turbo-INT8.json](../../recipes/krea2/Krea-2-Turbo-INT8.json) - Turbo INT8-ConvRot, 8 steps
- [Krea-2-better-VAE.json](../../recipes/krea2/Krea-2-better-VAE.json) - Turbo BF16, 8 steps

## Performance (Strix Halo)

- Turbo BF16 @ 8 steps: ~26 s (1024×768)
- Turbo INT8-ConvRot @ 8 steps: **~19 s**

## Examples

*Generated:*

<img width="45%" alt="thenoise_1070888462" src="https://github.com/user-attachments/assets/25afa568-8375-494a-8364-37e20421ded6" />
<img width="45%" alt="thenoise_1049596729" src="https://github.com/user-attachments/assets/fc8c175a-f625-4b1c-a90e-882a0d7c4161" />

## Download

```bash
.venv/bin/python scripts/download.py --model krea2
```

Fetches the bf16 Turbo DiT (~26 GB), the VAE, and the Qwen3-VL text encoder
(~8.9 GB). Options:

```bash
# also fetch the RAW (non-turbo) DiT (another ~26 GB)
.venv/bin/python scripts/download.py --model krea2 --include-raw

# int8-convrot DiT(s) instead of bf16
.venv/bin/python scripts/download.py --model krea2 --int8-convrot
```

## Usage

Generate:

```bash
./thenoise.sh generate \
  --dit ./models/krea2/diffusion_models/krea2_turbo_bf16.safetensors \
  --vae ./models/krea2/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/krea2/text_encoders/qwen3vl_4b_bf16.safetensors \
  --prompt "a lighthouse on a cliff at dusk, storm clouds" \
  --out lighthouse.png
```

Serve over HTTP with the web UI (open <http://localhost:8000/>):

```bash
./thenoise.sh serve \
  --dit ./models/krea2/diffusion_models/krea2_turbo_bf16.safetensors \
  --vae ./models/krea2/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/krea2/text_encoders/qwen3vl_4b_bf16.safetensors \
  --host 127.0.0.1 --port 8000
```

For every flag (size, steps, seed, LoRAs, upscaling, post-processing), see the
[CLI reference](../cli.md) and the [HTTP API reference](../api.md).
