# Ming-Image 0.1

**The Lumina/S3-DiT family, with real transparency.**

Ming-Image is an S3-DiT — the same transformer family as
[Z-Image](zimage.md) — conditioned by a condensed BailingMM2 ("Ling-mini-2.0")
mixture-of-experts encoder. Its VAE draws **RGBA**, so a "transparent background"
prompt produces a real alpha channel rather than a checkerboard.

- Text-to-image only (no editing)
- Built-in defaults: **12 steps, guidance 1.0** (CFG off)
- int8-convrot variants for the DiT *and* the text encoder
- The tokenizer ships inside the text-encoder checkpoint (nothing to download)

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

| | |
|---|---|
| Architecture | S3-DiT (flow matching), 30 blocks + 2 refiners, dim 3840 |
| Download size | ~49 GB bf16 (12 GB DiT + 35 GB text encoder + 0.25 GB VAE); ~15 GB int8-convrot |
| VAE | Wan2.1-family (Qwen-Image layout), **RGBA**, 16ch latent at 8× |
| Text encoder | BailingMM2 "Ling-mini-2.0" (16 B MoE thinker + 28-layer connector) |
| Editing | — |
| KV cache | — |
| Default settings | 1024×1024, 12 steps, guidance 1.0, euler sampler |

## Checkpoints

| Checkpoint | File |
|---|---|
| `--variant design` *(default)* | `ming_image_0.1_design_{bf16,int8_convrot}.safetensors` |

## Performance (Strix Halo)

Not measured yet.

## Examples

<img width="30%" alt="thenoise_3611708399" src="https://github.com/user-attachments/assets/7901b2eb-5c25-4524-9a43-7ed5f447d796" />
<img width="60%" alt="thenoise_71" src="https://github.com/user-attachments/assets/9e6d7ded-d8e6-43e1-b49b-b4deb7465247" /><img width="70%" alt="thenoise_1 (1)" src="https://github.com/user-attachments/assets/41feaff3-1097-4fb6-a41b-502faf476fe2" />


## Download

```bash
# bf16 (default)
.venv/bin/python scripts/download.py --model ming-image

# int8-convrot DiT + text encoder
.venv/bin/python scripts/download.py --model ming-image --int8-convrot
```

## Usage

Generate:

```bash
./thenoise.sh generate \
  --dit ./models/ming_image/diffusion_models/ming_image_0.1_design_bf16.safetensors \
  --vae ./models/ming_image/vae/ming_image_vae_bf16.safetensors \
  --text-encoder ./models/ming_image/text_encoders/ming_image_0.1_ling_mini_2.0_bf16.safetensors \
  --prompt "带透明通道，4通道RGBA图像. A glass flask of fireflies" \
  --out flask.png
```

Serve over HTTP with the web UI (open <http://localhost:8000/>):

```bash
./thenoise.sh serve \
  --dit ./models/ming_image/diffusion_models/ming_image_0.1_design_bf16.safetensors \
  --vae ./models/ming_image/vae/ming_image_vae_bf16.safetensors \
  --text-encoder ./models/ming_image/text_encoders/ming_image_0.1_ling_mini_2.0_bf16.safetensors \
  --host 127.0.0.1 --port 8000
```

For every flag (size, steps, seed, LoRAs, upscaling, post-processing), see the
[CLI reference](../cli.md) and the [HTTP API reference](../api.md).
