# H3-Image

**MiniMax's 20 B model: text-to-image and editing from one checkpoint.**

The biggest model this engine runs, and the one that needs the most memory to show for
it. It generates at 2048×2048 by default and edits one or more reference images from an
instruction; in the prompt, refer to them as `Picture 1`, `Picture 2`, … in the order
they are passed.

- Text-to-image **and editing** (one or more reference images + instruction)
- Reference-latent **KV cache** — automatic for edits
- Built-in defaults: **2048×2048, 4 steps, guidance 1.0, euler** — guidance is distilled
  in, so a negative prompt buys nothing
- 21 GB int8-convrot DiT, 40 GB bf16; roughly 48 GB resident with the int8 pair
- Tokenizer and image preprocessor are vendored in this repository; nothing is fetched

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

| Spec | Data |
|---|---|
| Parameters | 20.1 B |
| Download size | ~49.3 GB int8-convrot (21 GB DiT + 27.1 GB text encoder + 1.2 GB VAE); ~68.5 GB with the bf16 DiT |
| VAE | H3-Image AE, 24-ch latent at 16 px per cell |
| Sizes | any size you ask for, rounded up to a multiple of 32 px |
| Text encoder | Qwen3-VL 32B |
| Editing | ✓ |
| KV cache | ✓ (references, automatic for edits) |
| Default settings | 2048×2048, 4 steps, guidance 1.0, euler |

## Lemonade recipes

Ready-made [Lemonade](https://lemonade-server.ai/docs/dev/backends-reference/#backends)
recipes, each pinning the three checkpoints and the defaults:

- [H3-Image-INT8.json](../../recipes/h3image/H3-Image-INT8.json) - int8-convrot DiT, 49 GB
- [H3-Image.json](../../recipes/h3image/H3-Image.json) - bf16 DiT, 69 GB

## Performance (Strix Halo)

INT8-ConvRot, 768x1024:

- Generate @ 4 steps: **~4.1 s**

## Download

```bash
# bf16 DiT (40 GB) + the int8-convrot Qwen3-VL-32B text encoder + the VAE
.venv/bin/python scripts/download.py --model h3-image

# int8-convrot DiT (21 GB instead of 40)
.venv/bin/python scripts/download.py --model h3-image --int8-convrot
```

The three files land under `./models/h3-image/` in `diffusion_models/`, `text_encoders/`
and `vae/`. Plan the download without fetching with `--dry-run`.

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

Edit:

```bash
./thenoise.sh edit \
  --dit ./models/h3-image/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors \
  --vae ./models/h3-image/vae/minimax_h3_image_vae_fp16.safetensors \
  --text-encoder ./models/h3-image/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --image /tmp/fox.png \
  --prompt "put the fox in snow, keep its pose" --out /tmp/fox_snow.png
```

4 steps is the default and the fast end of the quality curve; the model was distilled on
a 6-step schedule, so `--steps 6` is the other setting worth trying on a subject you
care about. Guidance above 1.0 costs a second pass per step, and the reference images are
kept in both passes — they are the edit's subject, not something to negate.

`--upscale` here is a codec round trip (decode, resize, re-encode): there is no
upscaler trained for this model's latents.

For every flag (size, steps, seed, multi-image references, KV cache, LoRAs, upscaling,
post-processing), see the [CLI reference](../cli.md) and the
[HTTP API reference](../api.md).
