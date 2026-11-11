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

## Examples

*Generated:*

<img width="45%" alt="Photo" src="https://github.com/user-attachments/assets/49e903ac-704e-452c-b38b-6f359b32dce1" />
<img width="45%" alt="Sketch" src="https://github.com/user-attachments/assets/2964384a-4a41-4233-b636-0dd5dd3a5a10" />

*Edited:*

| Before | After | Prompt |
|---|---|---|
| <img width="100%" alt="Sketch" src="https://github.com/user-attachments/assets/2964384a-4a41-4233-b636-0dd5dd3a5a10" /> | <img width="100%" alt="thenoise_edit_32470856" src="https://github.com/user-attachments/assets/14911881-9266-4d0c-9a0f-ee7de91f8cfe" /> | Color image |
| <img width="100%" alt="Sketch" src="https://github.com/user-attachments/assets/2964384a-4a41-4233-b636-0dd5dd3a5a10" /> | <img width="100%" alt="thenoise_edit_514649780" src="https://github.com/user-attachments/assets/e6e1a5a8-f694-45c1-9cbc-a11e65eff670" /> | Make image realistic |
|<img width="100%" alt="Sketch" src="https://github.com/user-attachments/assets/2964384a-4a41-4233-b636-0dd5dd3a5a10" /> | <img width="100%" alt="thenoise_edit_1358020328" src="https://github.com/user-attachments/assets/d03df717-ff72-4c69-98f5-6fdcb6d626b2" /> | Write in diagonal on top "Rome, 2026" in hand-writing font. Make image colorful.|
| <img height="500px" alt="Photo" src="https://github.com/user-attachments/assets/49e903ac-704e-452c-b38b-6f359b32dce1" /><img height="500px" alt="thenoise_1235292549" src="https://github.com/user-attachments/assets/3d8ef355-b84c-4b0a-bcb7-9e1f33cdd58f" /> | <img width="2592" height="3456" alt="thenoise_edit_3501531793" src="https://github.com/user-attachments/assets/460ecaf3-a776-4d30-9985-d7b41c5d84c5" /> | Change the clothing of model from &lt;image1&gt; to that of model from &lt;image2&gt;. Remove the hat and make her hairstyle a wavy fashion bob. Make the background bordeaux.|

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
