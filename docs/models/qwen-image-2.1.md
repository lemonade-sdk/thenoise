# Qwen-Image 2.1

**One model for everything: t2i *and* edit.**

Qwen-Image 2.1 is the newest generation: a single-stream DiT that does
text-to-image **and** instruction-based editing from one checkpoint, with a full
Qwen3-VL-8B conditioner and the engine's first RGBA VAE (Wan 2.2 layout — the
pipeline composites alpha onto white at the boundaries that can't carry it).
The reference-latent KV cache is architectural here, so it is exact and on by
default.

- Text-to-image **and editing** (one DiT, both jobs)
- Reference-latent **KV cache** — exact, enabled by default
- int8-convrot variant available (~17 GB instead of ~32 GB total)
- Built-in defaults: **28 steps, guidance 1.0, euler**
- Web UI **Schedule** presets for the community turbo recipes (see below)

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

| | |
|---|---|
| Architecture | Single-stream DiT (flow matching, causal text/reference prefix) |
| Download size | ~32 GB (bf16) · ~17 GB (int8-convrot) |
| VAE | Wan 2.2-layout 64-ch/16× RGBA VAE (never quantized) |
| Text encoder | Qwen3-VL-8B (LM *and* vision tower) |
| Editing | ✓ |
| KV cache | ✓ (exact, on by default) |
| Default settings | 1024×1024, 28 steps, guidance 1.0, euler |

## Performance (Strix Halo)

BF16, 1024×768, ViggleAI Turbo LoRA:

- Generate @ 6 steps : **~11.6 s**
- Edit @ 6 steps: ~13.1 s

## Examples

*Generated:*

<img width="45%" alt="thenoise_3758387159" src="https://github.com/user-attachments/assets/68384c4f-bcd7-4260-84d7-50d69b49b289" />
<img width="45%" alt="thenoise_3144373276" src="https://github.com/user-attachments/assets/eeb66e4d-8c7e-43ad-8964-6ccd991a644e" />


*Edited:*

| Before | After | Prompt |
|---|---|---|
| <img width="100%" alt="thenoise_3758387159" src="https://github.com/user-attachments/assets/dfc04798-3d16-4606-b77c-66571c659220" /> | <img width="100%" alt="thenoise_edit_486449965" src="https://github.com/user-attachments/assets/b044fb9f-64b6-4384-8e15-5550dec04339"/> | Color image |
| <img width="100%" alt="thenoise_3758387159" src="https://github.com/user-attachments/assets/8f73c45e-2964-48bb-ad99-f112ebaa306a" /> | <img width="100%" alt="thenoise_edit_4274534890" src="https://github.com/user-attachments/assets/bfea56c6-f906-452a-baf8-1f3dd962d5e0"/> | Make image realistic |
|<img width="100%" alt="thenoise_3758387159" src="https://github.com/user-attachments/assets/86b1c67f-1d84-4abe-aa41-a1431e956ff2" /> | <img width="100%" alt="thenoise_edit_3432415454" src="https://github.com/user-attachments/assets/30c5ceaa-cbd2-4476-a560-e0d27faa54f1"/> | Write in diagonal on top "Rome, 2026" in hand-writing font. Make image colorful.|
| <img height="600px" alt="generated-1786770947" src="https://github.com/user-attachments/assets/448e73f2-eb3c-41a1-92ca-08b787d865f4" /> | <img height="600px" alt="thenoise_edit_3337899808" src="https://github.com/user-attachments/assets/8b089a1a-afdc-49c3-9588-fe038a502a05" /> | Remove background from the image. Place the subject on "theNoise" logo text written in yellow Space Mono font.|
| <img height="500px" alt="thenoise_2997842605" src="https://github.com/user-attachments/assets/a00c80fb-2a89-44fc-91f6-c07a9504a37d" /><img height="500px" alt="thenoise_1235292549" src="https://github.com/user-attachments/assets/3d8ef355-b84c-4b0a-bcb7-9e1f33cdd58f" /> | <img width="768" height="1024" alt="thenoise_edit_2132125500" src="https://github.com/user-attachments/assets/39f832c8-041b-44a3-8981-add59d7b8ec8" /> | Change the clothing of model from &lt;image1&gt; to that of model from &lt;image2&gt;. Remove the hat and make her hairstyle a wavy fashion bob. Make the background bordeaux.|

## Download

```bash
.venv/bin/python scripts/download.py --model qwen-image-2.1
```

Fetches the bf16 DiT (~14 GB), the Qwen3-VL-8B text encoder (~17.5 GB) and the
VAE (~0.7 GB). Options:

```bash
# int8-convrot DiT AND text encoder instead (~17 GB total)
.venv/bin/python scripts/download.py --model qwen-image-2.1 --int8-convrot
```

Everything lands in `./models/qwen_image21/...`.

## Usage

Generate:

```bash
./thenoise.sh generate \
  --dit ./models/qwen_image21/diffusion_models/qwen_image_2.1_bf16.safetensors \
  --vae ./models/qwen_image21/vae/qwen_image_2.1_vae_bf16.safetensors \
  --text-encoder ./models/qwen_image21/text_encoders/qwen3vl_8b_bf16.safetensors \
  --prompt "a bioluminescent forest at night" \
  --out forest.png
```

Edit (`--image` is repeatable; the KV cache is on by default for this model):

```bash
./thenoise.sh edit \
  --dit ./models/qwen_image21/diffusion_models/qwen_image_2.1_bf16.safetensors \
  --vae ./models/qwen_image21/vae/qwen_image_2.1_vae_bf16.safetensors \
  --text-encoder ./models/qwen_image21/text_encoders/qwen3vl_8b_bf16.safetensors \
  --image fox.png \
  --prompt "a fox wearing a red scarf" \
  --out fox_edited.png
```

Serve over HTTP with the web UI (open <http://localhost:8000/>):

```bash
./thenoise.sh serve \
  --dit ./models/qwen_image21/diffusion_models/qwen_image_2.1_bf16.safetensors \
  --vae ./models/qwen_image21/vae/qwen_image_2.1_vae_bf16.safetensors \
  --text-encoder ./models/qwen_image21/text_encoders/qwen3vl_8b_bf16.safetensors \
  --host 127.0.0.1 --port 8000
```

For every flag (size, steps, seed, multi-image references, KV cache, LoRAs,
upscaling, post-processing), see the [CLI reference](../cli.md) and the
[HTTP API reference](../api.md).

## Schedule presets

The web UI's **Schedule** dropdown replaces the step count with a hand-picked sigma
grid, for the turbo recipes the community ships:

| Schedule | Steps |
|---|---|
| `Pruna Image 2.1 8 steps` | 8 |
| `Pruna Image 2.1 5 steps` | 5 |
| `Viggle Turbo` | 6 |

They go over the wire as [`sigmas`](../api.md#custom-sigmas), so the same grids can be
posted to `/text2image` / `/edit` by hand.
