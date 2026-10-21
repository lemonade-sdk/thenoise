# Fibo 2.0

**Bria's generate-and-edit model. One checkpoint, both jobs.**

Fibo 2.0 is a single-stream diffusion transformer that never puts the prompt in the
sequence. A Perceiver squeezes it into a few hundred "gist" tokens, the five injection
blocks out of 44 cross-attend to the prompt directly, and the one stream they all share
carries `[gist, image, references]`. The codec is the Flux.2 AE, so latents are the
canonical 128-channel ones at 16 pixels per cell and one DiT token is one latent cell.

- Text-to-image **and editing** (one or more reference images + instruction)
- Built-in defaults: **4 steps, guidance 1.0, euler** (CFG off) — the Edit Turbo recipe
- 8.2 B DiT: 8.4 GB int8-convrot, 16.5 GB bf16
- A prompt is plain text or Bria's structured JSON caption; either can ask for text to
  be written into the picture, and this engine renders that text for the encoder
- int8-convrot checkpoints for both the DiT and the text encoder
- The tokenizer ships inside this repository — nothing to fetch

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

| Spec | Data |
|---|---|
| Architecture | single-stream DiT (flow matching), 44 blocks, dim 3584, 28 heads, SwiGLU 9728, + 2 noise refiners + a 2-layer Perceiver |
| Parameters | 8.2 B |
| Download size | ~13.6 GB int8-convrot (8.4 GB DiT + 4.9 GB text encoder + 0.3 GB VAE); ~25.7 GB with the bf16 DiT |
| VAE | Flux.2 AE, 128-ch latent at 16 px per cell |
| Text encoder | Qwen3-VL 4B, read as six bundles of three tapped decoder layers |
| Editing | ✓ |
| KV cache | ✓ |
| Default settings | 1024×1024, 4 steps, guidance 1.0, euler |

## Checkpoints

| Checkpoint | File |
|---|---|
| `--variant edit-turbo` *(default)* | `fibo2_edit_turbo_{bf16,int8_convrot}.safetensors` |
| `--variant dev` | `fibo2_dev_{bf16,int8_convrot}.safetensors` |

Same architecture and same 8.2 B weights: `dev` is the 50-step model at guidance 5 and
`edit-turbo` its 4-step distilled child. Nothing in either file says which is which, so
the engine ships the distilled recipe and the flags are the escape hatch:

```bash
--dit ./models/fibo2/fibo2_dev_int8_convrot.safetensors --steps 50 --guidance-scale 5
```

## Lemonade recipes

Ready-made [Lemonade](https://lemonade-server.ai/docs/dev/backends-reference/#backends)
recipes for this model - each pins the checkpoints and the generation defaults, with the
int8-convrot text encoder in every one:

- [Fibo-2.0-Edit-Turbo-INT8.json](../../recipes/fibo2/Fibo-2.0-Edit-Turbo-INT8.json) - Edit Turbo INT8-ConvRot, 4 steps
- [Fibo-2.0-Edit-Turbo.json](../../recipes/fibo2/Fibo-2.0-Edit-Turbo.json) - Edit Turbo BF16, 4 steps
- [Fibo-2.0-Dev-INT8.json](../../recipes/fibo2/Fibo-2.0-Dev-INT8.json) - Dev INT8-ConvRot, 50 steps at guidance 5
- [Fibo-2.0-Dev.json](../../recipes/fibo2/Fibo-2.0-Dev.json) - Dev BF16, 50 steps at guidance 5

## Performance (Strix Halo)

Not measured yet.

## Download

```bash
# bf16 DiT (default), Qwen3-VL 4B text encoder and the Flux.2 AE
.venv/bin/python scripts/download.py --model fibo2

# int8-convrot DiT (8.4 GB instead of 16.5)
.venv/bin/python scripts/download.py --model fibo2 --int8-convrot

# the 50-step parent instead of Edit Turbo
.venv/bin/python scripts/download.py --model fibo2 --variant dev --int8-convrot
```

All three files land in `./models/fibo2/`. The AE is the Flux.2 one, so if you already
have it from another model, point `--vae` at that file instead of downloading a second
copy.

## Usage

Generate:

```bash
./thenoise.sh generate \
  --dit ./models/fibo2/fibo2_edit_turbo_int8_convrot.safetensors \
  --vae ./models/fibo2/flux2-vae.safetensors \
  --text-encoder ./models/fibo2/qwen3vl_4b_int8_convrot.safetensors \
  --prompt "a half-eaten croissant on a ceramic plate, window light" \
  --out croissant.png
```

Edit — `--image` is repeatable, and each image becomes a reference on its own RoPE
plane, modulated at `t = 0` while the target is denoised:

```bash
./thenoise.sh edit \
  --dit ./models/fibo2/fibo2_edit_turbo_int8_convrot.safetensors \
  --vae ./models/fibo2/flux2-vae.safetensors \
  --text-encoder ./models/fibo2/qwen3vl_4b_int8_convrot.safetensors \
  --image croissant.png \
  --prompt "make it night, keep everything else exactly as it is" \
  --out croissant_night.png
```

Ask for words in the picture and Qwen3-VL is shown them rendered (see the notes) — with
images, `edit_instruction` is the instruction and the rest of the caption describes the
result:

```bash
./thenoise.sh edit \
  --dit ./models/fibo2/fibo2_edit_turbo_int8_convrot.safetensors \
  --vae ./models/fibo2/flux2-vae.safetensors \
  --text-encoder ./models/fibo2/qwen3vl_4b_int8_convrot.safetensors \
  --image poster.png \
  --prompt '{"edit_instruction": "add a headline across the top", "text_render": [{"text": "ROME, 2026"}]}' \
  --out poster2.png
```

Serve over HTTP with the web UI (open <http://localhost:8000/>) — the Edit tab appears
because this model supports editing:

```bash
./thenoise.sh serve \
  --dit ./models/fibo2/fibo2_edit_turbo_int8_convrot.safetensors \
  --vae ./models/fibo2/flux2-vae.safetensors \
  --text-encoder ./models/fibo2/qwen3vl_4b_int8_convrot.safetensors \
  --host 127.0.0.1 --port 8000
```
