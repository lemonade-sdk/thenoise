# Flux.3

**Newest Flux. Built to edit.**

Flux.3 is Black Forest Labs' third-generation Flux, released as a distilled
editing model: hand it one or more reference images plus an instruction and it
returns the edited image, leaving the parts you didn't mention alone. Plain
text-to-image works too. The released checkpoint is the small variant (5.5B
parameters), and it shares its VAE with Flux.2 Klein and its text encoder with
Mage-Flow — so if you already run either of them, you need one new file.

It wants more steps than the other distilled models here — **15**, not 4 — and
that is the deal: reach for it when the edit has to keep a face, a logo, a
product or a background intact, and reach for Mage-Flow or Klein when you want
the picture back quickly.

- Text-to-image **and editing** (one or more reference images + instruction)
- Built-in defaults: **15 steps, guidance 1.0, euler** (CFG off)
- Any size you ask for, rounded up to a multiple of 16 px
- int8-convrot checkpoint (5.8 GB); the bf16 original is ~11 GB
- **No KV cache** (see note below)

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

| Spec | Data |
|---|---|
| Architecture | Distilled flow-map DiT, dim 3072 (Flux.2 packed 128-ch latent) |
| Checkpoint size | 5.8 GB int8-convrot · ~11 GB bf16 |
| VAE | Flux.2 VAE (shared with Flux.2 Klein) |
| Text encoder | Qwen3-VL 4B (the same file Mage-Flow uses) |
| Editing | ✓ |
| KV cache | — (not supported, see note) |
| Default settings | 1024×1024, 15 steps, guidance 1.0, euler |

## Getting the weights

`scripts/download.py` has no Flux.3 entry. The checkpoint sits behind a login on
Hugging Face and there is no mirror to fall back on, so there is nothing the
script could fetch unattended. Grab three files yourself:

| File | What | Where |
|---|---|---|
| `flux3_image_klein_sd_int8_convrot.safetensors` | the DiT | [black-forest-labs/flux3](https://huggingface.co/black-forest-labs/flux3) — gated: accept the terms, then download |
| `flux2-vae.safetensors` | VAE | already in `./models/klein/` if you ran `--model klein` |
| `qwen3vl_4b_bf16.safetensors` | text encoder | already in `./models/mage_flow/` if you ran `--model mage-flow` |

They can live anywhere — every path goes on the command line. The examples
below assume the DiT in `./models/klein3/`. There is no Lemonade recipe for the
same reason there is no download entry: a recipe has to point at files the
installer can fetch by itself.

## Usage

Generate:

```bash
./thenoise.sh generate \
  --dit ./models/klein3/flux3_image_klein_sd_int8_convrot.safetensors \
  --vae ./models/klein/split_files/vae/flux2-vae.safetensors \
  --text-encoder ./models/mage_flow/qwen3vl_4b_bf16.safetensors \
  --prompt "a red panda drinking tea in a bamboo forest" \
  --out panda.png
```

Edit — `--image` is repeatable and each image becomes a reference the model can
look at. The **first** image sets the output size (1024 on its longest side,
aspect preserved) unless you pass `--width`/`--height`:

```bash
./thenoise.sh edit \
  --dit ./models/klein3/flux3_image_klein_sd_int8_convrot.safetensors \
  --vae ./models/klein/split_files/vae/flux2-vae.safetensors \
  --text-encoder ./models/mage_flow/qwen3vl_4b_bf16.safetensors \
  --image panda.png \
  --prompt "put the panda on a bar stool, keep its pose and fur exactly as they are" \
  --out panda_edited.png
```

Serve over HTTP with the web UI (open <http://localhost:8000/>) — the Edit tab
appears because this model supports editing:

```bash
./thenoise.sh serve \
  --dit ./models/klein3/flux3_image_klein_sd_int8_convrot.safetensors \
  --vae ./models/klein/split_files/vae/flux2-vae.safetensors \
  --text-encoder ./models/mage_flow/qwen3vl_4b_bf16.safetensors \
  --host 127.0.0.1 --port 8000
```

## Settings

| Flag | What to do with it |
|---|---|
| `--steps` | 15 by default. Draft at 8 while you are iterating on a prompt; past ~20 the picture stops moving. |
| `--guidance-scale` | leave it at 1.0. This checkpoint is distilled to guide itself. Above 1.0 your `--negative-prompt` comes into play and every step runs twice — roughly half the speed, and usually a worse result. |
| `--width` / `--height` | multiples of 16; 1024 by default. On `edit`, size comes from the first image unless you set them. |
| `--sampler` | `euler` — the schedule and the sampler are a matched pair on this model. |
| `--lora` | Flux.3 LoRAs work. Flux.2 Klein LoRAs do **not**: different architecture. |
| `--upscale-factor` / `--pixel-upscaler` | works as everywhere else; the built-in latent refiner is the Flux.2 one. |

## Prompting

- **Editing:** ask for one change and name what must survive it — "swap the sofa
  for a leather armchair, keep the rug, the lighting and the wall art". Vague
  "make it better" prompts give the model licence to repaint everything.
- **Multiple references:** they are read as equals, not as "image 1 with image 2
  pasted on it". Say what each one is for: "use the dress from the second image
  on the person in the first".
- **More than one edit:** do them in series, feeding the result back in. Each
  pass starts from a clean image and the instruction stays unambiguous.

## Performance (Strix Halo)

<!-- TODO(maintainer): measured numbers — int8-convrot DiT, Qwen3-VL 4B, Flux.2 VAE -->

| Run | Time |
|---|---|
| Generate @ 15 steps, 1024×1024 | — |
| Generate @ 8 steps, 1024×1024 | — |
| Edit @ 15 steps, 1 reference | — |
| Edit @ 15 steps, 3 references | — |

> **KV cache note:** Flux.3 cannot freeze a reference image's internals between
> steps. Its distilled steps are conditioned on where the *next* step lands, and
> that changes every step — so what the cache would reuse is out of date the
> moment it is written. Editing therefore costs proportionally more with each
> reference you add, and passing `--kv-cache` is refused with an explanation
> rather than silently ignored. If your workflow is many-reference editing on a
> budget, Flux.2 Klein with a KV-cache checkpoint is the faster tool.

## Examples

*Generated — 15 steps:*

<!-- image -->

*Edited — instruction: "…":*

| Before | After |
|---|---|
| <!-- before --> | <!-- after --> |

## Limits

- **One checkpoint size today** (`f3-image-small`, 5.5B). Detection reads the
  architecture from the file, so a larger Flux.3 should load when it appears —
  the 15-step default and the schedule are tuned to the released one.
- **No KV cache**, for the reason above.
- **No download entry and no Lemonade recipe**, until the weights get a
  login-free mirror.

For every flag (sizes, seeds, multi-image references, LoRAs, upscaling,
post-processing), see the [CLI reference](../cli.md) and the
[HTTP API reference](../api.md).
