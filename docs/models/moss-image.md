# MOSS-Image

**OpenMOSS' text-to-image model. Quality first, steps second.**

MOSS-Image is a joint-stream diffusion transformer: prompt tokens and image tokens
are put in one sequence and read by the same attention, in 42 blocks of width 3072.
It is conditioned by Qwen3.5-9B and draws on the Flux.2 AE, so its latent is the
packed 128-channel one at 16 pixels per cell.

- Text-to-image, plus `ref2i`: one reference read by the conditioner's vision tower
  (experimental) — there is no reference-latent path, so nothing is repainted
- Built-in defaults: **25 steps, guidance 3.0, er_sde**
- Not a distilled checkpoint: `--sampler euler --steps 50` is the release recipe, and
  the stochastic `er_sde` gets about the same picture in half the steps
- int8-convrot DiT (5.0 GB); the bf16 original is ~9.5 GB
- 8-step distilled DiTs of both checkpoints, bf16 and int8-convrot — one file, no LoRA
  to load
- The tokenizer ships inside this repository, chat template included — nothing to fetch

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

| Spec | Data |
|---|---|
| Architecture | joint-stream DiT (flow matching), 42 blocks, dim 3072, 24 heads |
| Download size | ~14 GB int8-convrot (5.0 GB DiT + 9.1 GB text encoder + 0.3 GB VAE); ~19 GB with the bf16 DiT |
| VAE | Flux.2 AE, 128-ch packed latent at 16× (the file Flux.2 Klein uses) |
| Text encoder | Qwen3.5-9B, int8-convrot (the only release); its vision tower reads a `ref2i` reference |
| Editing | `ref2i` — the reference reaches the DiT as conditioner tokens, never as a latent |
| KV cache | — |
| Default settings | 1024×1024, 25 steps, guidance 3.0, er_sde |

## Checkpoints

| Checkpoint | File |
|---|---|
| `--variant sft` *(default)* | `moss_image_sft_{bf16,int8_convrot}.safetensors` |
| `--variant base` | `moss_image_base_{bf16,int8_convrot}.safetensors` |

The `sft` checkpoint is the one to use: `base` is its un-fine-tuned parent, useful
for LoRA training rather than for drawing.

## Lemonade recipes

Ready-made [Lemonade](https://lemonade-server.ai/docs/dev/backends-reference/#backends)
recipes for this model - each pins the checkpoints and the generation defaults:

- [MOSS-Image-SFT-8steps.json](../../recipes/moss-image/MOSS-Image-SFT-8steps.json) - SFT BF16 with the 8-step LoRA baked, 8 steps, guidance 1
- [MOSS-Image-SFT-8steps-INT8.json](../../recipes/moss-image/MOSS-Image-SFT-8steps-INT8.json) - same, INT8-ConvRot
- [MOSS-Image-Base-8steps.json](../../recipes/moss-image/MOSS-Image-Base-8steps.json) - base BF16 with the 8-step LoRA baked
- [MOSS-Image-Base-8steps-INT8.json](../../recipes/moss-image/MOSS-Image-Base-8steps-INT8.json) - same, INT8-ConvRot
- [MOSS-Image-SFT.json](../../recipes/moss-image/MOSS-Image-SFT.json) - SFT BF16, 25 steps
- [MOSS-Image-SFT-INT8.json](../../recipes/moss-image/MOSS-Image-SFT-INT8.json) - SFT INT8-ConvRot, 25 steps

The 8-step recipes bake lodestones'
[MOSS-Image 8-step DMD2](https://huggingface.co/lodestones/Kromoss) LoRA into the DiT.
Draw with them at 8 steps, guidance 1 and `--sampler euler`.

## Performance (Strix Halo)

INT8-ConvRot, [8 step DMD LoRA](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd), 768x1024:

- Generate @ 8 steps : **~10.1 s**

## Examples

*Generated:*

<img width="45%" alt="thenoise_4115995311" src="https://github.com/user-attachments/assets/733672c2-9fc5-4924-92ba-8ee4b3a18c35" />
<img width="45%" alt="thenoise_3119577629" src="https://github.com/user-attachments/assets/8db26707-2e1c-4508-81b4-d9b29da42cdc" />

## Download

```bash
# bf16 DiT (default)
.venv/bin/python scripts/download.py --model moss-image

# int8-convrot DiT (the text encoder is int8-convrot either way)
.venv/bin/python scripts/download.py --model moss-image --int8-convrot

# the 8-step distilled DiT of either checkpoint (--variant 8steps | base-8steps)
.venv/bin/python scripts/download.py --model moss-image --variant 8steps

# ... and BFL's small-decoder AE, which decodes ~1.4x faster into the same latents
.venv/bin/python scripts/download.py --model moss-image --small-decoder
```

The VAE is the Flux.2 AE, so if you already have Flux.2 Klein's files, point
`--vae` at that `flux2-vae.safetensors` instead of downloading it again.

## Usage

Generate:

```bash
./thenoise.sh serve \
  --dit ./models/moss-image/diffusion_models/moss_image_sft_int8_convrot.safetensors \
  --text-encoder ./models/moss-image/text_encoders/qwen3.5_9b_int8_convrot.safetensors \
  --vae ./models/moss-image/vae/flux2-vae.safetensors
```

```bash
./thenoise.sh generate "a half-eaten croissant on a ceramic plate, window light" \
  --dit ./models/moss-image/diffusion_models/moss_image_sft_int8_convrot.safetensors \
  --text-encoder ./models/moss-image/text_encoders/qwen3.5_9b_int8_convrot.safetensors \
  --vae ./models/moss-image/vae/flux2-vae.safetensors
```

## Reading a reference (`ref2i`) — experimental

A picture cannot be repainted here: the architecture has no reference-latent path. What
it can do is show one to the vision tower its conditioner already ships with. The
released Qwen3.5 file carries the visual weights and this engine has always loaded them
and never run them; now the tower's output joins the conditioning, the DiT reads it as
tokens beside the prompt, and the run draws something new beside the reference instead
of over it.

```bash
./thenoise.sh edit \
  --dit ./models/moss-image/diffusion_models/moss_image_sft_int8_convrot.safetensors \
  --text-encoder ./models/moss-image/text_encoders/qwen3.5_9b_int8_convrot.safetensors \
  --vae ./models/moss-image/vae/flux2-vae.safetensors \
  --image /tmp/fox.png --prompt "the same fox asleep on a red scarf"
```

What to expect, and what not to:

- **One image**, arriving at the reference's own aspect ratio (about a megapixel)
  unless `--width`/`--height` say otherwise.
- **A hint, not a copy.** The reference is scaled to a 384² area on the Qwen3.5 vision
  grid (a 16 px patch, merged 2×2) before the tower sees it, so it carries a subject,
  a palette and a composition — no texture, no legible text, no pixel alignment.
- **Nothing here was trained for this.** The release never showed MOSS a picture, so
  how well the tower's features read into the DiT is an experiment, not a spec. The
  conditioner is int8-convrot throughout, vision tower included.

## Notes on the recipe

**Guidance is on by default.** CFG 3.0 — two forwards per step, one on the prompt and
one on the negative. Passing `--guidance-scale 1` drops to a single forward and saves
half the time, at a real cost in adherence.

**The sampler is `er_sde`.** The release recipe is 50 Euler steps, and the default is
25 steps of the stochastic `er_sde` instead, which reports better per second. Both
samplers read the same schedule: a statically shifted flow grid, unchanged by
resolution.

**Sizes are rounded up to 16 px**, one DiT token per latent cell: 1024×1024 is
64×64 tokens.


