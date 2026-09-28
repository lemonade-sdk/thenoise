# Ming-Image 0.1

**The Lumina/S3-DiT family, with real transparency.**

Ming-Image is an S3-DiT — the same transformer family as
[Z-Image](zimage.md) — conditioned by a condensed BailingMM2 ("Ling-mini-2.0")
mixture-of-experts encoder, and its VAE draws **RGBA**, so a "transparent
background" prompt produces a real alpha channel rather than a checkerboard.

> **Status: text encoding is not implemented yet.** The DiT, its two checkpoint
> formats (bf16 and int8-convrot), the resolution-aware schedule, the RGBA VAE, the
> upscaler wiring and the catalog entry are all in place; what is missing is the
> BailingMM2 conditioner that produces the two tensors the DiT is conditioned on.
> Until it lands, `generate` resolves the checkpoint, loads the DiT and VAE, and
> then stops with an explicit `NotImplementedError`. Tracked as phase 3 of
> [`docs/ming-image-plan.md`](../ming-image-plan.md).

- Text-to-image only (no editing, no `Design-Layer` reference frames yet)
- Built-in defaults **12 steps, guidance 1.0** (CFG off), 1024×1024
- int8-convrot variants for the DiT *and* the text encoder
- ~49 GB of weights in bf16, ~15 GB with int8-convrot

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

| | |
|---|---|
| Architecture | S3-DiT (flow matching), 30 blocks + 2 refiners, dim 3840 |
| Download size | ~49 GB bf16 (12 GB DiT + 35 GB text encoder + 0.25 GB VAE) |
| VAE | Wan2.1-family (Qwen-Image layout), **RGBA**, 16ch latent at 8× |
| Latent normalisation | scalar: `canonical = raw × 8.0064` |
| Text encoder | BailingMM2 "Ling-mini-2.0" (16 B MoE thinker + 28-layer connector) |
| Conditioning | two tensors: `cap_feats` (256 query tokens) + `direct_context` |
| Editing | — |
| KV cache | — |
| Default settings | 1024×1024, 12 steps, guidance 1.0, euler sampler |

## What makes it a model of its own

Inside the shared Lumina core it differs from Z-Image in exactly two ways, and both
are visible in the released files:

* **No learned pad tokens.** Both token streams are still padded to a multiple of
  32 — that padded caption length is what positions the image — but the pad slots
  hold zeros and are masked out of attention. This is also how the engine tells the
  two checkpoints apart: the int8 export names every module exactly like Z-Image
  does, and only the missing `x_pad_token`/`cap_pad_token` separates them.
* **Two conditioning tensors.** `cap_feats` (256 learned query tokens, 2560-wide)
  goes through `cap_embedder`; `direct_context` (per-prompt-token "shallow" VLM
  features, already 3840-wide) is concatenated to the caption block *after* it. The
  second one is part of the caption length, so it moves the image's temporal
  positions with it.

## Sampling: the shift moves with the resolution

The reference forces dynamic shifting on, and the shift is derived from the image
token count:

| Size | tokens | shift `e^mu` |
|---|---|---|
| 512×512 | 1024 | 1.878 |
| **1024×1024** | **4096** | **3.158** |
| 2048×2048 | 16384 | 3.857 |

The engine implements that curve rather than ComfyUI's hard-coded 3.16 — at 1024²
the two agree, above it they do not. Sizes are rounded up to multiples of 16
(VAE 8× × patch 2), and 1024 and 2048 are the buckets the model was trained on.

## Transparency

The VAE takes and returns four pixel channels, so the PNG the engine writes carries
the alpha the model produced — no keying, no matting. The trick is on the prompt
side: the model responds to explicit background language ("isolated on a
transparent background", "sticker", "logo on transparent background"), which is a
property of how it was trained, not of the engine.

## Memory

bf16 wants ~49 GB of resident weights plus the denoise/decode peak. On a 128 GB
Strix Halo the auto-detection keeps everything resident and nothing moves per
request. On a smaller GPU the text encoder is offloaded and re-streamed per prompt,
which is brutally slow — either give it a device that holds it, or download the
int8-convrot set (~15 GB) and/or pass `--offload-device` (see
[the CLI reference](../cli.md)).

## Checkpoints

| Checkpoint | File |
|---|---|
| `--variant design` *(default)* | `ming_image_0.1_design_{bf16,int8_convrot}.safetensors` |
| `--variant design-layer` | `ming_image_0.1_design_layer_{bf16,int8_convrot}.safetensors` |

The `Design-Layer` variant is a design/layout-specialised export. Its DiT loads and
detects like the base one; its extra behaviour (reference frames on the frame axis)
is not wired up yet, so treat it as a second text-to-image style.

## Download

```bash
# bf16 (default)
.venv/bin/python scripts/download.py --model ming-image

# int8-convrot DiT + text encoder
.venv/bin/python scripts/download.py --model ming-image --int8-convrot

# the Design-Layer variant
.venv/bin/python scripts/download.py --model ming-image --variant design-layer
```

Files land in `./models/ming_image/{diffusion_models,text_encoders,vae}/`.

## Usage

*(Not runnable until the text encoder lands — kept here so the invocation is ready.)*

```bash
./thenoise.sh generate \
  --dit ./models/ming_image/diffusion_models/ming_image_0.1_design_bf16.safetensors \
  --vae ./models/ming_image/vae/ming_image_vae_bf16.safetensors \
  --text-encoder ./models/ming_image/text_encoders/ming_image_0.1_ling_mini_2.0_bf16.safetensors \
  --prompt "a glass flask of fireflies, isolated on a transparent background" \
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

## Upscaling

The 2× refine reuses the committed Wan2.1 Sesqui network over Ming-Image's latent —
its VAE *is* the Wan/Qwen 16-channel VAE architecture, only normalised by a single
scalar instead of a per-channel z-score, so the latent-domain transform is exact.
Whether the *picture* it produces is good — especially for RGBA output, which Sesqui
never saw in training — is a measurement still out; if it disappoints, dropping back
to the engine's fallback behaviour is a one-line change in the adapter.

## LoRAs

LoRAs resolve against either naming generation: a trained against the released bf16
file (`to_q/to_k/to_v`, `norm_q/norm_k`, `to_out.0`) is folded onto the fused
`qkv`/`out`/`qk_norm` tree, and one trained against the fused tree applies unchanged.

## Performance (Strix Halo)

Not measured yet.
