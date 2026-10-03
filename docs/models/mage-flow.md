# Mage-Flow

**Four steps, two seconds, editing included.**

Mage-Flow is a dual-stream DiT — the 12-layer member of the Qwen-Image block
family — conditioned by Qwen3-VL-4B and decoded by the Mage-VAE, a one-step
diffusion codec. The distilled **Turbo** checkpoint is the fastest thing
TheNoise runs: ~2 s per 1024×1024 image on Strix Halo, CFG off. Generation and
editing are separate checkpoints that share one text encoder and one VAE, and
an edit (reference image + instruction) comes back in ~5 s.

- Text-to-image **and editing** (separate checkpoints, shared TE + VAE)
- Turbo: built-in defaults **4 steps, guidance 1.0, euler** (CFG off)
- Native resolution: any size you ask for, rounded up to a multiple of 16 px —
  no buckets and no clamping
- **No KV cache** (see note below)
- int8-convrot checkpoints for the distilled variants

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

| Spec | Data |
|---|---|
| Architecture | Dual-stream DiT (flow matching), 12 blocks, dim 3072, patch 1 |
| Download size | ~25 GB bf16 (7.7 GB per DiT + 8.9 GB TE + 0.35 GB VAE) · ~18 GB int8-convrot |
| VAE | Mage-VAE — one-step diffusion codec, 128-ch latent at 16× |
| Text encoder | Qwen3-VL-4B (LM *and* vision tower) |
| Editing | ✓ (Edit checkpoint) |
| KV cache | — (not supported) |
| Default settings | 1024×1024, 4 steps, guidance 1.0, euler |

## Checkpoints

| Variant | Steps | int8-convrot |
|---|---|---|
| `turbo` *(default)* | 4 (built-in default) | ✓ |
| `rl` | 20, guidance 5.0 | ✓ |
| `base` | 30, guidance 5.0 | — |

The engine's built-in defaults target Turbo and no checkpoint carries a marker
saying which variant it is, so pass explicit `--steps` / `--guidance-scale` on
every `rl` or `base` run. The un-distilled checkpoints were never released in
int8-convrot.

Turbo, RL and base share one architecture, and the t2i / edit split is a matter
of training, not of structure: any of these files loads under either command,
but use the matching `_edit_` checkpoint for `edit`. The `--image` references
reach the model twice — as Qwen3-VL vision tokens in the text stream and as
latent tokens in the image stream.

## Lemonade recipes

Ready-made [Lemonade](https://lemonade-server.ai/docs/dev/backends-reference/#backends)
recipes for this model - each pins the checkpoints and the generation defaults:

- [Mage-Flow-Turbo.json](../../recipes/mageflow/Mage-Flow-Turbo.json) - Turbo BF16, 4 steps
- [Mage-Flow-Turbo-Edit.json](../../recipes/mageflow/Mage-Flow-Turbo-Edit.json) - Turbo edit BF16, 4 steps
- [Mage-Flow-Turbo-INT8.json](../../recipes/mageflow/Mage-Flow-Turbo-INT8.json) - Turbo INT8-ConvRot, 4 steps
- [Mage-Flow-Turbo-Edit-INT8.json](../../recipes/mageflow/Mage-Flow-Turbo-Edit-INT8.json) - Turbo edit INT8-ConvRot, 4 steps
- [Mage-Flow.json](../../recipes/mageflow/Mage-Flow-RL.json) - RL BF16, 20 steps, CFG 5
- [Mage-Flow-Edit.json](../../recipes/mageflow/Mage-Flow-RL-Edit.json) - RL edit BF16, 20 steps, CFG 5
- [Mage-Flow-INT8.json](../../recipes/mageflow/Mage-Flow-RL-INT8.json) - RL INT8-ConvRot, 20 steps, CFG 5
- [Mage-Flow-Edit-INT8.json](../../recipes/mageflow/Mage-Flow-RL-Edit-INT8.json) - RL edit INT8-ConvRot, 20 steps, CFG 5

## Performance (Strix Halo)

Turbo BF16:

- Generate @ 4 steps: **~1.8 s** (1024×768)
- Edit @ 4 steps: ~4.9 s (1024×1024, no KV cache on this model)

## Examples

*Generated - Turbo @ 4 steps:*

<img width="45%" alt="c3773587-d948-4592-86fe-741f2e854ed2" src="https://github.com/user-attachments/assets/511d1bbe-4f81-41a0-840e-3315b454d673" />
<img width="45%" alt="9aa0cc36-2ec8-4aa5-bc2e-eac17970b125" src="https://github.com/user-attachments/assets/ddae7d8c-b7a4-4652-b0df-53f2bbc49208" />

*Edited - Turbo @ 4 steps:*

| Before | After | Prompt |
|---|---|---|
| <img width="100%" alt="9aa0cc36-2ec8-4aa5-bc2e-eac17970b125" src="https://github.com/user-attachments/assets/ddae7d8c-b7a4-4652-b0df-53f2bbc49208" /> | <img width="100%" alt="7367543c-1f6e-482d-aa61-aff09a0da276" src="https://github.com/user-attachments/assets/f1a7f3bf-d08c-4709-9d08-0269e1001262" /> | Paint the sketch with colors |
| <img width="100%" alt="9aa0cc36-2ec8-4aa5-bc2e-eac17970b125" src="https://github.com/user-attachments/assets/ddae7d8c-b7a4-4652-b0df-53f2bbc49208" /> | <img width="100%" alt="9aee500e-9fbb-4a7e-b354-95f0ab1d963a" src="https://github.com/user-attachments/assets/b8787ef7-5e87-45bd-a782-6696199bd041" /> | Make this realistic |
| <img width="100%" alt="9aa0cc36-2ec8-4aa5-bc2e-eac17970b125" src="https://github.com/user-attachments/assets/ddae7d8c-b7a4-4652-b0df-53f2bbc49208" /> | <img width="100%" alt="287e215f-08dd-45ca-83dd-690927f2c99f" src="https://github.com/user-attachments/assets/b136c5de-93e5-4bc4-bd07-ea19c33d40b4" /> | Write in diagonal on top "Rome, 2026" in hand-writing font. Make image colorful. |

## Download

```bash
# Turbo: t2i DiT + edit DiT + shared text encoder + VAE (default)
.venv/bin/python scripts/download.py --model mage-flow

# only one of the two DiTs (shared TE + VAE are still fetched)
.venv/bin/python scripts/download.py --model mage-flow --image-only
.venv/bin/python scripts/download.py --model mage-flow --edit-only

# non-distilled variants
.venv/bin/python scripts/download.py --model mage-flow --variant rl
.venv/bin/python scripts/download.py --model mage-flow --variant base

# int8-convrot DiT(s) instead of bf16 (turbo / rl)
.venv/bin/python scripts/download.py --model mage-flow --int8-convrot
```

Everything lands in `./models/mage_flow/...` with the DiTs under
`diffusion_models/`; the Qwen3-VL-4B encoder and the Mage VAE are shared by all
variants (and are always bf16). The checkpoints come from
[Comfy-Org/Mage-Flow](https://huggingface.co/Comfy-Org/Mage-Flow).

## Usage

Generate (Turbo):

```bash
./thenoise.sh generate \
  --dit ./models/mage_flow/diffusion_models/mage_flow_turbo_bf16.safetensors \
  --vae ./models/mage_flow/vae/mage_flow_vae_bf16.safetensors \
  --text-encoder ./models/mage_flow/text_encoders/qwen3vl_4b_bf16.safetensors \
  --prompt "a fox walking in the snow" \
  --out fox.png
```

Edit (`--image` is repeatable; the first image sets the output size — and note
there is no `--kv-cache` on this model):

```bash
./thenoise.sh edit \
  --dit ./models/mage_flow/diffusion_models/mage_flow_edit_turbo_bf16.safetensors \
  --vae ./models/mage_flow/vae/mage_flow_vae_bf16.safetensors \
  --text-encoder ./models/mage_flow/text_encoders/qwen3vl_4b_bf16.safetensors \
  --image fox.png \
  --prompt "a fox wearing a red scarf" \
  --out fox_edited.png
```

Serve over HTTP with the web UI (open <http://localhost:8000/>) — the Edit tab
appears because this model supports editing:

```bash
./thenoise.sh serve \
  --dit ./models/mage_flow/diffusion_models/mage_flow_turbo_bf16.safetensors \
  --vae ./models/mage_flow/vae/mage_flow_vae_bf16.safetensors \
  --text-encoder ./models/mage_flow/text_encoders/qwen3vl_4b_bf16.safetensors \
  --host 127.0.0.1 --port 8000
```

For every flag (size, steps, seed, multi-image references, LoRAs, upscaling,
post-processing), see the [CLI reference](../cli.md) and the
[HTTP API reference](../api.md).
