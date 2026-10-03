# Anima

**The small one. The fast one.**

Anima is a 2B-parameter Cosmos-Predict2 MMDiT distilled for few-step generation.
At ~5.4 GB it is the lightest model TheNoise ships with, and at ~5 seconds per
1024×768 image on Strix Halo it is also the fastest. If you want to iterate on
prompts quickly — dozens of takes, minimal wait — this is the model to start on.

- Text-to-image only (no editing)
- Built-in defaults: **8 steps, guidance 1.0** (CFG off) — no tuning needed
- int8-convrot variant available (smaller download, lower memory)

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

| Spec | Data |
|---|---|
| Architecture | 2B Cosmos-Predict2 MMDiT (flow matching) |
| Download size | ~5.4 GB (bf16), less with `--int8-convrot` |
| VAE | Qwen-Image VAE |
| Text encoder | Qwen3-0.6B |
| Editing | — |
| KV cache | — |
| Default settings | 1024×1024, 8 steps, guidance 1.0, flow shift 3.0 |

## DiT variants

| Variant | Steps | Character |
|---|---|---|
| `turbo-v1.0` *(default)* | 8 | fewest steps, the workhorse |
| `aesthetic-v1.1` | more | aesthetically tuned |
| `base-v1.0` | ~20, CFG ~4 | base (non-distilled) checkpoint |

## Lemonade recipes

Ready-made [Lemonade](https://lemonade-server.ai/docs/dev/backends-reference/#backends)
recipes for this model - each pins the checkpoints and the generation defaults:

- [Anima-Turbo.json](../../recipes/anima/Anima-Turbo.json) - Turbo BF16, 8 steps
- [Anima-Turbo-INT8.json](../../recipes/anima/Anima-Turbo-INT8.json) - Turbo INT8-ConvRot, 8 steps
- [Anima-Aesthetic.json](../../recipes/anima/Anima-Aesthetic.json) - Aesthetic BF16, 20 steps, CFG 4
- [Anima-Aesthetic-INT8.json](../../recipes/anima/Anima-Aesthetic-INT8.json) - Aesthetic INT8-ConvRot, 20 steps, CFG 4
- [Anima-Base.json](../../recipes/anima/Anima-Base.json) - Base BF16, 20 steps, CFG 4
- [Anima-Base-INT8.json](../../recipes/anima/Anima-Base-INT8.json) - Base INT8-ConvRot, 20 steps, CFG 4

## Performance (Strix Halo)

- Turbo @ 8 steps: **~5 s** (1024×768)
- Base @ 20 steps, CFG 4: ~20 s

## Examples

*Generated — `(Turbo, 8 steps)`:*

<img width="45%" alt="thenoise_1653665740" src="https://github.com/user-attachments/assets/f60d2264-69cc-4f7f-b169-9007bd63e9ec" />
<img width="45%" alt="thenoise_1980975552" src="https://github.com/user-attachments/assets/2a975ade-0b15-4bf3-aa04-5c0ec46d00ac" />

*Generated — `(Base, 20 steps, CFG 4)`:*

<img width="45%" alt="thenoise_561879631" src="https://github.com/user-attachments/assets/736299d7-761e-470d-8e35-ed507a708587" />
<img width="45%" alt="thenoise_561879631 (2)" src="https://github.com/user-attachments/assets/fc32adcb-f196-4f7f-8639-0e910ef2523c" />



## Download

```bash
.venv/bin/python scripts/download.py --model anima
```

Other variants / quantization:

```bash
# a specific DiT variant (the value becomes part of the --dit filename)
.venv/bin/python scripts/download.py --model anima --variant aesthetic-v1.1

# int8-convrot DiT instead of bf16
.venv/bin/python scripts/download.py --model anima --int8-convrot
```

## Usage

Generate:

```bash
./thenoise.sh generate \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --prompt "a fox walking in the snow" \
  --out fox.png
```

Serve over HTTP with the web UI (open <http://localhost:8000/>):

```bash
./thenoise.sh serve \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --host 127.0.0.1 --port 8000
```

For every flag (size, steps, seed, LoRAs, upscaling, post-processing), see the
[CLI reference](../cli.md) and the [HTTP API reference](../api.md).
