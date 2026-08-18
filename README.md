# TheNoise

A text-to-image diffusion inference engine. Tested on Strix Halo and Strix Point.

Loads one model at a time and generates images from text prompts. Available as a CLI tool, an HTTP API (with a simple web UI).

<img width="2048" height="1070" alt="thenoise-screenshot" src="https://github.com/user-attachments/assets/5731e570-efb2-43b5-8f71-b6d11d57c8aa" />

---

## Why? ComfyUI exists

Yes, and ComfyUI will always be better than this for the advanced user. This is good for the following scenarios:

1. You got a Strix Halo (congratulations!) and want to quickly start generating images
2. You don't want to care about "workflows"
3. You want to add an easy but powerful image generation endpoint for usage through other software
4. You want something targeted at your machine. Our goal is to optimize this for Strix Halo as much as possible.

---

## Acknowledgments

This project incorporates code from:

1. [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner)
2. [SD Scripts](https://github.com/kohya-ss/sd-scripts)
3. [SesquiLSR](https://github.com/LoganBooker/SesquiLSR)

plus smaller snippets from other sources or transitively inherited through the above codebases.

---

## Setup

The steps below are copy-pasteable end to end. They take you from a fresh clone
to a generated image.

### 1. Install `uv`

[`uv`](https://github.com/astral-sh/uv) is the only prerequisite — it provides
the Python interpreter and installs every dependency:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 2. Clone the repo

```bash
git clone https://github.com/lemonade-sdk/thenoise.git
cd thenoise
```

### 3. Bootstrap the environment

`thenoise.sh` creates the `.venv`, installs the ROCm build of PyTorch, and
installs the project in editable mode. Running it with `--help` does all of that
without needing any model weights yet:

```bash
./thenoise.sh --help
```

This is the slow step — it downloads several GB of ROCm PyTorch wheels.
Subsequent runs skip the torch install (detected via `import torch`).

By default the script autodetects the GPU's architecture. Override with the
`GFX_ARCH` environment variable, which applies to every `./thenoise.sh`
invocation. Supported targets are `gfx1150`, `gfx1151`, and `gfx1152`:

```bash
GFX_ARCH=gfx1151 ./thenoise.sh --help
```

### 4. Download a model

The download scripts need `huggingface_hub`, which is a `scripts` extra and is
**not** installed by `thenoise.sh` (it only installs runtime deps). Install it
once, then use the venv's Python (a bare system `python` will not work):

```bash
uv pip install -e ".[scripts]"
.venv/bin/python scripts/download_anima.py --out ./models/anima --variant turbo-v1.0
```

Anima is the smaller of the two original supported models (~5.4 GB total), so it
is the quickest way to get a first image. See [Supported Models](#supported-models)
for Krea 2 (larger, higher quality) and Z-Image-Turbo (distilled 8-step).

### 5. Generate an image

```bash
./thenoise.sh generate \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --prompt "a fox walking in the snow" --steps 8 --guidance-scale 1 \
  --out fox.png
```

The first generation is slow because the DiT is compiled with `torch.compile` —
see [Performance](#performance). To serve the same model over HTTP with a web UI
instead, use `serve` (see [CLI](#cli)).

## Portable (self-contained) builds

For machines without a dev toolchain, CI publishes **portable** bundles — one
directory with a standalone CPython, PyTorch ROCm, all dependencies, `thenoise`
itself, and a bundled `clang` (so `torch.compile`/Triton JIT works with no system
gcc). No installation, sudo, or Python needed on the target machine.

- Built per GPU target (`gfx1151`, `gfx1150`, `gfx1152`) — see
  `.github/workflows/build-thenoise-rocm.yml` and `scripts/build_portable.sh`.
- The `gfx1151` bundle is GPU-qualified (a real Anima generation) before release.

Download the release assets for your GPU and run:

```bash
# extract (split archives come as .partNN-of-MM.tar.gz — concatenate them first:
#   cat *.part*.tar.gz | tar -xz
# single archive: tar -xzf <tag>.tar.gz
./bin/thenoise generate \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --prompt "a fox walking in the snow" --steps 8 --guidance-scale 1
```

### Developing on TheNoise
`thenoise.sh` installs the runtime dependencies only. To run the test suite,
also install the dev extras:

```bash
uv pip install -e ".[dev]"
.venv/bin/python -m pytest tests/ -q
```

To use the model download scripts, install the `scripts` extra instead (or in
addition):

```bash
uv pip install -e ".[scripts]"
```

---

## Performance

**First-run compilation:** the DiT model is compiled with `torch.compile` on load. The first
generation will be noticeably slower while the inductor traces and compiles kernels. 
You will also see some warnings on the console, these are normal.
All subsequent generations use the cached compiled code and run at full speed. 
Compilation is transparent — no configuration needed.

### Troubleshooting: `InductorError` / `Python.h: No such file or directory`

If the first generation aborts with an `InductorError` wrapping a `gcc` failure
that references `-I/usr/include/python3.13`, the venv was built against a
**system** Python 3.13 whose development headers are not installed. Triton
JIT-compiles its HIP driver module at runtime and needs `Python.h`.

`thenoise.sh` avoids this by passing `--managed-python`, so uv uses its own
standalone CPython build (which always ships headers). If you have a venv
created before that fix, rebuild it:

```bash
rm -rf .venv
./thenoise.sh --help
```

Installing your distro's `python3.13-dev` package also works, if you would
rather keep the system interpreter.

---

## Supported Models

Anima, Krea 2, and Z-Image-Turbo are supported. New models will be added. PRs adding model support are welcome.

All download commands use `.venv/bin/python` and need the `scripts` extra
installed (`uv pip install -e ".[scripts]"`), because `huggingface_hub` lives
in the project venv created by [Setup](#setup) — a bare `python` will not work.

| Model | Download size | Notes |
|-------|---------------|-------|
| Anima | ~5.4 GB | 2B params; fastest to download and run |
| Krea 2 | ~35 GB | Higher quality; much larger text encoder and DiT |
| Z-Image-Turbo | ~21 GB | Distilled 8-step S3-DiT; Flux VAE + Qwen3 caption encoder |
| Z-Image | ~21 GB | Non-distilled version of Z-Image-Turob |

### Krea 2

Download:

```bash
.venv/bin/python scripts/download_krea2.py --out ./models/krea2
```

This fetches the bf16 Turbo DiT (~26 GB), the VAE (~0.25 GB), and the Qwen3-VL
text encoder (~8.9 GB). Add `--include-raw` for the non-turbo DiT (another
~26 GB).

### Anima

Download — the `--variant` you pick becomes part of the DiT filename, so use the
same value in your `--dit` path:

```bash
.venv/bin/python scripts/download_anima.py --out ./models/anima --variant turbo-v1.0
```

Available variants include `turbo-v1.0` (fewest steps), `aesthetic-v1.1`, and
`base-v1.0`.

### Z-Image-Turbo

```bash
.venv/bin/python scripts/download_zimage.py --out ./models/zimage
```

This fetches the single-file bf16 Turbo DiT (~12 GB), the Flux VAE (`ae.safetensors`),
and the Qwen3-4B text encoder (`qwen_3_4b.safetensors`, ~8 GB).

```bash
./thenoise.sh generate \
  --dit ./models/zimage/split_files/diffusion_models/z_image_turbo_bf16.safetensors \
  --vae ./models/zimage/split_files/vae/ae.safetensors \
  --text-encoder ./models/zimage/split_files/text_encoders/qwen_3_4b.safetensors \
  --prompt "a fox walking in the snow" \
  --out /tmp/zimage.png
```

---

## Operation Modes

TheNoise can be used in three ways:

1. **CLI** — generate a single image from the command line
2. **HTTP server** — serve a model over HTTP with a JSON API
3. **Web UI** — a very basic browser interface served at `http://localhost:8000/` when running the server

The model type is **auto-detected** from the DiT checkpoint — no need to specify which model you are using.

---

## CLI

### Serve a model over HTTP

Anima (matches the model downloaded in [Setup](#setup)):

```bash
./thenoise.sh serve \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --host 127.0.0.1 --port 8000
```

Krea 2:

```bash
./thenoise.sh serve \
  --dit ./models/krea2/diffusion_models/krea2_turbo_bf16.safetensors \
  --vae ./models/krea2/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/krea2/text_encoders/qwen3vl_4b_bf16.safetensors \
  --host 127.0.0.1 --port 8000
```

```bash
./thenoise.sh serve \
  --dit ./models/zimage/split_files/diffusion_models/z_image_turbo_bf16.safetensors \
  --vae ./models/zimage/split_files/vae/ae.safetensors \
  --text-encoder ./models/zimage/split_files/text_encoders/qwen_3_4b.safetensors \
  --host 127.0.0.1 --port 8000
```

Then open `http://localhost:8000/` for the web UI.

### Generate a single image

```bash
./thenoise.sh generate \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --prompt "a fox walking in the snow" --steps 8 --guidance-scale 1 \
  --out /tmp/fox.png
```

### Load LoRAs

Place `.safetensors` LoRA files in a directory and point `--lora-dir` at it (both `serve` and `generate`). Then apply LoRAs per-request:

```bash
./thenoise.sh generate \
  --dit ... --vae ... --text-encoder ... \
  --lora-dir ./models/loras \
  --prompt "a cyberpunk cityscape" \
  --lora "style-cyberpunk:0.8" \
  --lora "sub/detail-booster:0.5" \
  --out /tmp/city.png
```

LoRA format is `filename:weight` — the `.safetensors` extension is appended automatically. Omit `:weight` to use the default of `1.0`. LoRAs are switched in-memory without reloading the base model.

---

## HTTP API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/gallery` | List images in the persistent gallery, if available |
| `GET` | `/health` | Server status and loaded model |
| `GET` | `/lora` | List available LoRA names |
| `GET` | `/upscalers` | List available pixel upscaler names |
| `POST` | `/text2image` | Generate an image |

### `/text2image` request body

All fields except `prompt` are optional. Omitted fields use the loaded model's defaults.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `string` | *(required)* | Text prompt |
| `negative_prompt` | `string` | `""` | Negative prompt |
| `width` | `int` | model default | Output width in pixels |
| `height` | `int` | model default | Output height in pixels |
| `steps` | `int` | model default | Number of denoising steps |
| `guidance_scale` | `float` | model default | CFG scale (≤ 1.0 disables CFG) |
| `seed` | `int` | random | Random seed (`-1` for random) |
| `upscale` | `bool` | `false` | 2× latent-space upscale with refine denoise |
| `upscale_factor` | `float` | `1.0` | Upscale factor (max depends on the pixel upscaler scale) |
| `upscale_type` | `string` | `refined` | `refined` (latent 2x + refiner) or `no-refiner` (pixel upscaler only) |
| `pixel_upscaler` | `string` | `null` | Pixel upscaler name (no `.safetensors` suffix) from `--upscaler-dir` |
| `sampler` | `string` | `er_sde` | Denoising solver: `euler` or `er_sde` |
| `qwen_vae_enhance` | `bool` | `false` | Nyquist notch post-filter (removes 2px grid artifacts) |
| `film_grain` | `float` | `0.0` | Film grain strength, 0.0–10.0 |
| `sharpening` | `float` | `0.0` | RCAS sharpening strength, 0.0–1.0 |
| `lora_specs` | `string[]` | `null` | LoRA specs, e.g. `["style:0.8"]` |

### Response

Returns a PNG image directly (`Content-Type: image/png`).

### Example

```bash
curl -s localhost:8000/text2image \
  -H 'content-type: application/json' \
  -d '{"prompt":"a fox walking in the snow","steps":8}' \
  --output /tmp/fox.png
```

If no model is loaded, `/text2image` returns HTTP 503.

---

## CLI Parameters Reference

### Shared flags (`serve` and `generate`)

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--dit` | yes | — | Path to the DiT checkpoint (`.safetensors`) |
| `--vae` | yes | — | Path to the VAE checkpoint (`.safetensors`) |
| `--text-encoder` | yes | — | Path to the text encoder checkpoint (`.safetensors`) |
| `--lora-dir` | no | — | Directory containing LoRA `.safetensors` files |
| `--device` | no | `cuda` | Inference device (ROCm aliases `cuda` → `hip`) |

### `serve` only

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind host |
| `--port` | `8000` | Bind port |
| `--upscaler-dir` | — | Directory containing pixel upscaler `.safetensors` files (e.g. Real-ESRGAN); selected per-request via `pixel_upscaler` |
| `--gallery` | — | Directory to use as a persistent image store. If enabled, it also displays previously saved images |

### `generate` only

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--prompt` | yes | — | Text prompt |
| `--negative-prompt` | no | `""` | Negative prompt |
| `--width` | no | model default | Output width |
| `--height` | no | model default | Output height |
| `--steps` | no | model default | Denoising steps |
| `--guidance-scale` | no | model default | CFG scale |
| `--seed` | no | random | Random seed |
| `--out` | no | `out.png` | Output file path |
| `--lora` | no | — | LoRA to apply (repeatable, format: `file:weight`) |
| `--pixel-upscaler` | no | — | Full path to the pixel upscaler model (one-shot; e.g. a Real-ESRGAN `.safetensors`) |
| `--upscale-type` | no | `refined` | `refined` or `no-refiner` |
| `--upscale` | no | off | 2× latent upscale with refine denoise |
| `--sampler` | no | `er_sde` | Solver: `euler` or `er_sde` |
| `--qwen-vae-enhance` | no | off | Nyquist notch post-filter |
| `--film-grain` | no | `0.0` | Film grain strength (0.0–10.0) |
| `--sharpening` | no | `0.0` | RCAS sharpening strength (0.0–1.0) |
