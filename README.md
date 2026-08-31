# TheNoise

A text-to-image diffusion inference engine. Tested on Strix Halo, Strix Point and Krackan Point.

Loads one model at a time and generates images from text prompts. Editing-capable models can also edit existing images from a text instruction (image + prompt → edited image). Available as a CLI tool, an HTTP API (with a simple web UI).

<details open>
  <summary>Generate tab</summary>
  <img width="2048" height="1066" alt="thenoise-main-screenshot" src="https://github.com/user-attachments/assets/afaf2d89-5857-4f50-995f-06fdf556a3c4" />
</details>
<details>
  <summary>Edit tab</summary>
  <img width="2048" height="1066" alt="thenoise-edit" src="https://github.com/user-attachments/assets/17efedda-b887-4f87-b0c0-c151619b19ac" />
</details>
<details>
  <summary>Upscale tab</summary> 
  <img width="2048" height="1066" alt="thenoise-upscaler" src="https://github.com/user-attachments/assets/f7ce89b7-fd25-4ad9-a3e4-d5e367530ab7" />
</details>

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

There is also a `download_civitai.py` script which downloads from `civitai.com`.
This script requires `CivitaPy` which is **not** installed by `thenoise.sh`. If
`CivitaPy` is not installed, you will be prompted to install it with
`uv pip install civitapy`. To download from Civitai, you will also need an API
token, stored in the `CIVITAI_TOKEN` environment variable.

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

Anima, Krea 2, Z-Image-Turbo, Flux.2 Klein, and SDXL are supported. New models will be added. PRs adding model support are welcome.

All download commands use `.venv/bin/python` and need the `scripts` extra
installed (`uv pip install -e ".[scripts]"`), because `huggingface_hub` lives
in the project venv created by [Setup](#setup) — a bare `python` will not work.

| Model | Download size | Notes | Editing |
|-------|---------------|-------|------|
| Anima | ~5.4 GB | 2B params; fastest to download and run | x |
| Krea 2 | ~35 GB | Higher quality; much larger text encoder and DiT | x |
| Z-Image-Turbo | ~21 GB | Distilled 8-step S3-DiT; Flux VAE + Qwen3 caption encoder | x |
| Z-Image | ~21 GB | Non-distilled version of Z-Image-Turbo | x |
| Flux.2 Klein 4B | ~12 GB | Distilled 4-step flow MMDiT; Flux.2 VAE + Qwen3-4B | ✓ |
| Flux.2 Klein 9B | ~25 GB | Distilled 4-step flow MMDiT; Flux.2 VAE + Qwen3-8B | ✓ |
| SDXL | ~6.9 GB | Stable Diffusion XL (incl. anime fine-tunes like Illustrious-XL); CLIP-L + CLIP-G text encoders, discrete euler | x |

### Krea 2

Download:

```bash
.venv/bin/python scripts/download_krea2.py --out ./models/krea2
```

This fetches the bf16 Turbo DiT (~26 GB), the VAE (~0.25 GB), and the Qwen3-VL
text encoder (~8.9 GB). Add `--include-raw` for the non-turbo DiT (another
~26 GB), and `--int8-convrot` to fetch the int8-convrot DiT(s) instead of bf16.

### Anima

Download — the `--variant` you pick becomes part of the DiT filename, so use the
same value in your `--dit` path:

```bash
.venv/bin/python scripts/download_anima.py --out ./models/anima --variant turbo-v1.0
```

Available variants include `turbo-v1.0` (fewest steps), `aesthetic-v1.1`, and
`base-v1.0`. Add `--int8-convrot` to fetch the int8-convrot DiT (from
`Bedovyy/Anima-INT8`) instead of the bf16 one.

### Z-Image-Turbo

```bash
.venv/bin/python scripts/download_zimage.py --out ./models/zimage
```

This fetches the single-file bf16 Turbo DiT (~12 GB), the Flux VAE (`ae.safetensors`),
and the Qwen3-4B text encoder (`qwen_3_4b.safetensors`, ~8 GB). Add
`--int8-convrot` to fetch the int8-convrot DiT instead.

```bash
./thenoise.sh generate \
  --dit ./models/zimage/split_files/diffusion_models/z_image_turbo_bf16.safetensors \
  --vae ./models/zimage/split_files/vae/ae.safetensors \
  --text-encoder ./models/zimage/split_files/text_encoders/qwen_3_4b.safetensors \
  --prompt "a fox walking in the snow" \
  --out /tmp/zimage.png
```

### Flux.2 Klein

```bash
.venv/bin/python scripts/download_klein.py --out ./models/klein --variant 4b
```

This fetches the single-file bf16 DiT, the Flux.2 VAE (`flux2-vae.safetensors`),
and the Qwen3 text encoder (a single file from Comfy-Org). Pick `--variant` from
`4b` / `4b-base` / `9b` / `9b-base`. The 9B DiTs come from the official
black-forest-labs repos (they are not published as single files elsewhere).

Add `--int8-convrot` to fetch the int8-convrot DiT instead. The 4B DiT comes from
`wraps/FLUX.2-klein-4B-INT8-ConvRot-ComfyUI`; the 9B DiT is only published on
Civitai and is downloaded directly from there (if Civitai requires login, the
script prints the model page link). Base variants have no int8-convrot release.

The DiT size (4B vs 9B) is auto-detected from the checkpoint; the matching Qwen3
text encoder is selected automatically. The distilled variants (default) run 4
steps with CFG off (`guidance_scale` 1.0). Base variants need explicit CFG:

```bash
./thenoise.sh generate \
  --dit ./models/klein/split_files/diffusion_models/flux-2-klein-4b.safetensors \
  --vae ./models/klein/split_files/vae/flux2-vae.safetensors \
  --text-encoder ./models/klein/split_files/text_encoders/qwen_3_4b.safetensors \
  --prompt "a fox walking in the snow" \
  --steps 4 \
  --sampler euler \
  --out /tmp/klein.png
```

For a *base* checkpoint (`-base-4b` / `-base-9b`) use `--steps 50 --guidance-scale 4`
and pass a `--negative-prompt`. The default sampler is Euler; ER-SDE is also
selectable via `--sampler er_sde`.

### SDXL

Stable Diffusion XL and its variants (Illustrious, Juggernaut, Pony, Noob ...)
load with a single adapter; the model type is auto-detected from the checkpoint.
The prediction type (epsilon vs v-prediction) is likewise autodetected from
marker tensors in the checkpoint. SDXL uses the `euler` sampler only.

**Single-file checkpoints.** Many SDXL mixes on Civitai ship as one combined
`.safetensors`. Point `--checkpoint` straight at that file and thenoise
loads the DiT, VAE, and text encoders from it in memory — no splitting needed:

```bash
.venv/bin/python scripts/download_civitai.py --profile sdxl
./thenoise.sh generate \
  --checkpoint ./models/mix.safetensors \
  --prompt "a fox walking in the snow" --steps 28 --guidance-scale 5.5 \
  --out /tmp/sdxl.png
```

Or download and split into the DiT/VAE/text-encoder trio:

```bash
.venv/bin/python scripts/download_sdxl.py --out ./models/sdxl
./thenoise.sh generate \
  --dit ./models/sdxl/split_files/diffusion_models/sdxl_unet.safetensors \
  --vae ./models/sdxl/split_files/vae/sdxl_vae.safetensors \
  --text-encoder ./models/sdxl/split_files/text_encoders/clip_l_g.safetensors \
  --prompt "a fox walking in the snow" --steps 28 --guidance-scale 5.5 \
  --out /tmp/sdxl.png
```

`--checkpoint` is mutually exclusive with `--dit`/`--vae`/`--text-encoder`.

Some SDXL fine-tunes (e.g. noobai) are trained with a zero-terminal-SNR schedule;
pass `--sd-zsnr` if a checkpoint renders garbage. It is auto-enabled when the
checkpoint carries the `ztsnr` marker. Some models have this flag incorrectly
set, so it can be disabled with `--no-sd-zsnr`.

---

## Operation Modes

TheNoise can be used in several ways:

1. **CLI `generate`** — generate a single image from the command line
2. **CLI `edit`** — edit an existing image from an instruction (image + prompt → edited image)
3. **CLI `upscale`** — pixel-upscale an existing image (no diffusion model needed)
4. **HTTP server** — serve a model over HTTP with a JSON API (text-to-image, editing, upscaling)
5. **Web UI** — a very basic browser interface served at `http://localhost:8000/` when running the server

The model type is **auto-detected** from the DiT checkpoint — no need to specify which model you are using.

---

## Upscaling

TheNoise supports up to 8× upscaling through two complementary mechanisms. Both are optional and can be combined.

### Latent upscale + refiner (`refined`)

Every model ships a built-in **latent (SesquiLSR) upscaler** that runs in latent space before the VAE decode, upscaling the latent 2× and then running a short, low-strength refine denoise at the upscaled size. This is the default `upscale_type` and needs **no extra model files** — a 2× upscale works out of the box on any supported model.

### Pixel-domain upscaler (`no-refiner`, and beyond 2×)

Pixel upscaling operates purely in pixel space (after decode) and uses a dedicated upscaler model — today Real-ESRGAN. It is **not** a model concern: the upscaler directory is server configuration (`--upscaler-dir`), and the named model is selected per-request via `pixel_upscaler`. Only the last-used upscaler is kept loaded (switched on change).

A pixel upscaler is **required** for `no-refiner` mode (pixel upscaler only, no latent 2×), and for `refined` factors above the latent 2×. Without one, only `refined` factors up to 2× are available.

The max factor follows the detected upscaler scale. For a 4× Real-ESRGAN model: `no-refiner` is limited to 4×, and `refined` to 2× (latent) × 4× (pixel) = 8×.

```bash
# download the optional Real-ESRGAN x4 pixel upscaler (needs the scripts extra)
.venv/bin/python scripts/download_esrgan.py --out ./models/esrgan
```

The downloaded `RealESRGAN_x4plus.safetensors` goes into an `--upscaler-dir` (serve) or is passed by full path via `--pixel-upscaler` (generate/upscale).

---

## Editing

Editing-capable models can edit an existing image from a text instruction: **image + prompt → edited image**.

Editing is a **model** capability (`supports_edit`). At the moment only Flux.2 Klein supports it.

You may provide one or many reference images. Without an explicit `width`/`height`, the **first** reference image is resized to 1024 on its largest side (aspect preserved) and sets the output size; the rest are used as additional references.

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

### Edit an image

 `--image` is repeatable — the first image sets the output size, the rest are additional references:

```bash
./thenoise.sh edit \
  --dit ./models/klein/split_files/diffusion_models/flux-2-klein-4b.safetensors \
  --vae ./models/klein/split_files/vae/flux2-vae.safetensors \
  --text-encoder ./models/klein/split_files/text_encoders/qwen_3_4b.safetensors \
  --image /tmp/fox.png \
  --prompt "a fox wearing a red scarf" --steps 4 --sampler euler \
  --out /tmp/fox_edited.png
```

### Upscale a single image

```bash
./thenoise.sh upscale \
  --pixel-upscaler ./models/esrgan/RealESRGAN_x4plus.safetensors \
  --input /tmp/fox.png --upscale-factor 4 \
  --out /tmp/fox_4x.png
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
| `GET` | `/health` | Server status and loaded model |
| `GET` | `/lora` | List available LoRA names |
| `GET` | `/upscalers` | List available pixel upscaler names (works even with no model loaded) |
| `POST` | `/upscale` | Pixel-upscale an input image (works even with no model loaded) |
| `POST` | `/text2image` | Generate an image |
| `POST` | `/edit` | Edit an image from an instruction (image + prompt → edited image); requires an editing-capable model |

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
| `sampler` | `string` | `er_sde` | Denoising solver: `euler` or `er_sde` (SDXL only supports euler; er_sde auto-falls back) |
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

### `/edit` request body

Instruction-based editing: image(s) + prompt → edited image. Requires an editing-capable model (Flux.2 Klein); otherwise returns HTTP 400.

Accepts all `/text2image` fields plus:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | `string` \| `string[]` | *(required)* | One or more base64-encoded reference images (OpenAI-style; first sets the output size when `width`/`height` omitted) |

### Example

```bash
curl -s localhost:8000/edit \
  -H 'content-type: application/json' \
  -d '{"image":"<base64 png>","prompt":"a fox wearing a red scarf","steps":4,"sampler":"euler"}' \
  --output /tmp/fox_edited.png
```

### `/upscale` request body

Pixel-upscales an existing image by `upscale_factor`× with a named pixel upscaler. Unlike `/text2image`, this needs no diffusion model loaded — only an upscaler configured via `--upscaler-dir`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image_b64` | `string` | *(required)* | Base64-encoded input image (PNG/JPEG) |
| `upscale_factor` | `float` | `0.0` | Desired final factor (`0.0` = the upscaler's detected native scale; must be in [1, that scale]; larger values are rejected) |
| `pixel_upscaler` | `string` | *(required)* | Pixel upscaler name (no `.safetensors` suffix) from `--upscaler-dir` |
| `out` | `string` | `png` | `png` (returns an image) or `json` (returns `b64_json`) |

### Response

Returns a PNG image directly (`Content-Type: image/png`).

### Example

```bash
curl -s localhost:8000/upscale \
  -H 'content-type: application/json' \
  -d '{"image_b64":"<base64 png>","pixel_upscaler":"RealESRGAN_x4plus","upscale_factor":4}' \
  --output /tmp/fox_4x.png
```

## CLI Parameters Reference

### Shared flags (`serve`, `generate` and `edit`)

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

### `generate` only

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--prompt` | yes | — | Text prompt |
| `--negative-prompt` | no | `""` | Negative prompt |
| `--width` | no | model default | Output width (0..4096) |
| `--height` | no | model default | Output height (0..4096) |
| `--steps` | no | model default | Denoising steps |
| `--guidance-scale` | no | model default | CFG scale |
| `--seed` | no | random | Random seed |
| `--out` | no | `out.png` | Output file path |
| `--lora` | no | — | LoRA to apply (repeatable, format: `file:weight`) |
| `--pixel-upscaler` | no | — | Full path to the pixel upscaler model (one-shot; e.g. a Real-ESRGAN `.safetensors`) |
| `--upscale-type` | no | `refined` | `refined` or `no-refiner` |
| `--upscale` | no | off | 2× latent upscale with refine denoise (legacy alias for `--upscale-type refined --upscale-factor 2`) |
| `--upscale-factor` | no | `1.0` | Upscale factor (> 0.0; max depends on the pixel upscaler scale, see [Upscaling](#upscaling)) |
| `--sampler` | no | `er_sde` | Solver: `euler` or `er_sde` (SDXL only supports euler; er_sde auto-falls back) |
| `--qwen-vae-enhance` | no | off | Nyquist notch post-filter |
| `--film-grain` | no | `0.0` | Film grain strength (0.0–10.0) |
| `--sharpening` | no | `0.0` | RCAS sharpening strength (0.0–1.0) |

### `edit` only

Edits an existing image from an instruction (image + prompt → edited image). Requires an editing-capable model (Flux.2 Klein) and shares all generation flags with `generate`.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--image` | yes | — | Input image(s) to edit; repeatable for multiple reference images (first sets the output size) |
| `--out` | no | `out_edit.png` | Output file path |

### `upscale` only

Pixel-upscales an existing image. Model-free — no `--dit`/`--vae`/`--text-encoder` needed.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--pixel-upscaler` | yes | — | Full path to the pixel upscaler model (e.g. a Real-ESRGAN `.safetensors`) |
| `--input` | yes | — | Input image to upscale |
| `--upscale-factor` | no | `0.0` | Upscale factor (`0.0` = the model's detected scale; must be in [1, that scale]; larger values are rejected) |
| `--out` | no | `out_upscaled.png` | Output image path |
