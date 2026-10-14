# CLI

TheNoise's command-line interface. The model type is **auto-detected** from the
DiT checkpoint.

> Commands assume a dev checkout (`.venv/bin/python`, `./thenoise.sh`). On a
> [portable bundle](setup.md) use `./bin/thenoise` and `./bin/python3` instead.

## Operation modes

| Command | What it does | Model needed |
|---|---|---|
| `generate` | Generate one image from a text prompt | yes |
| `edit` | Edit an image from an instruction (image + prompt → edited image) | editing-capable model |
| `upscale` | Pixel-upscale an existing image | no (upscaler only) |
| `preprocess` | Extract a control map (canny / depth / pose) from an image | no |
| `serve` | Serve a model over HTTP: JSON API + web UI at <http://localhost:8000/> | yes (lazily) |

## Shared flags (`serve`, `generate`, `edit`)

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--dit` | yes | — | Path to the DiT checkpoint (`.safetensors`) |
| `--vae` | yes | — | Path to the VAE checkpoint (`.safetensors`) |
| `--text-encoder` | yes | — | Path to the text encoder checkpoint (`.safetensors`) |
| `--lora-dir` | no | — | Directory containing LoRA `.safetensors` files |
| `--device` | no | `cuda` | Inference device (ROCm aliases `cuda` → `hip`) |

## Serve a model over HTTP

```bash
./thenoise.sh serve \
  --dit ./models/krea2/diffusion_models/krea2_turbo_bf16.safetensors \
  --vae ./models/krea2/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/krea2/text_encoders/qwen3vl_4b_bf16.safetensors \
  --host 127.0.0.1 --port 8000
```

Then open <http://localhost:8000/> for the web UI. Full endpoint reference:
[HTTP API](api.md).

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Bind host |
| `--port` | `8000` | Bind port |
| `--upscaler-dir` | — | Directory containing pixel upscaler `.safetensors` files (e.g. Real-ESRGAN); selected per-request via `pixel_upscaler` |

## Generate a single image

```bash
./thenoise.sh generate \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --prompt "a fox walking in the snow" --steps 8 --guidance-scale 1 \
  --out /tmp/fox.png
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--prompt` | yes | — | Text prompt |
| `--negative-prompt` | no | `""` | Negative prompt |
| `--width` | no | model default | Output width (1..4096; omit for auto) |
| `--height` | no | model default | Output height (1..4096; omit for auto) |
| `--steps` | no | model default | Denoising steps |
| `--guidance-scale` | no | model default | CFG scale (≤ 1.0 disables CFG) |
| `--seed` | no | random | Random seed |
| `--out` | no | `out.png` | Output file path |
| `--lora` | no | — | LoRA to apply (repeatable, format: `file:weight`) |
| `--pixel-upscaler` | no | — | Full path to the pixel upscaler model (one-shot; e.g. a Real-ESRGAN `.safetensors`) |
| `--upscale-type` | no | `refined` | `refined` or `no-refiner` |
| `--upscale` | no | off | 2× latent upscale with refine denoise (legacy alias for `--upscale-type refined --upscale-factor 2`) |
| `--upscale-factor` | no | `1.0` | Upscale factor (> 0.0; max depends on the pixel upscaler scale, see [Upscaling](#upscaling)) |
| `--sampler` | no | model default | Solver: `euler` or `er_sde` |
| `--qwen-vae-enhance` | no | off | Nyquist notch post-filter (removes 2px grid artifacts) |
| `--film-grain` | no | `0.0` | Film grain strength (0.0–10.0) |
| `--sharpening` | no | `0.0` | RCAS sharpening strength (0.0–1.0) |
| `--kv-cache` / `--no-kv-cache` | no | auto | Reference-latent KV cache (edit only): freeze the reference tokens' K/V across denoise steps for faster editing. Auto = on for `index_timestep_zero` reference conditioning on models implementing the cache. `--kv-cache` also selects that method when `--ref-method` is left unset |
| `--ref-method` | no | auto | Reference conditioning for editing: `index` or `index_timestep_zero` (reference tokens conditioned at timestep zero, which is what makes the KV cache valid). Auto = detected from the checkpoint, else `index`. The cache default follows it |

> **Note:** `--kv-cache` is a reference-latent optimization and only applies to
> `edit` (it needs a reference image). On `generate` it raises an error.

## Edit an image

Edits an existing image from an instruction (image + prompt → edited image).
Requires an editing-capable model and shares all generation flags with `generate`.

`--image` is repeatable. The **first** image sets the output size when
`--width`/`--height` are omitted: 1024 on its largest side, aspect preserved.

```bash
./thenoise.sh edit \
  --dit ./models/klein/split_files/diffusion_models/flux-2-klein-4b.safetensors \
  --vae ./models/klein/split_files/vae/flux2-vae.safetensors \
  --text-encoder ./models/klein/split_files/text_encoders/qwen_3_4b.safetensors \
  --image /tmp/fox.png \
  --prompt "a fox wearing a red scarf" --steps 4 --sampler euler \
  --out /tmp/fox_edited.png
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--image` | yes | — | Input image(s) to edit; repeatable for multiple reference images (first sets the output size) |
| `--out` | no | `out_edit.png` | Output file path |

## Upscale a single image

Pixel-upscales an existing image. Model-free - no `--dit`/`--vae`/
`--text-encoder` needed:

```bash
./thenoise.sh upscale \
  --pixel-upscaler ./models/esrgan/RealESRGAN_x4plus.safetensors \
  --input /tmp/fox.png --upscale-factor 4 \
  --out /tmp/fox_4x.png
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--pixel-upscaler` | yes | — | Full path to the pixel upscaler model (e.g. a Real-ESRGAN `.safetensors`) |
| `--input` | yes | — | Input image to upscale |
| `--upscale-factor` | no | `0.0` | Upscale factor (`0.0` = the model's detected scale; must be in [1, that scale]; larger values are rejected) |
| `--out` | no | `out_upscaled.png` | Output image path |

## Preprocess a control map

Extracts a conditioning map - Canny edges, a Depth-Anything-V2 depth map or an
openpose-style pose skeleton - from an existing image. Model-free.

```bash
./thenoise.sh preprocess \
  --input /tmp/photo.jpg --type canny --out /tmp/photo_canny.png
./thenoise.sh preprocess \
  --input /tmp/photo.jpg --type depth --out /tmp/photo_depth.png \
  --preprocessor-dir ./models/preprocessors
./thenoise.sh preprocess \
  --input /tmp/photo.jpg --type pose --out /tmp/photo_pose.png \
  --preprocessor-dir ./models/preprocessors
```

Depth needs its checkpoint and pose needs two (a keypoint model and a person detector):
see [Control-map extraction](preprocess.md#installing-and-downloading-what-each-kind-needs).

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input` | yes* | — | Input image to extract from (*unless `--list`) |
| `--type` | yes* | — | Control map to extract; repeatable, each extra type writes its own `<out>_<type>.png` |
| `--model` | no | type default | Implementation to use for that type |
| `--preprocessor-dir` | no | — | Directory of control-map checkpoints (e.g. `./models/preprocessors`); each subdirectory's `config.json` declares the kind it implements |
| `--list` | no | — | List the available control-map types, and what a half-installed kind still needs, then exit |
| `--canny-low` | no | `100` | Canny low threshold, in Sobel magnitude units |
| `--canny-high` | no | `200` | Canny high threshold; weak edges connected to a strong one are kept |
| `--canny-thickness` | no | `1` | Rendered edge thickness in pixels |
| `--lineart-threshold` | no | `0.5` | Line-art: line intensity threshold, 0-1 |
| `--lineart-thickness` | no | `1` | Line-art: line thickness in pixels |
| `--softedge-threshold` | no | `0.2` | Soft-edge: where the gradient starts to count, 0-1 |
| `--softedge-blur` | no | `1.5` | Soft-edge: smoothing sigma in pixels, 0 = none |
| `--depth-invert` / `--no-depth-invert` | no | `false` | Depth: swap near/far (near is bright by default) |
| `--depth-colormap` | no | `gray` | Depth: `gray` or `turbo` (a vivid rainbow) |
| `--pose-detector` | no | first found | Pose: which RT-DETR checkpoint to crop people with |
| `--pose-threshold` | no | `0.5` | Pose: minimum detector score for a person to be in the map at all |
| `--pose-kp-threshold` | no | `0.3` | Pose: minimum confidence for a joint to be drawn |
| `--pose-line-width` | no | `4` | Pose: skeleton bone thickness in pixels |
| `--pose-radius` | no | `4` | Pose: radius of the joint dots |
| `--pose-background` | no | `black` | Pose: `black` or `green` (the canvas SD's original openpose preprocessor used) |
| `--out` | no | `out_control.png` | Output image path |

The `--<type>-<option>` flags are generated from the same option dataclasses the `/preprocess` endpoint and the web UI form are built from so a flag and a parameter of the map it produces cannot disagree, and a flag you do not pass is not sent as its own default.

The map is white edges on black (`--type canny`), dark lines on white
(`--type lineart`), soft dark edges on white (`--type softedge`), a grey or turbo
rainbow depth field (`--type depth`) or a coloured skeleton (`--type pose`), at the
size of the image it was extracted from, so it can be paired with that image later. It
is written as a PNG with a `preprocess_data` text chunk recording the type, the model
and every option that differed from its default, alongside whatever metadata the input
already carried.

## Upscaling

TheNoise supports up to 8× upscaling through two complementary mechanisms. Both
are optional and can be combined.

### Latent upscale + refiner (`refined`)

Every model ships a built-in **latent upscaler** that runs in latent
space before the VAE decode: it upscales the latent 2× and then runs a short,
low-strength refine denoise at the upscaled size. This is the default
`upscale_type` and needs **no extra model files** - a 2× upscale works out of the
box on any supported model.

```bash
./thenoise.sh generate \
  --dit ... --vae ... --text-encoder ... \
  --prompt "a fox walking in the snow" \
  --upscale-type refined --upscale-factor 2 \
  --out fox_2x.png
```

### Pixel-domain upscaler (`no-refiner`, and beyond 2×)

Pixel upscaling operates purely in pixel space (after decode) and uses a
dedicated upscaler model, Real-ESRGAN. It is **not** a model concern:
the upscaler directory is server configuration (`--upscaler-dir`), and the named
model is selected per-request via `pixel_upscaler`. Only the last-used upscaler
is kept loaded (switched on change).

```bash
# download the optional Real-ESRGAN x4 pixel upscaler (portable bundle:
# ./bin/python3 scripts/download.py --model esrgan)
.venv/bin/python scripts/download.py --model esrgan
```

A pixel upscaler is **required** for `no-refiner` mode (pixel upscaler only, no
latent 2×), and for `refined` factors above the latent 2×. Without one, only
`refined` factors up to 2× are available.

The max factor follows the detected upscaler scale. For a 4× Real-ESRGAN model:
`no-refiner` is limited to 4×, and `refined` to 2× (latent) × 4× (pixel) = 8×.

```bash
# generate with an 8x total upscale (2x latent refine + 4x pixel)
./thenoise.sh generate \
  --dit ... --vae ... --text-encoder ... \
  --prompt "a fox walking in the snow" \
  --upscale-type refined --upscale-factor 8 \
  --pixel-upscaler ./models/esrgan/RealESRGAN_x4plus.safetensors \
  --out fox_8x.png
```

## LoRAs

Place `.safetensors` LoRA files in a directory and point `--lora-dir` at it
(both `serve` and `generate`). Then apply LoRAs per-request:

```bash
./thenoise.sh generate \
  --dit ... --vae ... --text-encoder ... \
  --lora-dir ./models/loras \
  --prompt "a cyberpunk cityscape" \
  --lora "style-cyberpunk:0.8" \
  --lora "sub/detail-booster:0.5" \
  --out /tmp/city.png
```

LoRA format is `filename:weight` - the `.safetensors` extension is appended
automatically. Omit `:weight` to use the default of `1.0`. LoRAs are switched
in-memory without reloading the base model.

On `serve`, available LoRAs are listed at `GET /lora` and requested per-call
via `lora_specs` (see [HTTP API](api.md)).

## First run

The DiT model is compiled with `torch.compile` on load. The first generation
will be noticeably slower while the inductor traces and compiles kernels, and
you will see some warnings on the console — these are normal. All subsequent
generations use the cached compiled code and run at full speed. Compilation is
transparent: no configuration needed.
