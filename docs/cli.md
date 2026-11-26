# CLI

TheNoise's command-line interface. The model type is **auto-detected** from the
DiT checkpoint.

> Commands assume a dev checkout (`.venv/bin/python`, `./thenoise.sh`). On a
> [portable bundle](setup.md) use `./bin/thenoise` and `./bin/python3` instead.

## Operation modes

| Command | What it does | Model needed |
|---|---|---|
| `generate` | Generate one image from a text prompt | yes |
| `edit` | Edit an image from an instruction (image + prompt → edited image) | model that takes a reference |
| `video` | Generate one clip (text, or an image to start from) | video model |
| `mesh` | Reconstruct the shape in a picture (`.glb` / `.obj` / `.stl`) | mesh model |
| `upscale` | Pixel-upscale an existing image | no (upscaler only) |
| `controlmap` | Extract a control map (canny / depth / pose) from an image | no |
| `serve` | Serve a model over HTTP: JSON API + web UI at <http://localhost:8000/> | yes (lazily) |

## Shared flags (`serve`, `generate`, `edit`, `video`, `mesh`)

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--dit` | `generate` / `edit` / `video` | — | Path to the DiT checkpoint (`.safetensors`) |
| `--vae` | `generate` / `edit` / `video` | — | Path to the VAE checkpoint (`.safetensors`) |
| `--text-encoder` | `generate` / `edit` / `video` | — | Path to the text encoder checkpoint (`.safetensors`) |
| `--lora-dir` | no | — | Directory containing LoRA `.safetensors` files |
| `--device` | no | `cuda` | Inference device (ROCm aliases `cuda` → `hip`) |
| `--attention` | `generate` / `edit` / `video` / `mesh` | `auto` | How attention is computed: `auto` picks block-sparse or exact at every step, `full` is always exact, `optimized` is block-sparse even where it does not pay off yet. Only a model with block-sparse attention takes `optimized` — `/health` lists what the loaded model answers |

`serve` requires only `--dit`. Any further checkpoint is determined by the loaded
model, and named by flag if it is missing.

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
| `--dit-low` | — | Low-noise DiT of a two-expert model, e.g. Wan 2.2 A14B; `--dit` then selects the high-noise expert |
| `--audio-vae` | — | Audio VAE and vocoder checkpoint, for a release that ships them apart from the transformer |
| `--spatial-upscaler` | — | Spatial latent upscaler checkpoint, for a release that ships it apart |
| `--text-projection` | — | Text projection checkpoint, for a release that packs it apart from the transformer |
| `--host` | `127.0.0.1` | Bind host |
| `--port` | `8000` | Bind port |
| `--upscaler-dir` | — | Directory containing pixel upscaler `.safetensors` files (e.g. Real-ESRGAN); selected per-request via `pixel_upscaler` |
| `--controlmap-dir` | — | Directory of control-map checkpoints; each subdirectory's `config.json` declares the kind it implements |
| `--cache-dir` | `~/.cache/thenoise` | Where generated files are kept; queued jobs' images and clips land in `<cache-dir>/jobs` |

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
| `--width` | no | model default | Output width (omit for auto) |
| `--height` | no | model default | Output height (omit for auto) |
| `--steps` | no | model default | Denoising steps |
| `--guidance-scale` | no | model default | CFG scale (≤ 1.0 disables CFG) |
| `--seed` | no | random | Random seed |
| `--out` | no | `out.png` | Output file path |
| `--lora` | no | — | LoRA to apply (repeatable, format: `file:weight`) |
| `--pixel-upscaler` | no | — | Full path to the pixel upscaler model (one-shot; e.g. a Real-ESRGAN `.safetensors`) |
| `--upscale-type` | no | `refined` | `refined` or `no-refiner` |
| `--upscale` | no | off | 2× latent upscale with refine denoise (legacy alias for `--upscale-type refined --upscale-factor 2`) |
| `--upscale-factor` | no | `1.0` | Upscale factor (> 0.0; max depends on the pixel upscaler scale, see [Upscaling](#upscaling)) |
| `--sampler` | no | model default | Solver: `er_sde`, `euler` or `unipc` |
| `--qwen-vae-enhance` | no | off | Nyquist notch post-filter (removes 2px grid artifacts) |
| `--saturation` | no | `0.0` | Colour saturation around neutral: -1.0 greyscales, 1.0 doubles (-1.0–1.0) |
| `--vibrance` | no | `0.0` | Saturation that lifts washed-out pixels and leaves vivid ones alone (-1.0–1.0) |
| `--temperature` | no | `0.0` | White balance: -1.0 cools to blue, 1.0 warms to amber (-1.0–1.0) |
| `--tint` | no | `0.0` | White balance across the green–magenta axis: -1.0 greens, 1.0 magentas (-1.0–1.0) |
| `--sharpening` | no | `0.0` | RCAS sharpening strength (0.0–1.0) |
| `--film-grain` | no | `0.0` | Film grain strength (0.0–10.0) |
| `--kv-cache` / `--no-kv-cache` | no | auto | Reuse step-invariant K/V across denoising steps. T2I defaults off (opt-in); compatible edits auto-enable for `index_timestep_zero` |
| `--ref-method` | no | auto | How a reference is used for editing: `index`, `index_timestep_zero` (reference tokens conditioned at timestep zero, which is what makes the KV cache valid) or `vision` (the text encoder reads the picture instead of the VAE, so nothing is repainted). Auto = detected from the checkpoint, else the model's first listed method. The cache default follows it |

> **Note:** T2I caching is supported by models with a cacheable text prefix;
> other cache-capable models may accept the flag without a T2I speedup.

## Edit an image

Edits an existing image from an instruction (image + prompt → edited image).
Requires an editing-capable model and shares all generation flags with `generate`.
`--image` is repeatable. The **first** image sets the output size when
`--width`/`--height` are omitted: its own aspect ratio at about a megapixel.

A model whose only way with a picture is the text encoder — MOSS-Image — lists
`vision` and nothing else, so `edit` is how you show it one.

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

## Generate a single clip

Renders one clip and writes it to `--out`. Needs a video model — the ones
supported today are in the [video models](../README.md#video-models) list — and a
two-expert variant takes `--dit-low` beside `--dit`. A model that renders sound
takes the component flags above wherever its release ships those parts as their
own files.

```bash
./thenoise.sh video \
  --dit ./models/wan22/split_files/diffusion_models/fastwan2.2_ti2v-5b_bf16.safetensors \
  --vae ./models/wan22/split_files/vae/wan2.2_vae.safetensors \
  --text-encoder ./models/wan22/split_files/text_encoders/umt5_xxl_fp16.safetensors \
  --prompt "a fox walking through snow" --frames 33 --steps 4 --guidance-scale 1 \
  --out /tmp/fox.mp4
```

Unlike `generate`, no flag of this one has a default: what each release ships is
the variant's to say, so a flag you leave out stays unset. The run logs the size,
frame count, frame rate, solver and seed it settled on before the first step, and
bars the solver's steps and the written frames as it goes.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--prompt` | yes | — | Text prompt |
| `--negative-prompt` | no | model default | Negative prompt (not sent at all at CFG ≤ 1.0) |
| `--width` / `--height` | no | model default | Clip size, as the model rounds it; with `--image` and neither given, the image's aspect ratio decides |
| `--frames` | no | model default | Frames, rounded onto the model's temporal grid |
| `--duration` | no | — | Clip length in seconds, an alternative to `--frames` |
| `--fps` | no | model default | Frame rate, both of the clip and of the container |
| `--steps` | no | model default | Denoising steps |
| `--guidance-scale` | no | model default | CFG; on a two-expert model, of the high-noise half |
| `--flow-shift` | no | model default | Schedule shift; a distilled LoRA wants the one it ships with |
| `--sampler` | no | model default | Denoising solver: `er_sde`, `euler` or `unipc` |
| `--seed` | no | random | Random seed |
| `--image` | no | — | image the clip starts on |
| `--end-image` | no | — | image the clip ends on; needs `--image`, and a model that can be told where a clip ends ([FLF2V](models/wan22.md#first-and-last-frame-flf2v)) |
| `--reference` | no | — | something for the clip to look like rather than a frame it starts or ends on; repeatable. A picture, a clip or a sound — the file is asked which. `path@FRAME` pins it to that frame instead, counted from the end when negative. Needs a model that answers `ref2v` ([references](models/h3-video.md#references)) |
| `--lora` | no | — | LoRA for the primary transformer; repeatable, `file:weight` |
| `--lora-low` | no | — | LoRA for the low-noise expert of a `--dit-low` model; a LoRA meant for the whole clip goes to both flags |
| `--output-format` | no | model default | `mp4` or `webm`; `--out`'s extension answers when this is left out |
| `--out` | no | `out.mp4` | Output clip path; with no extension, the container the clip was written in |

Waiting for a clip is the difference from the job API: queue the same render with
`POST /jobs/video` instead — see [HTTP API](api.md#video-generation).

## Generate a single shape

Reconstructs the surface in one picture and writes it to `--out` as `.glb`, `.obj` or
`.stl`. Needs a mesh model — [hunyuan3d-2.1](models/hunyuan3d-2.1.md) is the one supported today
— whose transformer, conditioner and codec live in one file, so `--dit` is the only path
flag.

```bash
./thenoise.sh mesh \
  --dit ./models/hunyuan3d-2.1/hunyuan_3d_v2.1.safetensors \
  --image ./chair.png --out /tmp/chair.glb
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--image` | yes | — | The picture whose shape to reconstruct. Repeat it for the views of one object, which only a model with a multi-view tower accepts |
| `--latents` | no | model default | Latent tokens to denoise |
| `--steps`, `--guidance-scale`, `--sampler`, `--seed` | no | model default | Sampling parameters, as for `generate` |
| `--octree-resolution` | no | model default | Finest voxel grid the surface is cut from; the knob that costs time and memory, and 768 is the ceiling |
| `--bounds` | no | model default | Half-width of the sampled volume, in latent units |
| `--mc-level` | no | model default (`1/512`) | The field value the surface is cut at |
| `--scale` | no | `1.0` | Multiply the written coordinates: a mesh carries no unit, so a print sizes itself here |
| `--floaters` | no | `0.0` | Throw away pieces of the shape smaller than this fraction of its widest part; `0` keeps everything |
| `--triangles` | no | `1.0` | Keep this fraction of the surface detail; less gives a smaller file and a coarser shape |
| `--smooth` | no | `0.0` | Ease out the roughness the cut leaves, on whatever the detail level left; corners are held, so a hard edge stays hard |
| `--lora` | no | — | LoRA for the shape transformer; repeatable, `file:weight` |
| `--output-format` | no | model default | `glb`, `obj` or `stl`; `--out`'s extension answers when this is left out |
| `--out` | no | `out.glb` | Output mesh path; with no extension, the container the mesh was written in |

Queue the same render with `POST /jobs/mesh` — see [HTTP API](api.md#mesh-generation).

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

## Control-map a control map

Extracts a conditioning map - Canny edges, a Depth-Anything-V2 depth map or an
openpose-style pose skeleton - from an existing image. Model-free.

```bash
./thenoise.sh controlmap \
  --input /tmp/photo.jpg --type canny --out /tmp/photo_canny.png
./thenoise.sh controlmap \
  --input /tmp/photo.jpg --type depth --out /tmp/photo_depth.png \
  --controlmap-dir ./models/controlmaps
./thenoise.sh controlmap \
  --input /tmp/photo.jpg --type pose --out /tmp/photo_pose.png \
  --controlmap-dir ./models/controlmaps
```

Depth needs its checkpoint and pose needs two (a keypoint model and a person detector):
see [Control-map extraction](controlmap.md#installing-and-downloading-what-each-kind-needs).

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--input` | yes* | — | Input image to extract from (*unless `--list`) |
| `--type` | yes* | — | Control map to extract; repeatable, each extra type writes its own `<out>_<type>.png` |
| `--model` | no | type default | Implementation to use for that type |
| `--controlmap-dir` | no | — | Directory of control-map checkpoints (e.g. `./models/controlmaps`); each subdirectory's `config.json` declares the kind it implements |
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
| `--pose-background` | no | `black` | Pose: `black` or `green` (the canvas SD's original openpose map used) |
| `--out` | no | `out_control.png` | Output image path |

The `--<type>-<option>` flags, the `/controlmap` request fields and the web UI form all
come from the same declaration.

The map is white edges on black (`--type canny`), dark lines on white
(`--type lineart`), soft dark edges on white (`--type softedge`), a grey or turbo
rainbow depth field (`--type depth`) or a coloured skeleton (`--type pose`), at the
size of the image it was extracted from, so it can be paired with that image later. It
is written as a PNG with a `controlmap_data` text chunk recording the type, the model
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
