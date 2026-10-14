# HTTP API

Start the server with [`serve`](cli.md#serve-a-model-over-http) and TheNoise exposes a small JSON API plus the web UI at the root path.

> Commands assume a dev checkout (`./thenoise.sh`). On a [portable
> bundle](setup.md) use `./bin/thenoise` instead.

```bash
./thenoise.sh serve \
  --dit ./models/krea2/diffusion_models/krea2_turbo_bf16.safetensors \
  --vae ./models/krea2/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/krea2/text_encoders/qwen3vl_4b_bf16.safetensors \
  --host 127.0.0.1 --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Server status, loaded model and its capabilities (see below) |
| `GET` | `/lora` | List available LoRA names |
| `GET` | `/upscalers` | List available pixel upscaler names (works even with no model loaded) |
| `POST` | `/upscale` | Pixel-upscale an input image (works even with no model loaded) |
| `GET` | `/preprocessors` | List the extractable control-map types (works even with no model loaded) |
| `POST` | `/preprocess` | Extract a control map from an input image (works even with no model loaded) |
| `POST` | `/text2image` | Generate an image |
| `POST` | `/edit` | Edit an image from an instruction (image + prompt → edited image); requires an editing-capable model |

## `GET /health`

```json
{"status": "ok", "models": ["qwen_image"], "capabilities": {"edit": true, "kv_cache": true}}
```

`models` is empty until a DiT is loaded, and `capabilities` is the loaded
adapter's `CAPABILITIES` dict verbatim (`{}` with no model) - the web UI uses it
to gate the Edit tab and the KV-cache control, and a request asking for a
capability the model lacks is rejected with HTTP 400 rather than silently
ignored.

## `POST /text2image`

All fields except `prompt` are optional. An omitted field resolves as
**request → checkpoint marker → model default**: whatever you send always wins,
otherwise a marker in the loaded checkpoint may imply a value, and failing that
the model's own default applies (see
[`DiffusionModel.pref`](../thenoise/models/base.py)). Today the only marker
read is `__index_timestep_zero__`, which implies `ref_method:
index_timestep_zero` (see [`/edit`](#post-edit)).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt` | `string` | *(required)* | Text prompt |
| `negative_prompt` | `string` | `""` | Negative prompt |
| `width` | `int` | model default | Output width in pixels |
| `height` | `int` | model default | Output height in pixels |
| `steps` | `int` | model default | Number of denoising steps |
| `sigmas` | `float[]` | model default | Custom sigma grid, replacing `steps` (see [Custom sigmas](#custom-sigmas)) |
| `guidance_scale` | `float` | model default | CFG scale (≤ 1.0 disables CFG) |
| `seed` | `int` | random | Random seed (`-1` for random) |
| `upscale` | `bool` | `false` | 2× latent-space upscale with refine denoise |
| `upscale_factor` | `float` | `1.0` | Upscale factor (max depends on the pixel upscaler scale) |
| `upscale_type` | `string` | `refined` | `refined` (latent 2x + refiner) or `no-refiner` (pixel upscaler only) |
| `pixel_upscaler` | `string` | `null` | Pixel upscaler name (no `.safetensors` suffix) from `--upscaler-dir` |
| `sampler` | `string` | model default | Denoising solver: `euler` or `er_sde` |
| `qwen_vae_enhance` | `bool` | `false` | Nyquist notch post-filter (removes 2px grid artifacts) |
| `film_grain` | `float` | `0.0` | Film grain strength, 0.0–10.0 |
| `sharpening` | `float` | `0.0` | RCAS sharpening strength, 0.0–1.0 |
| `lora_specs` | `string[]` | `null` | LoRA specs, e.g. `["style:0.8"]` |

### Response

Returns a PNG image directly (`Content-Type: image/png`).

If no model is loaded, `/text2image` returns **HTTP 503**.

### Example

```bash
curl -s localhost:8000/text2image \
  -H 'content-type: application/json' \
  -d '{"prompt":"a fox walking in the snow","steps":8}' \
  --output /tmp/fox.png
```

### Custom sigmas

`sigmas` replaces `steps` with an explicit denoising schedule: a list of decreasing
noise levels in `[0.0, 1.0]`, where `1.0` is pure noise and `0.0` is clean. The values
are the same ones the model's own schedule produces, so they are used verbatim - no
resolution-dependent shift is applied on top.

- The trailing `0.0` is implied: `[1.0, 0.8, 0.5]` and `[1.0, 0.8, 0.5, 0.0]` are the
  same 3-step schedule (ComfyUI's `BasicScheduler` convention). The step count is
  always `len(sigmas) - 1`.
- The grid must be strictly decreasing and inside `[0.0, 1.0]`; anything else is
  rejected with **HTTP 400**.
- `sigmas` wins over `steps` in the same request. When the two disagree
  the server logs a warning and runs the grid.
- `sigmas: []` switches a default grid off explicitly.
- The upscale refine pass is unaffected: it keeps its own short, low-strength
  sub-schedule.
- The resolved grid (terminal `0.0` included) is recorded in the PNG `generation_data`
  chunk.

There is no CLI flag for `sigmas`: it is an API-only field.

```bash
curl -s localhost:8000/text2image \
  -H 'content-type: application/json' \
  -d '{"prompt":"a fox walking in the snow","sigmas":[1.0,0.85,0.6,0.35,0.15]}' \
  --output /tmp/fox.png
```

## `POST /edit`

Instruction-based editing: image(s) + prompt → edited image. Requires an
editing-capable model; otherwise returns **HTTP 400**.

Accepts all `/text2image` fields plus:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | `string` \| `string[]` | *(required)* | One or more base64-encoded reference images (OpenAI-style; first sets the output size when `width`/`height` omitted) |
| `kv_cache` | `boolean` | auto | Reference-latent KV cache (freeze the reference tokens' K/V across denoise steps for faster editing). Auto = on when the resolved `ref_method` is `index_timestep_zero` and the model implements the cache, off otherwise. Forcing it to `true` also selects that method when `ref_method` is left on auto |
| `ref_method` | `string` | auto | Reference conditioning: `index` or `index_timestep_zero` (reference tokens conditioned at timestep zero, which is what makes the KV cache valid). Auto = detected from the checkpoint, else `index`. The cache default follows it |

> **Reference method.** Checkpoints trained to condition their reference tokens
> at timestep zero carry the `__index_timestep_zero__` marker, which makes
> `ref_method` resolve to `index_timestep_zero` automatically — and the KV cache
> with it. An explicit `ref_method` always wins over detection: markers are only
> a hint, and some trained checkpoints do not carry one. An explicit `kv_cache`
> together with an explicit `ref_method: index` is rejected rather than silently
> degraded.

### Example

```bash
curl -s localhost:8000/edit \
  -H 'content-type: application/json' \
  -d '{"image":"<base64 png>","prompt":"a fox wearing a red scarf","steps":4,"sampler":"euler"}' \
  --output /tmp/fox_edited.png
```

## `POST /upscale`

Pixel-upscales an existing image by `upscale_factor`× with a named pixel
upscaler. Unlike `/text2image`, this needs no diffusion model loaded - only an
upscaler configured via `--upscaler-dir`.

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

## `GET /preprocessors`

```json
{
  "preprocessors": {"canny": ["canny"]},
  "defaults": {"canny": "canny"},
  "detectors": [],
  "options": {
    "canny": [
      {"name": "low", "label": "Low edge strength", "type": "number", "default": 100.0, "min": 0, "step": 1},
      {"name": "high", "label": "High edge strength", "type": "number", "default": 200.0, "min": 0, "step": 1},
      {"name": "thickness", "label": "Line width", "type": "integer", "default": 1, "min": 1, "step": 1}
    ]
  }
}
```

`preprocessors` maps each control-map type to the implementations it can choose
between, `defaults` to the one used when a request does not name one. Both are
empty when nothing is implemented, and no diffusion model is needed to list them.

`options` describes the request fields of every kind the engine knows - label,
control type, default, bounds - and `pool` names the list a field is filled from
(`detectors`, above). It is generated from the option dataclasses the extraction
itself runs on, which is also where the CLI flags and the web UI form come from, so
none of the three can show a default the engine would not use.

`detectors` lists the person detectors on disk that a `pose` request crops its people
with, by name. `pose` appears in `preprocessors` only once both its checkpoints - the
keypoint model *and* a detector - are in the directory: an estimator with nothing to
crop with measures nothing.

## `POST /preprocess`

Extracts a conditioning map - Canny or line-art edges, a soft edge field, a
Depth-Anything-V2 depth map or an openpose-style pose skeleton. Like `/upscale`, this
needs no diffusion model loaded.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image_b64` | `string` | *(required)* | Base64-encoded input image (PNG/JPEG) |
| `type` | `string` | *(required)* | Control-map type, one of those listed by `/preprocessors` |
| `model` | `string` | type default | Implementation to use for that type |
| `canny` | `object` | defaults | `low` (100), `high` (200) thresholds in Sobel magnitude units, `thickness` (1) in pixels |
| `lineart` | `object` | defaults | `threshold` (0.5) line intensity, `thickness` (1) in pixels |
| `softedge` | `object` | defaults | `threshold` (0.2) soft edge cutoff, `blur` (1.5) smoothing sigma |
| `depth` | `object` | defaults | `invert` (false: near is bright), `colormap` (`gray` or `turbo`) |
| `pose` | `object` | defaults | `detector` ("": the first one listed by `/preprocessors`), `threshold` (0.5) for a person to count, `kp_threshold` (0.3) for a joint to be drawn, `line_width` (4), `radius` (4), `background` (`black`, or `green` for the original openpose canvas) |
| `out` | `string` | `png` | `png` (returns an image) or `json` (returns `b64_json`) |

The map comes back the size of the image it was extracted from, so it can be paired
with that same image later - whoever conditions on it does its own resizing.

A `depth` request needs its checkpoint on disk (``scripts/download.py --model
preprocess``) and a `pose` request needs both of its own (``scripts/download.py --model
preprocess --variant pose``); when either is missing the endpoint answers **HTTP 400**
naming the command that fetches it. A type the engine does not implement, an unknown `model` or an
invalid option value is also **HTTP 400**; a `type` outside the implemented set is a
schema error (**HTTP 422**).

### Response

Returns a PNG image directly (`Content-Type: image/png`), with a `preprocess_data`
text chunk holding the type, the model and the options that differed from their
defaults, next to whatever metadata the input image already carried.

### Example

```bash
curl -s localhost:8000/preprocess \
  -H 'content-type: application/json' \
  -d '{"image_b64":"<base64 png>","type":"canny","canny":{"low":30,"high":60}}' \
  --output /tmp/fox_canny.png

curl -s localhost:8000/preprocess \
  -H 'content-type: application/json' \
  -d '{"image_b64":"<base64 png>","type":"pose","pose":{"kp_threshold":0.4}}' \
  --output /tmp/fox_pose.png
```

## Using it from Python

```python
import base64
import json
import urllib.request

req = urllib.request.Request(
    "http://127.0.0.1:8000/text2image",
    data=json.dumps({"prompt": "a fox walking in the snow", "steps": 8}).encode(),
    headers={"content-type": "application/json"},
)
with urllib.request.urlopen(req) as r:
    open("fox.png", "wb").write(r.read())

# editing: the reference image is base64-encoded
img_b64 = base64.b64encode(open("fox.png", "rb").read()).decode()
req = urllib.request.Request(
    "http://127.0.0.1:8000/edit",
    data=json.dumps({"image": img_b64, "prompt": "a fox wearing a red scarf", "steps": 4}).encode(),
    headers={"content-type": "application/json"},
)
with urllib.request.urlopen(req) as r:
    open("fox_edited.png", "wb").write(r.read())
```

## Error semantics

| Status | Meaning |
|--------|---------|
| `400` | Request asks for a capability the loaded model lacks (e.g. `/edit` without an editing model, `kv_cache` with `ref_method: index`), or an invalid field value |
| `503` | No model loaded for `/text2image` / `/edit` |
