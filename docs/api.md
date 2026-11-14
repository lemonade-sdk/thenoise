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
| `GET` | `/lora` | List available LoRA names, and the roles they can be given to |
| `GET` | `/upscalers` | List available pixel upscaler names (works even with no model loaded) |
| `GET` | `/postprocess` | Describe the post-decode knobs, in the order their filters run (works even with no model loaded) |
| `POST` | `/upscale` | Pixel-upscale an input image (works even with no model loaded) |
| `GET` | `/controlmaps` | List the extractable control-map types (works even with no model loaded) |
| `POST` | `/controlmap` | Extract a control map from an input image (works even with no model loaded) |
| `POST` | `/text2image` | Generate an image |
| `POST` | `/edit` | Edit an image from an instruction (image + prompt → edited image); requires an editing-capable model |
| `POST` | `/video` | Generate a video synchronously, returning MP4 or WebM bytes |
| `POST` | `/videos` | Queue an asynchronous video generation (202) |
| `GET` | `/videos/{id}` | Get job status and progress |
| `GET` | `/videos/{id}/content` | Download a completed video |
| `POST` | `/videos/{id}/cancel` | Cancel a queued/running video |

## Video generation

When a video model is loaded, TheNoise serves the model through either the
synchronous `POST /video` endpoint (video bytes directly), or the asynchronous
job endpoints (`POST /videos` + polling). Both accept the same JSON body and use
the loaded model's defaults when optional fields are omitted.

| Field | Type | Description |
|---|---|---|
| `prompt` | string | Required text prompt |
| `model` | string | Optional loaded model name, checked if provided |
| `image` | string | Optional base64-encoded PNG/JPEG first frame (I2V; omit for T2V) |
| `negative_prompt` | string | Optional negative text prompt |
| `width`, `height` | integer | Video dimensions (Wan rounds down to multiples of 32) |
| `frames` | integer | Desired frame count (Wan uses a `4n+1` grid) |
| `duration`, `fps` | number, integer | Duration in seconds and frames/second; explicit `frames` takes priority |
| `steps`, `guidance_scale`, `seed` | number | Sampling parameters |
| `sampler` | string | Model-supported sampler (`unipc` or `euler` for Wan) |
| `lora_specs` | string[] \| object | Optional LoRAs, as `name:weight`, looked up in `--lora-dir`. A model with more than one LoRA-able component also takes them by component: `{"dit": ["a:0.8"], "dit_low": ["b"]}` |
| `output_format` | string | `mp4` or `webm` (model default when omitted) |

For a synchronous MP4 response:

```bash
curl -fS http://127.0.0.1:8000/video -H 'content-type: application/json' \
  -d '{"prompt":"A fox walking through snow","width":704,"height":480,
       "frames":33,"steps":8,"sampler":"unipc","output_format":"mp4"}' \
  -o fox.mp4
```

### Image-to-video (I2V)

Send an existing PNG or JPEG as the first frame with the `image` field. Its value
is base64 of the raw file bytes, **not** a `data:` URL or a file path. The
request field is `image`, not `image_b64`. For example, on Linux:

```bash
IMAGE_B64="$(base64 -w 0 reference.png)"
curl -fS http://127.0.0.1:8000/video \
  -H 'content-type: application/json' \
  --data-binary "{\"prompt\":\"The camera moves forward as the fox walks\",\"image\":\"$IMAGE_B64\",\"width\":704,\"height\":480,\"frames\":33,\"steps\":8,\"output_format\":\"mp4\"}" \
  -o fox_i2v.mp4
```

The image is cover-cropped to the requested output size and encoded as first-frame
conditioning. For a platform-independent Python-based example, see the
[Wan 2.2 model page](models/wan22.md#image-to-video). To queue I2V instead,
send the same request to `POST /videos` and poll its job ID.

### Asynchronous jobs

For asynchronous generation, send the same body to `/videos`. A successful
request returns HTTP 202 with `id`, `status`, `model` and `created_at`. Poll
`GET /videos/{id}`: the `progress` object reports `stage`, `step` and `of`.
Once `status` becomes `completed`, download `/videos/{id}/content`. This returns
HTTP 409 until completion. To stop a queued or running job use
`POST /videos/{id}/cancel`. The queue returns HTTP 429 when full.

The normal `GET /lora` endpoint also lists LoRAs available to a loaded video
model, along with the `roles` a `lora_specs` request may name. A bare list goes
to the model's single transformer; a two-expert model offers `dit` and
`dit_low`, so each expert can be given its own LoRA, and asking for a role the
model does not have is refused at submit. Only LoRA factors matching the
transformer's supported layers are applied; unsupported targets are logged as
unused.

## `GET /health`

```json
{"status": "ok", "models": ["qwen_image"],
 "capabilities": {"edit": true, "kv_cache": true, "upscale": true,
               "video": false, "t2v": false, "i2v": false}}
```

`models` is empty until a DiT is loaded, and `capabilities` is the loaded
adapter's `CAPABILITIES` dict verbatim (`{}` with no model) - the web UI uses it
to gate the Edit tab and the KV-cache control, and a request asking for a
capability the model lacks is rejected with HTTP 400 rather than silently
ignored. One vocabulary serves both mediums, so a video model answers the same
keys with `video`/`t2v`/`i2v` true and `upscale` false, and `i2v` follows the
released variant rather than the model name.

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
| `kv_cache` | `boolean` | auto | Reuse step-invariant K/V across denoising steps. T2I defaults off (opt-in); compatible edits auto-enable based on `ref_method`. Explicit `true` requires model support |
| `qwen_vae_enhance` | `bool` | `false` | Nyquist notch post-filter (removes 2px grid artifacts) |
| `saturation` | `float` | `0.0` | Colour saturation around neutral `0.0`: -1.0 greyscales, 1.0 doubles |
| `vibrance` | `float` | `0.0` | Saturation that lifts washed-out pixels and leaves vivid ones alone |
| `temperature` | `float` | `0.0` | White balance around neutral `0.0`: -1.0 cools to blue, 1.0 warms to amber |
| `tint` | `float` | `0.0` | White balance across the green–magenta axis: -1.0 greens, 1.0 magentas |
| `sharpening` | `float` | `0.0` | RCAS sharpening strength, 0.0–1.0 |
| `film_grain` | `float` | `0.0` | Film grain strength, 0.0–10.0 |
| `lora_specs` | `string[]` \| `object` | `null` | LoRA specs, e.g. `["style:0.8"]`, or by component on a two-expert model: `{"dit": ["a"], "dit_low": ["b"]}` |

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

## `GET /postprocess`

```json
{
  "options": [
    {"name": "qwen_vae_enhance", "label": "Reduce grid pattern", "type": "boolean",
     "default": false, "hint": "enable when needed"},
    {"name": "saturation", "label": "Saturation", "type": "number",
     "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}
  ]
}
```

## `GET /controlmaps`

```json
{
  "controlmaps": {"canny": ["canny"]},
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

`controlmaps` maps each control-map type to the implementations it can choose
between, `defaults` to the one used when a request does not name one. Both are
empty when nothing is implemented, and no diffusion model is needed to list them.

`options` describes the request fields of every kind the engine knows - label,
control type, default, bounds - and `pool` names the list a field is filled from
(`detectors`, above). It is generated from the option dataclasses the extraction
itself runs on, which is also where the CLI flags and the web UI form come from, so
none of the three can show a default the engine would not use.

`detectors` lists the person detectors on disk that a `pose` request crops its people
with, by name. `pose` appears in `controlmaps` only once both its checkpoints - the
keypoint model *and* a detector - are in the directory: an estimator with nothing to
crop with measures nothing.

## `POST /controlmap`

Extracts a conditioning map - Canny or line-art edges, a soft edge field, a
Depth-Anything-V2 depth map or an openpose-style pose skeleton. Like `/upscale`, this
needs no diffusion model loaded.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image_b64` | `string` | *(required)* | Base64-encoded input image (PNG/JPEG) |
| `type` | `string` | *(required)* | Control-map type, one of those listed by `/controlmaps` |
| `model` | `string` | type default | Implementation to use for that type |
| `canny` | `object` | defaults | `low` (100), `high` (200) thresholds in Sobel magnitude units, `thickness` (1) in pixels |
| `lineart` | `object` | defaults | `threshold` (0.5) line intensity, `thickness` (1) in pixels |
| `softedge` | `object` | defaults | `threshold` (0.2) soft edge cutoff, `blur` (1.5) smoothing sigma |
| `depth` | `object` | defaults | `invert` (false: near is bright), `colormap` (`gray` or `turbo`) |
| `pose` | `object` | defaults | `detector` ("": the first one listed by `/controlmaps`), `threshold` (0.5) for a person to count, `kp_threshold` (0.3) for a joint to be drawn, `line_width` (4), `radius` (4), `background` (`black`, or `green` for the original openpose canvas) |
| `out` | `string` | `png` | `png` (returns an image) or `json` (returns `b64_json`) |

The map comes back the size of the image it was extracted from, so it can be paired
with that same image later - whoever conditions on it does its own resizing.

A `depth` request needs its checkpoint on disk (``scripts/download.py --model
controlmap``) and a `pose` request needs both of its own (``scripts/download.py --model
controlmap --variant pose``); when either is missing the endpoint answers **HTTP 400**
naming the command that fetches it. A type the engine does not implement, an unknown `model` or an
invalid option value is also **HTTP 400**; a `type` outside the implemented set is a
schema error (**HTTP 422**).

### Response

Returns a PNG image directly (`Content-Type: image/png`), with a `controlmap_data`
text chunk holding the type, the model and the options that differed from their
defaults, next to whatever metadata the input image already carried.

### Example

```bash
curl -s localhost:8000/controlmap \
  -H 'content-type: application/json' \
  -d '{"image_b64":"<base64 png>","type":"canny","canny":{"low":30,"high":60}}' \
  --output /tmp/fox_canny.png

curl -s localhost:8000/controlmap \
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

## Caching

Pipeline stages cache their output, keyed by the request fields that affect them,
so a request repeated verbatim — same prompt, guidance, size, steps, seed and
sampler — is served from the cache: identical bytes, a fraction of the time, no
resampling. A request that omits `seed` (or passes `-1`) draws a fresh seed
before the key is built and so never hits it; when timing repeated requests, vary
the seed or prompt.

## Error semantics

A route answers a refusal in the format it speaks.

| Endpoints | Refusal body |
|-----------|--------------|
| `/text2image`, `/edit`, `/upscale`, `/controlmap`, `/video` | `text/plain`: the reason |
| `/videos`, `/videos/{id}`, `/videos/{id}/content`, `/videos/{id}/cancel` | `{"error": "<reason>"}` |

| Status | Meaning |
|--------|---------|
| `400` | A value the loaded model refuses: a capability it lacks (e.g. `/edit` without an editing model, `kv_cache` with `ref_method: index`), a sampler it does not implement, an `image` it cannot decode |
| `404` | No such video job, or its content is gone |
| `409` | Video job not in the state the request needs: content before completion, cancel after it settled |
| `429` | The video queue is full |
| `500` | Unexpected failure, logged with a traceback |
| `503` | No model loaded for `/text2image` / `/edit`; a video job the service cannot take, because no video model is loaded or it is shutting down |

A body the request contract itself refuses — an unknown field, a type it cannot read, a value outside its bounds (`frames: 0`) — is FastAPI's `422` JSON report.
