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
| `GET` | `/health` | Server status, what the loaded model produces and its capabilities (see below) |
| `GET` | `/lora` | List available LoRA names, and the roles they can be given to |
| `GET` | `/upscalers` | List available pixel upscaler names (works even with no model loaded) |
| `GET` | `/postprocess` | Describe the loaded medium's post-processing knobs, in the order their filters run: colour, sharpening and grain, or a shape's finish (works even with no model loaded) |
| `POST` | `/upscale` | Pixel-upscale an input image (works even with no model loaded) |
| `GET` | `/controlmaps` | List the extractable control-map types (works even with no model loaded) |
| `POST` | `/controlmap` | Extract a control map from an input image (works even with no model loaded) |
| `POST` | `/text2image` | Generate an image |
| `POST` | `/edit` | Edit an image from an instruction (image + prompt → edited image); requires an editing-capable model |
| `POST` | `/video` | Generate a video synchronously, returning MP4 or WebM bytes |
| `POST` | `/mesh` | Generate a shape synchronously, returning GLB, OBJ or STL bytes |
| `POST` | `/jobs/text2image` | Queue a generation (202) |
| `POST` | `/jobs/edit` | Queue an edit (202) |
| `POST` | `/jobs/video` | Queue a video generation (202) |
| `POST` | `/jobs/mesh` | Queue a shape generation (202) |
| `GET` | `/jobs` | List the jobs the queue still remembers |
| `GET` | `/jobs/{id}` | Job status, progress, and what the request resolved to |
| `GET` | `/jobs/{id}/content` | Download a finished job's image or clip |
| `POST` | `/jobs/{id}/cancel` | Cancel a queued or running job |

## Asynchronous jobs

Each generator has a queued twin: the body you would send to `/text2image`, `/edit` or
`/video`, sent to `/jobs/text2image`, `/jobs/edit` or `/jobs/video` instead. The request
is answered straight away with **HTTP 202** and the job that now owns it:

```json
{"id": "job_2f9c4a7d…", "kind": "image", "status": "queued", "model": "krea2",
 "created_at": "2026-01-08T09:12:44.019236+00:00"}
```

`kind` is the medium the job produces (`image` or `video`) and `status` runs
`queued → running → completed`, or ends `failed` — with `error` saying why — or
`canceled`. `GET /jobs/{id}` adds progress and the resolved request:

```json
{"id": "job_2f9c4a7d…", "kind": "image", "status": "running", "model": "krea2",
 "created_at": "…", "started_at": "…",
 "progress": {"stage": "sampling", "step": 6, "of": 8},
 "resolved": {"mode": "t2i", "width": 1024, "height": 1024, "steps": 8, "sigmas": null,
              "guidance_scale": 1.0, "sampler": "euler", "seed": 7412,
              "upscale_factor": 1.0, "upscale_type": "refined", "pixel_upscaler": null,
              "kv_cache": false, "ref_method": null, "lora_specs": {}}}
```

`resolved` is the request as this model will actually run it: every default filled in,
and the seed already drawn, so a request that left `seed` out is still reproducible from
its own job. It is worked out at submit, which is also where a request this model cannot
serve is answered with **400**.

`progress.stage` runs `loading`, then `conditioning`, `sampling` (counting `step` of
`of`), `decoding` and `postprocessing` for an image — with `upscaling` between sampling
and decode when the request refines — and `loading`, `sampling`, `decoding`, `encoding`
for a video. A finished job's stage is its terminal status.

Once the status is `completed`, `GET /jobs/{id}/content` answers with the PNG or the
MP4/WebM bytes whose `mime_type` the status names. A finished artifact is **handed out
once**: serving it deletes it, as does the queue making room for a newer job and the
server exiting. Fetch it again and the answer is **404**; fetch it before completion and
the answer is **409**. While they wait, outputs live under `<cache-dir>/jobs`.

`POST /jobs/{id}/cancel` drops a queued job there and asks a running one to stop at its
next step: the record stays `running` until the engine acknowledges it, then becomes
`canceled`. Four requests may wait behind the one being rendered (**429** when the queue
is full), and the thirty-two most recent finished jobs are kept, oldest dropped first.

`GET /jobs` lists the jobs the queue still holds, oldest first, each entry exactly the
record `GET /jobs/{id}` gives. It shrinks when a submit needs room and when the server
exits; a restart empties it and the files under `<cache-dir>/jobs` with it. A record
outlives its bytes: a fetched job stays `completed` while its content answers 404.

Only generation and editing are queueable: `/upscale` and `/controlmap` have no queued
twin. `out`, the synchronous endpoints' `png`-or-`json` switch, is refused with **400**
by a submit.

```bash
id=$(curl -sS localhost:8000/jobs/text2image -H 'content-type: application/json' \
       -d '{"prompt":"a fox walking in the snow","steps":8}' | jq -r .id)
curl -sS localhost:8000/jobs/$id | jq -r .status      # queued, running, completed
curl -sS localhost:8000/jobs/$id/content -o fox.png
```

## Video generation

When a video model is loaded, TheNoise serves it through either the synchronous
`POST /video` endpoint (video bytes directly), or `POST /jobs/video`, which queues the
same body — see [Asynchronous jobs](#asynchronous-jobs). Both use the loaded model's
defaults when optional fields are omitted.

| Field | Type | Description |
|---|---|---|
| `prompt` | string | Required text prompt |
| `model` | string | Optional loaded model name, checked if provided |
| `image` | string | Optional base64-encoded first frame (I2V; omit for T2V) |
| `end_image` | string | Optional base64-encoded last frame (FLF2V; needs `image`) |
| `references` | array | Optional things the clip should look like without being frames of it. Each item is base64, or `{"media": ..., "frame": n}` to pin it to the frame at `n` instead (negative counts back from the last). A file is asked what it is — picture, clip or sound — so there is no kind to declare. Needs `video.ref2v` |
| `attention` | string | `auto` (default), `full` or `optimized` — how attention is computed. See [`attention_methods`](#get-health) |
| `negative_prompt` | string | Optional negative text prompt |
| `width`, `height` | integer | Video dimensions |
| `frames` | integer | Desired frame count |
| `duration`, `fps` | number, integer | Duration in seconds and frames/second; explicit `frames` takes priority |
| `steps`, `guidance_scale`, `seed` | number | Sampling parameters |
| `flow_shift` | number | Noise schedule shift. Defaults to the model's release value |
| `sampler` | string | Denoising solver, model default when omitted — one of `GET /health`'s `samplers` |
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

Send an existing image as the first frame with the `image` field. Its value
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
conditioning.

### First-and-last-frame video (FLF2V)

Add an `end_image` — same encoding, same rules — and the clip is pinned at both ends:
it starts on `image`, ends on `end_image`, and the prompt describes the change between
them. Both frames are cover-cropped to the output size; the first one decides it when
`width`/`height` are omitted.

```bash
FIRST="$(base64 -w 0 first.png)"
LAST="$(base64 -w 0 last.png)"
curl -fS http://127.0.0.1:8000/video \
  -H 'content-type: application/json' \
  --data-binary "{\"prompt\":\"the fox looks up and the snow thickens\",\"image\":\"$FIRST\",\"end_image\":\"$LAST\",\"width\":704,\"height\":480,\"frames\":33,\"steps\":8,\"output_format\":\"mp4\"}" \
  -o fox_flf2v.mp4
```

Which model can do this is `video.flf2v` in [GET /health](#get-health); a request that
asks a variant without it is refused with HTTP 400.

The normal `GET /lora` endpoint also lists LoRAs available to a loaded video
model, along with the `roles` a `lora_specs` request may name. A bare list goes
to the model's single transformer; a two-expert model offers `dit` and
`dit_low`, so each expert can be given its own LoRA, and asking for a role the
model does not have is refused at submit. Only LoRA factors matching the
transformer's supported layers are applied; unsupported targets are logged as
unused.

## Mesh generation

When a mesh model is loaded, TheNoise serves it through either the synchronous
`POST /mesh` endpoint (mesh bytes directly), or `POST /jobs/mesh`, which queues the
same body — see [Asynchronous jobs](#asynchronous-jobs). Both use the loaded model's
defaults when optional fields are omitted.

A shape is conditioned on a picture, so `image` is required. An array of pictures names the
views of one object, which only a model with a multi-view tower accepts; a model without
one refuses the second picture.

| Field | Type | Description |
|---|---|---|
| `image` | string \| string[] | Required base64-encoded picture, or an array of views of one object |
| `model` | string | Optional loaded model name, checked if provided |
| `latents` | integer | Latent tokens to denoise |
| `steps`, `guidance_scale`, `seed` | number | Sampling parameters |
| `sampler` | string | Denoising solver — one of `GET /health`'s `samplers` |
| `octree_resolution` | integer | Finest voxel grid the surface is cut from; the field that costs the most |
| `bounds` | number | Half-width of the sampled volume, in latent units |
| `mc_level` | number | The field value the surface is cut at |
| `scale` | number | Multiply the written coordinates; no mesh format carries a unit |
| `floaters` | number | Drop pieces spreading less than this fraction of the widest piece; `0` keeps everything |
| `triangles` | number | Keep this fraction of the surface detail |
| `smooth` | number | Ease out the roughness the cut leaves, holding creases |
| `lora_specs` | string[] \| object | Optional LoRAs, as `name:weight`, looked up in `--lora-dir` |
| `output_format` | string | `glb`, `obj` or `stl` (model default when omitted) |

`floaters`, `triangles` and `smooth` are described with their labels, defaults and bounds
by `GET /postprocess`, which describes a shape's finish chain while a shape model is
loaded.

For a synchronous glb response:

```bash
curl -fS http://127.0.0.1:8000/mesh -H 'content-type: application/json' \
  -d "{\"image\": \"$(base64 -w0 chair.png)\", \"octree_resolution\": 192}" \
  -o chair.glb
```

## `GET /health`

```json
{"status": "ok", "models": ["qwen_image"], "medium": "image",
  "samplers": ["er_sde", "euler", "unipc"],
  "capabilities": {"image": {"t2i": true, "i2i": true}, "kv_cache": true,
                   "upscale": true, "ref_methods": ["index", "index_timestep_zero", "vision"],
                   "attention_methods": ["auto", "full"]}}
```

With a video model:

```json
{"status": "ok", "models": ["wan22"], "medium": "video",
  "samplers": ["er_sde", "euler", "unipc"],
  "capabilities": {"image": null, "kv_cache": false, "upscale": false,
                   "video": {"t2v": true, "i2v": true, "flf2v": true, "ref2v": false},
                   "attention_methods": ["auto", "full", "optimized"]}}
```

With a mesh model:

```json
{"status": "ok", "models": ["hunyuan3d21"], "medium": "mesh",
  "samplers": ["er_sde", "euler", "unipc"],
  "capabilities": {"kv_cache": false, "mesh": {"i2d": true, "t2d": false},
                   "attention_methods": ["auto", "full"]}}
```

`models` is empty until a DiT is loaded, `medium` says what the loaded model produces
(`image` when none is loaded, so every route stays listed), and `capabilities` is that
adapter's own `CAPABILITIES` dict plus the two lists a request chooses from:
`ref_methods` and `attention_methods` (`{}` with nothing loaded).

The request modes sit under their medium: `t2i` and `i2i` under `image`, `t2v`, `i2v`,
`flf2v` and `ref2v` under `video`. A model answers the other medium with `null`.
`kv_cache` and `upscale` are not modes, so they stay at the top; either medium
can use them.

`ref2v` is whether the loaded model has anywhere in its packing for a picture the clip is
asked to resemble — as opposed to the frames it starts or ends on, which are `i2v` and
`flf2v`. It is answered by the architecture: a checkpoint does not say which release it is.
`ref2v: false` means a `references` field is refused, not ignored.

One mesh mode is not a boolean: `mv2d` is how many views of one object the model's tower
numbers, and its absence is the refusal — which is every mesh model served today.

`attention_methods` lists what the loaded model accepts for `attention` — a field every
medium's request takes (`/text2image`, `/edit`, `/jobs/video`, `/jobs/mesh`), whether or
not that trunk has more than one way to compute attention — on the box it is running on,
which is why the list can come and go with the hardware:

- `auto` — the engine chooses at every step. Block-sparse attention runs on a clip long
  enough for it to be the faster of the two; exact attention runs otherwise.
- `full` — exact attention at every step: every token reads every token.
- `optimized` — block-sparse attention whatever the clip's length, including the short
  clips where exact attention is normally faster.

Block-sparse attention reads the key blocks it keeps exactly and approximates the rest, so
it is faster and can cost detail. Only a model that has it lists `optimized`; sending a
value the loaded model does not list is a 400.

`i2i` says a model takes a picture with the request, not what it does with one. That is
`ref_method`: `index` and `index_timestep_zero` encode the picture as a latent the DiT
can line the result up against, while `vision` hands it to the text encoder's vision
tower, which steers what gets drawn and leaves nothing of the picture in place. A
model's `ref_methods` says which of those it offers; naming one it lacks is a 400.

The web UI reads all of this to decide what to show: the Edit tab follows
`i2i`, the clip form's Reference box follows `video.ref2v`, and
`attention_methods` decides whether the Attention picker is there at all — the picker's
own first option is the empty one, so two further words are needed before it is a choice,
and a model answering `auto` and `full` only hides the control. `medium` decides what the Generate tab becomes — a clip form for a
video model, a shape form for a mesh model. A request for something the loaded
model cannot do is refused with HTTP 400, not quietly downgraded. `i2v` and
`flf2v` come from the variant you loaded, not from the model name.
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
| `sampler` | `string` | model default | Denoising solver — one of `GET /health`'s `samplers` |
| `attention` | `string` | model default | How attention is computed — one of `GET /health`'s `attention_methods`. Every medium takes it; a model with one way to compute it answers only `auto` and `full` |
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

The same body sent to `/jobs/text2image` is queued instead of rendered: see
[Asynchronous jobs](#asynchronous-jobs).

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

Instruction-based editing: image(s) + prompt → edited image. Requires a model that
takes a reference; otherwise returns **HTTP 400**.

What the model does with the picture is `ref_method`, and each model accepts only the
methods `ref_methods` lists for it: an edit model the latent pair, a model whose text
encoder is multimodal also `vision` for being shown the picture without being asked to
repaint it, and a model with no latent path `vision` alone (MOSS-Image today). The
latent pair takes a per-reference timestep-zero marker; `vision` takes any number of
pictures and keeps none of them pixel for pixel.

Accepts all `/text2image` fields plus:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | `string` \| `string[]` | *(required)* | One or more base64-encoded reference images (OpenAI-style; first sets the output size when `width`/`height` omitted) |
| `ref_method` | `string` | auto | How the reference is used: `index`, `index_timestep_zero` (reference tokens conditioned at timestep zero, which is what makes the KV cache valid), or `vision` (the text encoder reads the picture; no latent, nothing repainted). Auto = detected from the checkpoint, else the model's first listed method. The cache default follows it |

> **Reference method.** Checkpoints trained to condition their reference tokens
> at timestep zero carry the `__index_timestep_zero__` marker, which makes
> `ref_method` resolve to `index_timestep_zero` automatically — and the KV cache
> with it. An explicit `ref_method` always wins over detection: markers are only
> a hint, and some trained checkpoints do not carry one. An explicit `kv_cache`
> together with an explicit `ref_method: index` is rejected rather than silently
> degraded. A method the loaded model does not list is a **400** too, as is any
> `ref_method` on a request without an image: what cannot be acted on is refused
> rather than ignored.

### Example

```bash
curl -s localhost:8000/edit \
  -H 'content-type: application/json' \
  -d '{"image":"<base64 png>","prompt":"a fox wearing a red scarf","steps":4,"sampler":"euler"}' \
  --output /tmp/fox_edited.png
```

`/jobs/edit` queues the same body.

## `POST /upscale`

Pixel-upscales an existing image by `upscale_factor`× with a named pixel
upscaler. Unlike `/text2image`, this needs no diffusion model loaded - only an
upscaler configured via `--upscaler-dir`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image_b64` | `string` | *(required)* | Base64-encoded input image |
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

What it describes is the tail of the medium that is loaded, since that is what a pane
has to build: the pixel filters an image or clip decodes into, or the filters a shape
is cleaned with. With nothing loaded it describes the pixel filters. For a shape model:

```json
{
  "options": [
    {"name": "floaters", "label": "Clean up junk", "type": "number",
     "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
    {"name": "triangles", "label": "Level of detail", "type": "number",
     "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
    {"name": "smooth", "label": "Smooth the surface", "type": "number",
     "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}
  ]
}
```

The three are described as one dial each: they run in list order, so a request cannot
say how much to smooth before saying how many triangles to keep, and the pane reads
the order off this payload rather than repeating it.

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
| `image_b64` | `string` | *(required)* | Base64-encoded input image |
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

A video run caches its two encode stages and not the clip: repeating a request
over the same reference image skips the codec, and one that changes only the
prompt skips the text encoder as well. A changed image costs the text encoder
too.

A shape run caches its conditioning and its sampling, so the same picture with
the same settings is cut again without resampling; changing the grid, the
finish or the scale only costs the cut.

## Error semantics

A route answers a refusal in the format it speaks.

| Endpoints | Refusal body |
|-----------|--------------|
| `/text2image`, `/edit`, `/upscale`, `/controlmap`, `/video` | `text/plain`: the reason |
| `/jobs/*` | `{"error": "<reason>"}` |

| Status | Meaning |
|--------|---------|
| `400` | A value the loaded model refuses: a capability it lacks (e.g. `/edit` without an editing model, `kv_cache` with `ref_method: index`), a `ref_method` it does not offer, a sampler it does not implement, an `image` it cannot decode. A job's `400` arrives at submit, never in its status |
| `404` | No such job, or its content has already been handed out |
| `409` | A job is not in the state the request needs: content before completion, cancel after it settled |
| `429` | The job queue is full |
| `500` | Unexpected failure, logged with a traceback |
| `503` | No model loaded for `/text2image` / `/edit`, or a job the queue cannot take because nothing is loaded or the server is shutting down |

A body the request contract itself refuses — an unknown field, a type it cannot read, a value outside its bounds (`frames: 0`) — is FastAPI's `422` JSON report.
