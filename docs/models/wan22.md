# Wan 2.2

## TI2V-5B

**Native text-to-video and image-to-video**, using the Comfy-Org Wan 2.2 5B
checkpoint, UMT5 text encoder and a causal 3D video VAE. Output is MP4/H.264
or WebM/VP9.

- T2V and I2V use the same model and HTTP API.
- Built-in defaults: **1280×704, 33 frames, 24 FPS, 12 steps, guidance 5.0, UniPC**.
- The temporal frame grid follows **`4n+1`**, with no fixed maximum frame count.
- Supports optional DiT LoRAs through the same `--lora-dir` and `lora_specs`
  conventions as image generation.

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Specs

| Spec | Data |
|---|---|
| Model | Wan 2.2 TI2V-5B, 48-channel video latent |
| Download | ~23 GB (FP16 DiT, UMT5, VAE combined) |
| Text encoder | UMT5-XXL, BF16 compute |
| VAE | Native causal 3D VAE, FP32 compute |
| Modes | Text-to-video, image-to-video |
| Defaults | 1280×704, 33 frames, 24 FPS, 12 steps, CFG 5.0, UniPC, WebM |
| Samplers | UniPC and Euler; fixed Wan flow shift |

## Performance (Strix Halo)

Successfully tested end-to-end on Strix Halo (Radeon 8060S, gfx1151,
ROCm 10.1). At **704×480, 33 frames, 8 UniPC steps**, sampling took **28.8 s**.
A VAE transient of **24.6 GiB** was observed above the stage baseline; the
full generation time was not recorded. RTX 5090 T2V and I2V are also tested,
but no comparative timings are recorded here.

## Download

```bash
uv pip install -e ".[scripts]"
.venv/bin/python scripts/download.py --model wan22
```

The three files are:

```
models/wan22/split_files/
  diffusion_models/wan2.2_ti2v_5B_fp16.safetensors
  text_encoders/umt5_xxl_fp16.safetensors
  vae/wan2.2_vae.safetensors
```

## Usage

Start the server using the ROCm-configured launcher (which sets
`MIOPEN_FIND_MODE=FAST` on AMD GPUs):

```bash
./thenoise.sh serve \
  --dit ./models/wan22/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors \
  --text-encoder ./models/wan22/split_files/text_encoders/umt5_xxl_fp16.safetensors \
  --vae ./models/wan22/split_files/vae/wan2.2_vae.safetensors \
  --port 8000
```

### Text-to-video (T2V)

Generate a video directly from a text prompt:

```bash
curl -fS http://127.0.0.1:8000/video -H 'content-type: application/json' \
  -d '{"prompt":"A fox walking through snow","frames":33,"width":704,
       "height":480,"steps":8,"sampler":"unipc","output_format":"mp4"}' \
  -o fox.mp4
```

### Image-to-video (I2V)

To animate an existing image, supply the first frame as a **PNG or JPEG** in the
`image` request field. The value must be base64 of the image file bytes (not a
`data:image/...;base64,` URL and not a filesystem path). TheNoise cover-crops the
input to the requested video dimensions, encodes it using the video VAE and
conditions the first frame of the generated clip on that image. Text in `prompt`
describes what should happen **after** the input frame.

For example, with `reference.png` in the current directory, generate an MP4
synchronously. The bundled Python builds the JSON payload, so no additional
`jq` or platform-specific `base64` command is needed:

```bash
.venv/bin/python - <<'PYTHON' | curl -fS http://127.0.0.1:8000/video \
  -H 'content-type: application/json' --data-binary @- -o fox_i2v.mp4
import base64
import json
from pathlib import Path

image = base64.b64encode(Path("reference.png").read_bytes()).decode("ascii")
print(json.dumps({
    "prompt": "The fox starts walking forward while the camera tracks it",
    "image": image,
    "width": 704,
    "height": 480,
    "frames": 33,
    "fps": 24,
    "steps": 8,
    "sampler": "unipc",
    "output_format": "mp4",
}))
PYTHON
```

For an **asynchronous I2V** job, use the same body but send it to `POST /videos`
(instead of `/video`) and omit `-o fox_i2v.mp4`. Poll and download the clip
using the job API below. Omitting `image` selects text-to-video; setting it
selects image-to-video. Both modes accept `lora_specs`.

### Asynchronous generation and retrieval

Or create an asynchronous text-to-video job:

```bash
curl -sS http://127.0.0.1:8000/videos -H 'content-type: application/json' \
  -d '{"prompt":"A fox walking through snow","frames":33,"steps":8}'
```

The response contains a video ID. Poll `/videos/{id}` and fetch
`/videos/{id}/content` after completion. Send `POST /videos/{id}/cancel`
to request cancellation. I2V uses the same job flow and the same base64
`image` field as the synchronous request above.

For LoRAs, supply `--lora-dir ./models/loras` when starting the server, then
send `"lora_specs": ["style:0.8"]` in either video request. The engine loads
only supported DiT projections, reports unmatched keys and switches LoRAs
between requests. The FP8 text-encoder checkpoint is not supported.

See the [HTTP API](../api.md#video-generation) for parameters and responses.

## T2V-A14B (two experts)

This model uses **two** separately loaded DiTs (high-/low-noise), the **Wan 2.1
16-channel VAE** (`wan_2.1_vae.safetensors`), and the same FP16 UMT5 encoder as
5B. TheNoise finds the low-noise expert beside the high-noise `--dit` path.
FP16 and ComfyUI FP8-scaled experts are supported (FP8 requires Comfy Kitchen).
Both experts are swapped by the normal device memory manager at the sigma
boundary; do not run both as resident GPU models on a 32 GiB card.

```bash
./thenoise.sh serve \
  --dit ./models/wan22/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors \
  --text-encoder ./models/wan22/split_files/text_encoders/umt5_xxl_fp16.safetensors \
  --vae ./models/wan22/split_files/vae/wan_2.1_vae.safetensors \
  --port 8000

curl -fS http://127.0.0.1:8000/video -H 'content-type: application/json' \
  -d '{"prompt":"A fox walking through snow","width":704,"height":480,
       "frames":33,"steps":12,"sampler":"euler","output_format":"mp4"}' \
  -o fox_14b.mp4
```

Defaults: 832x480, 33 frames, 16 FPS, 40 steps, Euler, CFG 4.0,
flow shift 12, and high/low boundary 0.875. The single API CFG value
is the high-noise CFG (4.0 by default); low-noise CFG is 0.75x (3.0 by default).
**T2V only**: 14B I2V uses a separate 36-channel
model and CLIP-Vision path; 14B LoRAs are not part of this patch.
