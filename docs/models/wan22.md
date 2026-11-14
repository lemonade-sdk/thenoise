# Wan 2.2

Two Wan 2.2 releases are supported, and they are different models rather than two
sizes of one.

| | TI2V-5B | T2V-A14B |
|---|---|---|
| Modes | text-to-video, image-to-video | text-to-video |
| Transformer | one checkpoint (`--dit`) | two experts: `--dit` high noise, `--dit-low` low noise |
| Latent and codec | 48 channels, Wan 2.2 causal 3D VAE, BF16 compute | 16 channels, Wan 2.1 causal 3D VAE, FP16 compute |
| Text encoder | UMT5-XXL, FP16 file, BF16 compute | the same file |
| Samplers | UniPC or Euler | UniPC or Euler |
| Defaults | 1280×704, 33 frames, 24 FPS, 12 steps, CFG 5.0, UniPC, WebM, fixed Wan flow shift | 832×480, 33 frames, 16 FPS, 40 steps, Euler, CFG 4.0, flow shift 12, high/low boundary 0.875 |

Both answer the same video request over the same [HTTP API](../api.md#video-generation):
`POST /video` generates and returns the clip, `POST /videos` queues a job. Output is
MP4/H.264 or WebM/VP9 on the **`4n+1`** temporal grid, with no fixed maximum frame count.

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## TI2V-5B

Native text-to-video and image-to-video from the Comfy-Org Wan 2.2 5B checkpoint.
The two modes are the same model and the same request: `image` selects one, `prompt`
describes what happens after the first frame.

### Download

```bash
uv pip install -e ".[scripts]"
.venv/bin/python scripts/download.py --model wan22
```

The three files, about 23 GB together:

```
models/wan22/split_files/
  diffusion_models/wan2.2_ti2v_5B_fp16.safetensors
  text_encoders/umt5_xxl_fp16.safetensors
  vae/wan2.2_vae.safetensors
```

### Serve

Start the server using the ROCm-configured launcher (which sets
`MIOPEN_FIND_MODE=FAST` on AMD GPUs):

```bash
./thenoise.sh serve \
  --dit ./models/wan22/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors \
  --text-encoder ./models/wan22/split_files/text_encoders/umt5_xxl_fp16.safetensors \
  --vae ./models/wan22/split_files/vae/wan2.2_vae.safetensors \
  --port 8000
```

### Text-to-video

```bash
curl -fS http://127.0.0.1:8000/video -H 'content-type: application/json' \
  -d '{"prompt":"A fox walking through snow","frames":33,"width":704,
       "height":480,"steps":8,"sampler":"unipc","output_format":"mp4"}' \
  -o fox.mp4
```

### Image-to-video

Supply the first frame as a **PNG or JPEG** in the `image` request field. The value
must be base64 of the image file bytes (not a `data:image/...;base64,` URL and not a
filesystem path). TheNoise cover-crops the input to the requested video dimensions,
encodes it with the video VAE and conditions the first frame of the generated clip on
that image.

With `reference.png` in the current directory, generate an MP4 synchronously. The
bundled Python builds the JSON payload, so no additional `jq` or platform-specific
`base64` command is needed:

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

For an asynchronous I2V job, send the same body to `POST /videos` — see
[Video jobs](#video-jobs).

### Performance (Strix Halo)

Successfully tested end-to-end on Strix Halo (Radeon 8060S, gfx1151,
ROCm 10.1). At **704×480, 33 frames, 8 UniPC steps**, sampling took **28.8 s**;
the full generation time was not recorded. RTX 5090 T2V and I2V are also tested,
but no comparative timings are recorded here.

## T2V-A14B (two experts)

Two separately loaded DiTs (high- and low-noise), the **Wan 2.1 16-channel VAE**
(`wan_2.1_vae.safetensors` — Wan 2.1 by latent format, not a version mismatch: it is the
codec Qwen-Image uses for stills), and the same FP16 UMT5 encoder as 5B.

Pass **both expert files explicitly**: `--dit` selects high noise and `--dit-low`
selects low noise. File names and directories are not inspected to infer the expert
role. FP16 and ComfyUI FP8-scaled experts are supported. The latent width is read off
the `--vae` file, and a codec that does not match the transformer is refused at load
rather than at the first denoising step.

Both experts are swapped by the normal device memory manager at the sigma boundary,
which keeps both on the card when they fit; do not run both as resident GPU models on a
32 GiB card. The single `guidance_scale` a request carries is the high-noise CFG (4.0 by
default); low-noise CFG is 0.75× that (3.0 by default).

**T2V only**: 14B I2V is a separate 36-channel model with a CLIP-Vision path, which this
engine does not load.

### Download

```bash
.venv/bin/python scripts/download.py --model wan22-14b
.venv/bin/python scripts/download.py --model wan22-14b --variant fp8_scaled
```

The default is FP16; `--variant fp8_scaled` takes the ComfyUI FP8-scaled experts:

```
models/wan22/split_files/
  diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors
  diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors
  text_encoders/umt5_xxl_fp16.safetensors
  vae/wan_2.1_vae.safetensors
```

### Serve and generate

```bash
./thenoise.sh serve \
  --dit ./models/wan22/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
  --dit-low ./models/wan22/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors \
  --text-encoder ./models/wan22/split_files/text_encoders/umt5_xxl_fp16.safetensors \
  --vae ./models/wan22/split_files/vae/wan_2.1_vae.safetensors \
  --port 8000

curl -fS http://127.0.0.1:8000/video -H 'content-type: application/json' \
  -d '{"prompt":"A fox walking through snow","width":704,"height":480,
       "frames":33,"steps":12,"sampler":"euler","output_format":"mp4"}' \
  -o fox_14b.mp4
```

### One LoRA per expert

Each expert is LoRA-able on its own:

```json
{"lora_specs": {"dit": ["style:0.8"], "dit_low": ["pose"]}}
```

A bare list belongs to the high-noise expert, so a LoRA meant for the whole clip is
sent under both roles. `GET /lora` lists the roles this model offers.

## Video jobs

Both variants queue the same way:

```bash
curl -sS http://127.0.0.1:8000/videos -H 'content-type: application/json' \
  -d '{"prompt":"A fox walking through snow","frames":33,"steps":8}'
```

The response contains a video ID. Poll `/videos/{id}`, whose `progress` reports `stage`,
`step` and `of`, and fetch `/videos/{id}/content` once the status is `completed`. Send
`POST /videos/{id}/cancel` to request cancellation. I2V uses the same flow and the same
base64 `image` field.

For LoRAs, supply `--lora-dir ./models/loras` when starting the server, then
send `"lora_specs": ["style:0.8"]` in either video request — the same conventions as
image generation. The engine loads only supported DiT projections, reports unmatched
keys and switches LoRAs between requests. The FP8 text-encoder checkpoint is not
supported.

See the [HTTP API](../api.md#video-generation) for parameters and responses.
