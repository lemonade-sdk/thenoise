# LTX 2.3 / LTX 2.5 audiovisual video

LTX is wired into the **current** TheNoise video architecture: the shared
`VideoPipelineController`, `VideoModel` and `JobQueue` own execution, staging,
progress, cancellation and output. The model adapter supplies only LTX's
text/AV conditioning, latent generation, stages, original checkpoint loading,
spatial refinement and video/audio decoding. `diffusers>=0.43` from PyPI
supplies Apache-2 modular numerical classes; it does not run a Diffusers
pipeline. No `ltx-core`/`ltx-pipelines` or Git-Python dependencies are used.

**Validation:** LTX 2.3 and LTX 2.5 have both been tested with full
checkpoints on GPU. Weight-free CPU tests cover the integration and
checkpoint mapping; they do not replace end-to-end GPU validation.

## Setup / weights (INT8 ConvRot default)

TheNoise uses its existing `QuantizedLinear` and comfy-kitchen ConvRot kernels.
The model is not dequantized into a BF16 DiT. The auxiliary video/audio VAEs and
spatial upscaler remain BF16 as supplied by Lightricks.

```sh
uv pip install --python .venv/bin/python -e '.[ltx,scripts]'
# Prints exact HF repositories, files, and which already exist locally:
.venv/bin/python scripts/download.py --model ltx23 --dry-run
.venv/bin/python scripts/download.py --model ltx25 --dry-run
# INT8 ConvRot for BOTH the DiT and Gemma, by default:
.venv/bin/python scripts/download.py --model ltx23
.venv/bin/python scripts/download.py --model ltx25
# If you explicitly want the much larger BF16 checkpoints:
.venv/bin/python scripts/download.py --model ltx25 --variant bf16
```

**LTX 2.3 INT8 sources:** The original Lightricks 2.3 repo does not provide an
official INT8 ConvRot Distilled 1.1 file. We use the *community-produced*
`milo01/ComfyUI-Native-INT8_ConvRot` Distilled 1.1 monolith and the
`Stick9190/gemma-3-12b-it-int8-convrot` Gemma-3 INT8 ConvRot encoder.
The small Gemma-3 configuration/tokenizer assets (not BF16 shards) come from
`Lightricks/gemma-3-12b-it-qat-q4_0-unquantized`. **Caution:** Stick9190
quantizes the regular Gemma-3-12B-IT checkpoint, not Lightricks' QAT release.
Its behavior may differ from Lightricks' QAT release. Inspect the terms
and provenance of both community exports.

**LTX 2.5 INT8 source:** Both Distilled DiT and Gemma-4 with-projection weights
are the official `*-comfy-int8-convrot.safetensors` files on
`Lightricks/LTX-2.5`. `--variant bf16` remains available but is never the default.

Run LTX 2.3 on CUDA (or ROCm with the corresponding PyTorch build):

```sh
./thenoise.sh serve \
  --dit ./models/ltx23/checkpoints/ltx-2.3-22b-distilled-1.1-int8-ConvRot.safetensors \
  --vae ./models/ltx23/vae/LTX23_video_vae_bf16.safetensors \
  --ltx-audio-vae ./models/ltx23/vae/LTX23_audio_vae_bf16.safetensors \
  --ltx-upscaler ./models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --ltx-text-projection ./models/ltx23/text_encoders/ltx-2.3_text_projection_bf16.safetensors \
  --text-encoder ./models/gemma3-12b \
  --device cuda:0 --offload-device cpu --port 8000
```

The community INT8 checkpoint supplies only the DiT and connector attention.
The downloader adds separate Video VAE, Audio VAE, and the **2.31 GB LTX 2.3
text projection** (`text_encoders/ltx-2.3_text_projection_bf16.safetensors`)
from Kijai. The latter supplies the four learned audio/video aggregate projection
parameters missing from the INT8 DiT. All must load strictly; random/meta
weights are not acceptable. The Audio VAE file also contains the vocoder.
For the full BF16 monolith, `--dit` and `--vae` may still point to
the same file. The `--text-encoder` directory contains Gemma-3 INT8 weights
and the HF tokenizer/config assets.

Run LTX 2.5:

```sh
./thenoise.sh serve \
  --dit ./models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
  --vae ./models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  --ltx-audio-vae ./models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --ltx-upscaler ./models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --text-encoder ./models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
  --device cuda:0 --offload-device cpu --port 8000
```

The separate LTX 2.5 audio VAE, vocoder, spatial upscaler and text connectors
are passed explicitly through the CLI. The code checks scale
and `.comfy_quant` siblings before assigning a quantized weight; absent or
incompatible metadata is a hard error, not a silent BF16 fallback.

**GPU status:** Both LTX 2.3 and LTX 2.5 have passed end-to-end GPU
generation tests. T2V and I2V use the same TheNoise video controller;
audio is encoded through the shared PyAV container path. The CPU suite
tests these interfaces without asserting audiovisual output quality.

## Request

Use `POST /video` for synchronous output or `POST /jobs/video` plus
`GET /jobs/{id}` and `GET /jobs/{id}/content` for queued rendering. The
legacy `POST /videos` route no longer exists on the current `main`.

Text-to-video:

```sh
curl -sS http://127.0.0.1:8000/video -H 'content-type: application/json' \
  -d '{"prompt":"A fox walking through snow, audible footsteps and cold wind","width":512,"height":512,"frames":33,"fps":24,"steps":8,"sampler":"euler","guidance_scale":1,"output_format":"mp4"}' \
  -o fox.mp4
```

Image-to-video (PNG/JPEG image in the `image` field as **plain base64**, no
`data:image/png;base64,` prefix):

```sh
.venv/bin/python - <<'PY' > /tmp/ltx_i2v.json
import base64, json
with open('input.png', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode('ascii')
print(json.dumps({"prompt":"The person looks toward the horizon, moving naturally",
                  "image":b64, "width":512,"height":512,"frames":33,
                  "fps":24,"steps":8,"sampler":"euler", "output_format":"mp4"}))
PY
curl -sS -w '\nHTTP %{http_code}\n' \
  http://127.0.0.1:8000/video -H 'content-type: application/json' \
  --data-binary @/tmp/ltx_i2v.json -o i2v.mp4
```

This model accepts the historical `"sampler":"native"` in video requests as
an alias for TheNoise's `euler` sampler. `steps=8` means the distilled 8+3 stage
schedule. MP4 contains H.264 video and AAC audio; WebM, runtime LoRA,
negative prompts, CFG other than 1, and FLF2V are not supported.

Check the output with `ffprobe -v error -show_streams i2v.mp4`.

## Notes

- The output is twice the first-stage resolution, so sizes are aligned to 64.
- I2V reference pixels arrive from the shared `pil_to_pixels` path already in
  `[-1,1]`; do **not** normalize them a second time.
- Video stages use TheNoise's `MemoryManager`, including optional CPU block
  offloading for DiTs that exceed device memory.
- The `VideoPipelineController` keeps the 8-step sampling and 3-step refinement
  under the same DiT instance and moves the spatial upsampler only between them.
- On shutdown, `JobQueue` owns the generated-file cache and its cleanup;
  do not add a second video cache manager.
