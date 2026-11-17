# LTX 2.3 / LTX 2.5 (distilled, audio + video)

LTX uses the existing `VideoModel`, `VideoPipelineController`, `MemoryManager`,
checkpoint and quantization utilities. The LTX adapter owns model-specific
kernels, checkpoint *key mappings*, audio/video conditioning and refinement.
It does **not** install custom block offloading. The standard memory manager
controls component residency and stage eviction as for other TheNoise models.

## Checkpoints

The downloader uses the same switches as other TheNoise models: BF16 by default,
`--int8-convrot` for the Comfy-format quantized release. No new quantization
variant or filename-derived model configuration is involved.

```sh
.venv/bin/python scripts/download.py --model ltx23 --int8-convrot
.venv/bin/python scripts/download.py --model ltx25 --int8-convrot
# Or, if your machine has sufficient memory, omit --int8-convrot for BF16.
```

LTX 2.3 BF16 uses the official `Lightricks/LTX-2.3` monolith for video,
audio and projections, with a separate, **single-file** Gemma 3 BF16 encoder
from `Comfy-Org/ltx-2`. The
**community** INT8 ConvRot checkpoint (`milo01/ComfyUI-Native-INT8_ConvRot`)
requires separate video/audio VAEs and text projections from
`Kijai/LTX2.3_comfy`. Its Gemma 3 export from
`Stick9190/gemma-3-12b-it-int8-convrot` is **not** the Lightricks QAT variant;
the results may differ. Respect the respective model licenses.

LTX 2.5 uses the [comfyicu/LTX-2.5](https://huggingface.co/comfyicu/LTX-2.5)
mirror of the Lightricks split checkpoints. In particular,
`gemma4-12b-with-proj-ltx-2.5-...safetensors` already **contains** the LTX
text projections. Do **not** pass a separate `--text-projection` for LTX 2.5.
The documented video VAE is the *conv* VAE, not the different DiffVAE.

All required model files are passed **explicitly**. No model is selected from
its filename or from its directory layout. `--audio-vae` and
`--spatial-upscaler` are optional generic component paths, used by
LTX and available to other adapters. LTX requires both, and only split
LTX 2.3 requires `--text-projection`. The Gemma3 tokenizer and minimal
model config are bundled in TheNoise. `--text-encoder` must be one
`.safetensors` file, never a sharded model directory.

### LTX 2.3 INT8 ConvRot

```sh
./thenoise.sh serve \
  --dit ./models/ltx23/checkpoints/ltx-2.3-22b-distilled-1.1-int8-ConvRot.safetensors \
  --vae ./models/ltx23/vae/LTX23_video_vae_bf16.safetensors \
  --audio-vae ./models/ltx23/vae/LTX23_audio_vae_bf16.safetensors \
  --spatial-upscaler ./models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --text-projection ./models/ltx23/text_encoders/ltx-2.3_text_projection_bf16.safetensors \
  --text-encoder ./models/ltx23/gemma3/gemma_3_12B_it_int8_convrot.safetensors \
  --device cuda:0 --offload-device cpu --port 8000
```

### LTX 2.5 INT8 ConvRot

```sh
./thenoise.sh serve \
  --dit ./models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
  --vae ./models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae ./models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upscaler ./models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --text-encoder ./models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
  --device cuda:0 --offload-device cpu --port 8000
```

The shared memory manager moves whole registered components. If the DiT
cannot fit the available accelerator memory even after other components are
evicted, the request may fail with an OOM; this PR deliberately **does not**
add per-block offloading. Use INT8 weights or a device with more memory.

## Video generation

Synchronous `POST /video` and queued `POST /jobs/video` use the shared video
API. Output is MP4 (H.264 + AAC audio). Distilled generation uses eight Euler
steps followed by three refinement steps. First-frame I2V is supported;
last-frame conditioning, runtime LoRAs, negative prompts and CFG != 1 are not.

```sh
curl -sS http://127.0.0.1:8000/video -H 'content-type: application/json' \
  -d '{"prompt":"A fox moving through snow with footsteps", "width":512, "height":512, "frames":33, "fps":24, "steps":8, "guidance_scale":1, "sampler":"euler", "output_format":"mp4"}' \
  -o fox.mp4
```

Image-to-video: supply the reference image as plain base64 in the request's
`image` field (no `data:` prefix). Dimensions are aligned to multiples of 64;
frame counts to the `8n+1` temporal grid.

**Validation status:** The earlier Diffusers-based prototype running on GPU
is **not** evidence that the native port in this PR is correct. CPU/kernel
and checkpoint-ABI tests are useful but real checkpoint loading, T2V and I2V,
audio/video synchronization, and cleanup must be revalidated on GPU for both
releases before this can be considered merge-ready. There is no Diffusers
runtime dependency.
