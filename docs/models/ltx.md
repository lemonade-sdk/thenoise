# LTX 2.3 and LTX 2.5

Audiovisual video: picture and sound from one run, from a prompt, a first frame, or the
two frames a clip runs between.

| | LTX 2.3 Distilled 1.1 | LTX 2.5 Distilled |
|---|---|---|
| Modes | text-to-video, image-to-video, first-and-last-frame | the same |
| Transformer | 22 B distilled, video and audio in one pack | the same |
| Codec | 128-channel video at 32× spatial / 8× temporal, plus an audio codec and vocoder | the same |
| Text encoder | Gemma 3 12B, connector weights separate | Gemma 4 12B, projections inside |
| Defaults | 768×512, 33 frames, 24 FPS, 8 steps + a 3-step refine, CFG 1.0, Euler, flow shift 1, WebM | the same |

Output is MP4/H.264 or WebM/VP9 on an **`8n+1`** frame grid; sound is AAC in `mp4` and
Opus in `webm`. Size rounds down to multiples of **64 px**, frame count up to the grid
(`--duration` takes the nearest rung).

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`). On a
> [portable bundle](../setup.md) use `./bin/thenoise` and `./bin/python3` instead.

## Performance (Strix Halo)

INT8-ConvRot, @ 8 steps, CFG 1:

- Generate 832×480, 5s clip: **~ 42.3 s**
- Generate 1344×768, 5s clip: **~ 120.2 s**

## Examples

*Generated:*


## Download

```bash
.venv/bin/python scripts/download.py --model ltx23 --int8-convrot   # 43 GB, 72 GB BF16
.venv/bin/python scripts/download.py --model ltx25 --int8-convrot   # 40 GB, 71 GB BF16
```

## Usage

Serve LTX 2.3 (int8-convrot) over HTTP with the web UI (open <http://localhost:8000/>):

```bash
./thenoise.sh serve \
  --dit ./models/ltx23/checkpoints/ltx-2.3-22b-distilled-1.1-int8-ConvRot.safetensors \
  --vae ./models/ltx23/vae/LTX23_video_vae_bf16.safetensors \
  --audio-vae ./models/ltx23/vae/LTX23_audio_vae_bf16.safetensors \
  --spatial-upscaler ./models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --text-projection ./models/ltx23/text_encoders/ltx-2.3_text_projection_bf16.safetensors \
  --text-encoder ./models/ltx23/gemma3/gemma_3_12B_it_int8_convrot.safetensors \
  --port 8000
```

The 2.3 BF16 release carries both codecs, the vocoder and the projections inside the
transformer, so three flags name it all:

```bash
./thenoise.sh serve \
  --dit ./models/ltx23/ltx-2.3-22b-distilled-1.1.safetensors \
  --spatial-upscaler ./models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --text-encoder ./models/ltx23/gemma3/gemma_3_12B_it.safetensors \
  --port 8000
```

LTX 2.5 ships every part apart, and its Gemma carries the text projections — passing
`--text-projection` is refused:

```bash
./thenoise.sh serve \
  --dit ./models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
  --vae ./models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae ./models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upscaler ./models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --text-encoder ./models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
  --port 8000
```

Generate one clip:

```bash
./thenoise.sh video \
  --dit ./models/ltx23/checkpoints/ltx-2.3-22b-distilled-1.1-int8-ConvRot.safetensors \
  --vae ./models/ltx23/vae/LTX23_video_vae_bf16.safetensors \
  --audio-vae ./models/ltx23/vae/LTX23_audio_vae_bf16.safetensors \
  --spatial-upscaler ./models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --text-projection ./models/ltx23/text_encoders/ltx-2.3_text_projection_bf16.safetensors \
  --text-encoder ./models/ltx23/gemma3/gemma_3_12B_it_int8_convrot.safetensors \
  --prompt "A fox moving through snow with footsteps" \
  --width 768 --height 512 --frames 33 --fps 24 --steps 8 --out fox.mp4
```

`--image start.png` starts the clip on a picture — with no `--width`/`--height`, its aspect
ratio sizes the clip — and `--end-image end.png` pins the far end too, the prompt then being
the change between the two. Neither release is an FLF2V build and none is needed: a pinned
frame is written at timestep zero and never denoised, and the clip ends on your image rather
than on your file. Over HTTP those frames go in `image` and `end_image` as base64 of the
file bytes, and a queued render is `POST /jobs/video`.

For every flag (size, frames, fps, steps, seed, LoRAs), see the
[CLI reference](../cli.md) and the [HTTP API reference](../api.md).
