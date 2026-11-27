# MiniMax-H3 Video

Video with synchronized sound, from a prompt, a first frame, or the two frames a clip runs
between. Picture and sound come out of the same run, not stitched afterwards.

| | fl2va | ref2va |
|---|---|---|
| Modes | text-to-video, image-to-video, first-and-last-frame | the same, plus [references](#references) |
| Transformer | pruned ~20 B trunk, both streams in one pack | the same trunk, with room for references |
| Codec | 24-channel video at 16× spatial / 4× temporal; 32-channel stereo sound at 40 latent frames a second, vocoded at 32 kHz | the same |
| Text encoder | Qwen3-VL-32B, int8-convrot | the same |
| Defaults | 1344×768, 124 frames at 24 FPS, 20 steps, CFG 1.0, Euler, flow shift 12 | the same |

Output is MP4/H.264 or WebM/VP9. Sound is AAC in `mp4` and Opus in `webm`, cut or padded
to the clip's length.

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`). On a
> [portable bundle](../setup.md) use `./bin/thenoise` and `./bin/python3` instead.

## Performance (Strix Halo)

BF16 + [6-steps Pika LoRA](https://huggingface.co/Kijai/MiniMax-H3_comfy/tree/main/loras/Pika), @ 6 steps, CFG 1:

- Generate 832x480, 5s clip: **~ 194.5 s**
- Generate 1344x768, 5s clip: **~ 565.8 s**

## Examples

*Generated:*

[T2V](https://github.com/user-attachments/assets/d3a402d3-ee3f-457b-9443-d3f47d84e027)

[I2V](https://github.com/user-attachments/assets/d4f80eba-bbac-4834-88ee-65876d5f0fc8)

[Ref2V](https://github.com/user-attachments/assets/50be8696-6a77-41f1-9484-b975ab1a3062)


## Frames and canvas

The codec runs 17 pixel frames per latent group, so a clip is `17k` frames or
`17k + 1, 5, 9, 13`: `17, 18, 22, … 34, 35, 39 … 124 …`. `--frames` rounds up to that
ladder, `--duration` takes the nearest rung; 124 frames at 24 fps are 5.17 seconds. Width
and height round **down** to multiples of 32, and beyond the trained 1344×768 area runs but
logs a warning. There is no latent upscaler for these latents; the pixel-domain upscalers
(`--upscaler-dir`) work on any model's output.

## Download

```bash
.venv/bin/python scripts/download.py --model h3-video --int8-convrot   # 54 GB, 73 GB BF16
```

`--include-turbo` adds the two 768p turbo LoRAs. The Qwen3-VL-32B encoder — 25 GB,
published quantized only — is the one H3-Image ships, so an existing `models/h3-image`
copy can be passed to `--text-encoder` instead of downloading it twice.

## Usage

Serve over HTTP with the web UI (open <http://localhost:8000/>). Both codecs are required.

```bash
./thenoise.sh serve \
  --dit ./models/h3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --vae ./models/h3/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae ./models/h3/vae/minimax_h3_audio_vae_fp32.safetensors \
  --text-encoder ./models/h3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --port 8000
```

Generate one clip:

```bash
./thenoise.sh video \
  --dit ./models/h3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --vae ./models/h3/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae ./models/h3/vae/minimax_h3_audio_vae_fp32.safetensors \
  --text-encoder ./models/h3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --prompt "a lighthouse sweeping over dark water, waves breaking, the beam humming" \
  --width 1344 --height 768 --frames 124 --fps 24 --steps 20 --out lighthouse.mp4
```

`--image start.png` starts the clip on a picture and `--end-image end.png` pins the far end
too. Neither is pasted into the output — each enters as conditioning at a fixed timestep,
which is what makes a last frame possible in a checkpoint that was never an FLF2V build.
With no `--width`/`--height`, the first frame's aspect ratio sizes the clip. Over HTTP the
two frames go in `image` and `end_image` as base64 of the file bytes, and a queued render
is `POST /jobs/video`.

### References

`ref2va` also takes pictures the clip should look like without being frames of it, so
`--image` and `--reference` answer different questions and can be combined. The downloader
fetches `fl2va` only, so the `--dit` below is a
[`ref2va` checkpoint](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/diffusion_models)
you fetched yourself; its name is the fl2va one with `fl2va` swapped.

```bash
./thenoise.sh video \
  --dit ./models/h3/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
  --vae ./models/h3/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae ./models/h3/vae/minimax_h3_audio_vae_fp32.safetensors \
  --text-encoder ./models/h3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --prompt "<Picture 1> trotting through a garden, late afternoon" \
  --reference dog.png --width 1344 --height 768 --frames 124 --out dog.mp4
```

Name them in the prompt — `<Picture 1>`, `<Picture 2>` — in the order sent. `--reference
path@FRAME` (or `{"media": ..., "frame": n}` over HTTP) pins one to that frame of the clip,
negative counting back from the last, making it an anchor rather than a resemblance.

A reference is **resized and never cropped**: it keeps its own shape and packs on a grid of
its own ahead of the clip, the canvas capping only its area. A keyframe is different — it
*is* a frame of the clip, so it is cover-cropped to it. Name no size and the first
reference's aspect sizes the run. Every other video model refuses references outright;
`GET /health` says whether the loaded one answers `ref2v`.

### Turbo LoRAs

Both 768p turbos `--include-turbo` fetches — 4-step v1.3 and 8-step v1.1 — are distilled at
**flow shift 6**, not the release's 12, and say so nowhere in the file:

```bash
./thenoise.sh video \
  --dit ./models/h3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --vae ./models/h3/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae ./models/h3/vae/minimax_h3_audio_vae_fp32.safetensors \
  --text-encoder ./models/h3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --lora ./models/h3/loras/minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16.safetensors \
  --prompt "a courier bike cutting through wet neon streets, tyres hissing" \
  --width 1344 --height 768 --frames 124 --steps 4 --flow-shift 6
```

The sound keeps its own shift of 3.0 in every release and turbo: `--flow-shift` moves the
video schedule and the audio's carry follows. The newer 8-step v1.2 is not offered — it
carries 51 `adaln_proj.linear` targets 2688 wide, the *unpruned* trunk's width, so a pruned
checkpoint has nothing for them to attach to.

## Notes

`--attention optimized` runs attention through the block-sparse kernel where
`comfy_kitchen` compiled it — 4.6x on the attention call at the release's own shape
(gfx1151, bf16, tau 1.3). `auto`, the default, routes only clips long enough for it to pay,
`full` never does; a box without the kernel does not answer `optimized` at all.

The unpruned checkpoints and the `sol` codec are refused rather than half-supported: the
first condition each block from a 2688-wide timestep embedding, the second's decoder is a
different tree from the one implemented here.

For every flag (size, frames, fps, steps, seed, references, LoRAs, attention), see the
[CLI reference](../cli.md) and the [HTTP API reference](../api.md).
