# MiniMax-H3 Video (fl2va)

Video with synchronized sound from a text prompt, a starting image, or the two frames a
clip runs between — the `fl2va` release, which answers `t2v`, `i2v` and `flf2v` from one
checkpoint. Use the normal `/video` or `/jobs/video` endpoints. Picture and sound come out
of the same run: the transformer holds both streams in one pack, so a clip's audio is
generated rather than fitted afterwards. Sound lands in `mp4` as AAC and in `webm` as Opus.

20 steps at guidance 1.0 with Euler is the released recipe and the default; ask for another
step count, solver, guidance scale or negative prompt and you get that instead.

## Download models

```sh
python scripts/download.py --model h3-video --int8-convrot
```

Omit `--int8-convrot` for the BF16 pruned DiT (37 GB against 20 GB). `--include-turbo` adds
the two 768p turbo LoRAs described below. The text encoder is the int8-convrot Qwen3-VL-32B
the image release ships too — 25 GB, published quantized only — so an existing
`models/h3-image` copy can be passed to `--text-encoder` instead of downloading it again.

Both codecs are needed: `--vae` is the clip codec, `--audio-vae` the sound one, and a
request missing either is refused before anything loads. They are a pair — a codec whose
latent has fewer or more channels than the transformer emits is refused by name.

## Start it

```sh
./thenoise.sh serve \
  --dit ./models/h3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --vae ./models/h3/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae ./models/h3/vae/minimax_h3_audio_vae_fp32.safetensors \
  --text-encoder ./models/h3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --port 8000
```

## Generate a clip

```sh
curl -sS http://127.0.0.1:8000/video -H 'content-type: application/json' \
  -d '{"prompt":"a lighthouse sweeping over dark water, waves breaking, the beam humming",
       "width":1344,"height":768,"frames":124,"fps":24,"steps":20,"guidance_scale":1,
       "sampler":"euler","output_format":"mp4"}' \
  -o lighthouse.mp4
```

The sound belongs to the clip rather than to a file format: it is muxed into `mp4` and
`webm` alike, and cut or padded to the clip's length. Its own rate is the codec's 32 kHz,
which travels with it — `webm` encodes it as Opus at 48 kHz, resampled on the way in, since
Opus will not open at 32.

### Image-to-video, and both frames

Add `image` for a clip that starts on a picture, and `end_image` — same encoding, same rules
— for one that ends on another. Both are cover-cropped to the clip's canvas, and the first
frame decides the size when the request names none. Neither is pasted into the output: each
enters the run as conditioning at a fixed timestep, and what you see at the ends is the
model's own rendering of them, which is what makes a last frame possible in a checkpoint that
was never an FLF2V build.

## Frames and canvas

A clip is not any length. The codec runs 17 pixel frames per latent group, opening with one
frame and then four per latent, so a clip is `17k` frames or `17k + 1, 5, 9, 13`: `5, 9, 13,
17, 18, 22, 26, 30, 34, 35, 39 … 124 …`. `--frames` rounds up to that ladder — a clip shorter
than the first rung would have nothing to decode — and `--duration` takes the nearest rung.
At 24 fps the released 124 frames are 5.17 seconds.

Width and height round **down** to multiples of 32. Asking for more area than the 1344×768
the release was trained on logs a warning rather than rewriting the request: the rope
normalizes by area, so a bigger canvas runs — it is just off distribution.

There is no latent upscaler for these 24-channel clip latents, so `upscale` is off; the
pixel-domain upscalers (`--upscaler-dir`) work on any model's output.

## Turbo LoRAs

The 768p turbos are distilled at **video flow shift 6** while the base release runs at 12, so
a recipe that leaves `flow_shift` alone will be sampling a distilled model on the wrong
ladder:

```sh
./thenoise.sh video \
  --dit ./models/h3/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --vae ./models/h3/vae/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae ./models/h3/vae/minimax_h3_audio_vae_fp32.safetensors \
  --text-encoder ./models/h3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors \
  --lora ./models/h3/loras/minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16.safetensors \
  --prompt "a courier bike cutting through wet neon streets, tyres hissing" \
  --width 1344 --height 768 --frames 124 --steps 4 --flow-shift 6
```

The audio's own shift is 3.0 in every release and every turbo, and it stays there: what
`--flow-shift` changes is the video's schedule, and the ratio between the two — which is how
the audio travels inside the pack — follows automatically. `--flow-shift 6` therefore asks
for half the carry of the default 12 while the sound is denoised on exactly the ladder it
would be at 12.

`--include-turbo` fetches the 4-step v1.3 and the 8-step v1.1, both trained at 768p. The
newer 8-step v1.2 is deliberately not among them: besides the attention and MLP projections
every other turbo touches, it carries 51 `adaln_proj.linear` targets whose input is 2688 wide
— the *unpruned* trunk's width, which a pruned checkpoint does not have — so there is nothing
for it to attach to. `ref2v` turbos belong to the reference-to-video release, which is not
wired up here yet.

## What it is

| | |
|---|---|
| Transformer | ~20 B, 50 blocks + 2 token refiners, 56×128 heads |
| Picture latent | 24 channels, 16× spatial, 4× temporal |
| Sound latent | 32 channels, 40 frames per second, stereo |
| Vocoder | BigVGAN-style, 32 kHz out |
| Text encoder | Qwen3-VL-32B, int8-convrot; a picture arrives as vision tokens |
| Released defaults | 1344×768, 124 frames at 24 fps, 20 steps, guidance 1.0, Euler, flow shift 12 |

On a GPU where `comfy_kitchen` compiled its block-sparse attention kernel, the trunk's
attention is routed rather than dense: each 64-row query block attends the key blocks that
matter exactly and pools the rest, with every prompt, keyframe and audio row held exact for
every query because pooling those is how a run loses them. At the release's own shape that
measured 4.6x on the attention call (gfx1151, bf16, tau 1.3). There is no option for it — a
run uses it when its tensors fit and says so once in the log, naming the token count and the
sink ranges, or naming why it stayed dense.

Two more releases exist upstream and are refused rather than half-supported: the unpruned
checkpoints, which condition each block from a 2688-wide timestep embedding (24 GB of the
file) where this engine reads the pruned release's distilled curve; and the `sol` codec,
whose decoder is a different tree from the one implemented here and is refused by name
instead of being loaded onto the wrong modules.
