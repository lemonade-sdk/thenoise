# LTX 2.3 and LTX 2.5 (distilled)

Generate MP4 video with synchronized audio from a text prompt, a starting image, or
the two frames a clip runs between. Use the normal `/video` or `/jobs/video` endpoints.
Supported: T2V, first-frame I2V, first-and-last-frame I2V, and the released recipe of
eight Euler sampling steps plus a three-step spatial refinement. Ask for another step
count, solver, guidance scale or negative prompt and you get that instead: the released
numbers are the default, not a floor. Sound lands in `mp4` as AAC and in `webm` as Opus.

## Download models

```sh
python scripts/download.py --model ltx23 --int8-convrot
python scripts/download.py --model ltx25 --int8-convrot
```

Omit `--int8-convrot` for BF16 releases if you have sufficient memory. Respect
the individual model licenses. LTX 2.5 files are distributed from the
[comfyicu/LTX-2.5](https://huggingface.co/comfyicu/LTX-2.5) repository;
LTX 2.3 BF16 uses the official Lightricks monolith. All model checkpoints
are specified explicitly as individual `.safetensors` files, not directories.

## Start LTX 2.3 INT8 ConvRot

```sh
./thenoise.sh serve \
  --dit ./models/ltx23/checkpoints/ltx-2.3-22b-distilled-1.1-int8-ConvRot.safetensors \
  --vae ./models/ltx23/vae/LTX23_video_vae_bf16.safetensors \
  --audio-vae ./models/ltx23/vae/LTX23_audio_vae_bf16.safetensors \
  --spatial-upscaler ./models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --text-projection ./models/ltx23/text_encoders/ltx-2.3_text_projection_bf16.safetensors \
  --text-encoder ./models/ltx23/gemma3/gemma_3_12B_it_int8_convrot.safetensors \
  --port 8000
```

## Start LTX 2.5 INT8 ConvRot

```sh
./thenoise.sh serve \
  --dit ./models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors \
  --vae ./models/ltx-2.5/vae/ltx-2.5-video-vae-conv-bf16.safetensors \
  --audio-vae ./models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
  --spatial-upscaler ./models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
  --text-encoder ./models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors \
  --port 8000
```

LTX 2.5 Gemma weights include the text projections; do not specify
`--text-projection`.

## Start LTX 2.3 BF16

The released BF16 checkpoint holds the transformer, both codecs, the vocoder and
the text projections, so one `--dit` names all of them:

```sh
./thenoise.sh serve \
  --dit ./models/ltx23/ltx-2.3-22b-distilled-1.1.safetensors \
  --spatial-upscaler ./models/ltx23/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --text-encoder ./models/ltx23/gemma3/gemma_3_12B_it.safetensors \
  --port 8000
```

Each part is read from whichever file the request gave that carries it, so a
checkpoint holding one of them separately is named with its own flag and takes
that part from the file above. A part missing from every file given is refused
by name, flag included.

## Generate a clip

```sh
curl -sS http://127.0.0.1:8000/video -H 'content-type: application/json' \
  -d '{"prompt":"A fox moving through snow with footsteps", "width":512, "height":512, "frames":33, "fps":24, "steps":8, "guidance_scale":1, "sampler":"euler", "output_format":"mp4"}' \
  -o fox.mp4
```

For image-to-video, add the first frame as base64 in `image`. Video dimensions
are aligned to multiples of 64 pixels, and frame counts to `8n+1`.

### First-and-last-frame (FLF2V)

Add `end_image` — same encoding, same rules — and the clip is pinned at both ends, the
model inventing the change between them. Neither LTX release is an FLF2V build and none
is needed: the frame a clip starts on is already a latent frame the run never denoises,
so lighting its far end the same way is a request rather than a checkpoint.

```bash
FIRST="$(base64 -w 0 first.png)"
LAST="$(base64 -w 0 last.png)"
curl -fS http://127.0.0.1:8000/video -H 'content-type: application/json' \
  --data-binary "{\"prompt\":\"the fox looks up, the snow thickens and the camera pulls back\",\"image\":\"$FIRST\",\"end_image\":\"$LAST\",\"width\":512,\"height\":512,\"frames\":33,\"fps\":24,\"steps\":8,\"output_format\":\"mp4\"}" \
  -o fox_flf2v.mp4
```

Both frames are cover-cropped to the clip's canvas, the first deciding its size when the
request names none. Each is encoded at both of the recipe's resolutions, so an ending
survives the internal spatial upsample as surely as a beginning does. The endpoints are
conditioning rather than a paste: the codec's round trip, and the frames around them
being generated, mean the clip ends on your image rather than on your file. Sound spans
the whole clip either way — a last frame is something the video half is told.
