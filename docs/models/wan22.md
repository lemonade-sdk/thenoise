# Wan 2.2

Three released Wan 2.2 transformers are supported, in two families that are
different models rather than two sizes of one.

| | TI2V-5B | T2V-A14B |
|---|---|---|
| Modes | text-to-video, image-to-video | text-to-video, or image-to-video with the I2V pair, which also takes a last frame |
| Transformer | one checkpoint (`--dit`) | two experts: `--dit` high noise, `--dit-low` low noise |
| Latent and codec | 48 channels, Wan 2.2 causal 3D VAE | 16 channels, Wan 2.1 causal 3D VAE |
| Text encoder | UMT5-XXL | UMT5-XXL |
| Samplers | UniPC or Euler | UniPC or Euler |
| Defaults | 1280×704, 33 frames, 24 FPS, 12 steps, CFG 5.0, UniPC, WebM, flow shift 5 | 832×480, 33 frames, 16 FPS, 40 steps, Euler, CFG 4.0, flow shift 12, high/low boundary 0.875 (the I2V pair: shift 5, boundary 0.900, CFG 3.5 on both halves) |

Each of the three is downloadable in two forms: the **turbo** one the downloader fetches
by default, distilled to 4 steps at CFG 1.0, and the **base** release the Defaults row
describes. [What to send](#what-to-send) keeps them straight.

Both answer the same video request, from the command line with
[`video`](../cli.md#generate-a-single-clip) or over the
[HTTP API](../api.md#video-generation), where `POST /video` returns the clip and
`POST /jobs/video` queues a job. Output is MP4/H.264 or WebM/VP9 on the **`4n+1`**
temporal grid, with no fixed maximum frame count.

> Commands below assume a dev checkout (`./thenoise.sh`, `.venv/bin/python`).
> On a [portable bundle](../setup.md) use `./bin/thenoise` and
> `./bin/python3` instead.

## Performance (Strix Halo)

I2V-A14B, BF16, @ 4 steps, CFG 1, flow shift 5:

- Generate 832x480: **~161.7 s**
- Generate 1344x768: **~731.4 s**


## TI2V-5B

Native text-to-video and image-to-video. The downloaded DiT is FastWan by default, the
distill of the Comfy-Org Wan 2.2 5B checkpoint: same geometry, same codec, most of its
steps distilled away. `--variant base` takes that checkpoint instead. The two modes are
the same model and the same request: `--image` selects one,
`prompt` describes what happens after the first frame.

### Download

```bash
uv pip install -e ".[scripts]"
.venv/bin/python scripts/download.py --model wan22
```

The three files, about 23 GB together:

```
models/wan22/split_files/
  diffusion_models/fastwan2.2_ti2v-5b_bf16.safetensors
  text_encoders/umt5_xxl_fp16.safetensors
  vae/wan2.2_vae.safetensors
```

`--variant base` takes the FP16 release, and `--int8-convrot` the int8-convrot form of
the turbo DiT and of the encoder. On the base release `--int8-convrot` reaches the
encoder alone, that DiT being published in FP16 only:

```bash
.venv/bin/python scripts/download.py --model wan22 --variant base
.venv/bin/python scripts/download.py --model wan22 --int8-convrot
```

### Serve

Start the server using the ROCm-configured launcher (which sets
`MIOPEN_FIND_MODE=FAST` on AMD GPUs):

```bash
./thenoise.sh serve \
  --dit ./models/wan22/split_files/diffusion_models/fastwan2.2_ti2v-5b_bf16.safetensors \
  --text-encoder ./models/wan22/split_files/text_encoders/umt5_xxl_fp16.safetensors \
  --vae ./models/wan22/split_files/vae/wan2.2_vae.safetensors \
  --port 8000
```

### Text-to-video

One clip, no server: `video` loads the same three files, renders, and writes the
file `--out` names. Every other flag is optional — what the release ships as its
defaults is in the table above.

```bash
./thenoise.sh video \
  --dit ./models/wan22/split_files/diffusion_models/fastwan2.2_ti2v-5b_bf16.safetensors \
  --text-encoder ./models/wan22/split_files/text_encoders/umt5_xxl_fp16.safetensors \
  --vae ./models/wan22/split_files/vae/wan2.2_vae.safetensors \
  --prompt "A fox walking through snow" \
  --width 704 --height 480 --frames 33 --steps 4 --guidance-scale 1 --sampler unipc \
  --out fox.mp4
```


### Image-to-video

`--image` names the image the clip starts on, and the prompt says what happens
after it. With `--width`/`--height` left out the image's own aspect ratio decides
the clip's shape.

```bash
./thenoise.sh video \
  --dit ./models/wan22/split_files/diffusion_models/fastwan2.2_ti2v-5b_bf16.safetensors \
  --text-encoder ./models/wan22/split_files/text_encoders/umt5_xxl_fp16.safetensors \
  --vae ./models/wan22/split_files/vae/wan2.2_vae.safetensors \
  --image reference.png \
  --prompt "The fox starts walking forward while the camera tracks it" \
  --width 704 --height 480 --frames 33 --fps 24 --steps 4 --guidance-scale 1 --sampler unipc \
  --out fox_i2v.mp4
```

TheNoise cover-crops the frame to the clip's canvas, encodes it with the video VAE
and conditions the clip's first frame on it. Over HTTP that frame goes in the
`image` field as base64 of the file bytes — not a `data:image/...;base64,` URL and
not a path — see [HTTP API](../api.md#image-to-video-i2v), and
[Asynchronous jobs](../api.md#asynchronous-jobs) to queue a render instead of waiting
for it.

## Examples

*Generated:*

[T2V](https://github.com/user-attachments/assets/961dcaf3-4bcc-4c40-93e9-1a6c548dfc64)

[I2V](https://github.com/user-attachments/assets/3dd84d51-af65-478c-a7c5-1f1805cc1cea)


## T2V-A14B (two experts)

Two separately loaded DiTs (high- and low-noise), the **Wan 2.1 16-channel VAE**
(`wan_2.1_vae.safetensors` — Wan 2.1 by latent format, not a version mismatch: it is the
codec Qwen-Image uses for stills), and the same UMT5 encoder as 5B.

Pass **both expert files explicitly**: `--dit` selects high noise and `--dit-low`
selects low noise. File names and directories are not inspected to infer the expert
role. The 4-step int8-convrot pair and ComfyUI's FP16 pair are both supported. The
latent width is read off the `--vae` file, and a codec that does not match the
transformer is refused at load rather than at the first denoising step.

Both experts are swapped by the normal device memory manager at the sigma boundary,
which keeps both on the card when they fit; do not run both as resident GPU models on a
32 GiB card. The single `guidance_scale` a request carries is the high-noise CFG (4.0 by
default); low-noise CFG is 0.75× that (3.0 by default).

`--image` is refused by these two files: image-to-video is
[its own pair of experts](#i2v-a14b).

### Download

```bash
.venv/bin/python scripts/download.py --model wan22-14b
.venv/bin/python scripts/download.py --model wan22-14b --variant base
```

The default is the 4-step pair, int8-convrot because that is how it is published;
`--variant base` takes ComfyUI's FP16 experts and `--int8-convrot` swaps the encoder for
its int8-convrot form:

```
models/wan22/split_files/
  diffusion_models/wan2.2_t2v_high_noise_14B_4step_int8_convrot.safetensors
  diffusion_models/wan2.2_t2v_low_noise_14B_4step_int8_convrot.safetensors
  text_encoders/umt5_xxl_fp16.safetensors
  vae/wan_2.1_vae.safetensors
```

### Serve and generate

```bash
./thenoise.sh serve \
  --dit ./models/wan22/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_4step_int8_convrot.safetensors \
  --dit-low ./models/wan22/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_4step_int8_convrot.safetensors \
  --text-encoder ./models/wan22/split_files/text_encoders/umt5_xxl_fp16.safetensors \
  --vae ./models/wan22/split_files/vae/wan_2.1_vae.safetensors \
  --port 8000
```

Or one clip, with both experts passed the same way:

```bash
./thenoise.sh video \
  --dit ./models/wan22/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_4step_int8_convrot.safetensors \
  --dit-low ./models/wan22/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_4step_int8_convrot.safetensors \
  --text-encoder ./models/wan22/split_files/text_encoders/umt5_xxl_fp16.safetensors \
  --vae ./models/wan22/split_files/vae/wan_2.1_vae.safetensors \
  --prompt "A fox walking through snow" \
  --width 704 --height 480 --frames 33 --steps 4 --guidance-scale 1 \
  --out fox_14b.mp4
```

## Examples

*Generated:*

[Example 1](https://github.com/user-attachments/assets/74b234b5-3d80-418c-9165-b750eb7338f9)

[Example 2](https://github.com/user-attachments/assets/ba732be9-b729-479e-86a0-e537f5a82918)

### One LoRA per expert

Each expert is LoRA-able on its own — `--lora` for the one `--dit` names,
`--lora-low` for the one `--dit-low` names, both against a `--lora-dir`:

```bash
  --lora wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise:1.0 \
  --lora-low wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise:1.0
```

The 4-step pair has its distillation baked in already, so that recipe is for the
`--variant base` experts.

## I2V-A14B

The image-to-video A14B pair is the T2V trunk with a wider door: 36 channels, the 20
extra being a mask and the reference clip concatenated onto the noise at every step.
Two experts, Euler, the same LoRA roles, and no CLIP vision tower anywhere in it. Its
release numbers are its own — see [what to send](#what-to-send). Send it two images and
the same door takes a clip that ends as well as starts — see
[first-and-last-frame](#first-and-last-frame-flf2v).

### Download

```bash
.venv/bin/python scripts/download.py --model wan22-14b-i2v
```

Turbo by default; `--variant base` and `--int8-convrot` mean what they mean for
[T2V-A14B](#t2v-a14b-two-experts):

```
models/wan22/split_files/
  diffusion_models/wan2.2_i2v_high_noise_14B_4step_int8_convrot.safetensors
  diffusion_models/wan2.2_i2v_low_noise_14B_4step_int8_convrot.safetensors
  text_encoders/umt5_xxl_fp16.safetensors
  vae/wan_2.1_vae.safetensors
```

### Serve and generate

```bash
./thenoise.sh serve \
  --dit ./models/wan22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_4step_int8_convrot.safetensors \
  --dit-low ./models/wan22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_4step_int8_convrot.safetensors \
  --vae ./models/wan22/split_files/vae/wan_2.1_vae.safetensors \
  --text-encoder ./models/wan22/split_files/text_encoders/umt5_xxl_fp16.safetensors
```

```bash
./thenoise.sh video \
  --dit ./models/wan22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_4step_int8_convrot.safetensors \
  --dit-low ./models/wan22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_4step_int8_convrot.safetensors \
  --vae ./models/wan22/split_files/vae/wan_2.1_vae.safetensors \
  --text-encoder ./models/wan22/split_files/text_encoders/umt5_xxl_fp16.safetensors \
  --image reference.png --prompt "a fox in the snow" \
  --steps 4 --guidance-scale 1 \
  --out fox_i2v.mp4
```

Omit `--width` and `--height` and the reference decides the clip's size, as it does for
an image edit: its aspect ratio at the 832×480 this variant defaults to. The 5B path
conditions differently — its first frame is denoised at timestep zero instead of being
concatenated — but asks for the same request.

### First-and-last-frame (FLF2V)

The same two experts and no third download: pass `--end-image` beside `--image` and the
clip is pinned at both ends, the model inventing the change between them. Wan 2.2 has no
FLF2V release and needs none — a concatenated reference carries a mask naming which of
its frames are given, so lighting its far end as well as its near one is a request, not a
checkpoint.

```bash
./thenoise.sh video \
  --dit ./models/wan22/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_4step_int8_convrot.safetensors \
  --dit-low ./models/wan22/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_4step_int8_convrot.safetensors \
  --vae ./models/wan22/split_files/vae/wan_2.1_vae.safetensors \
  --text-encoder ./models/wan22/split_files/text_encoders/umt5_xxl_fp16.safetensors \
  --image first.png --end-image last.png \
  --prompt "the fox looks up, the snow thickens and the camera pulls back" \
  --steps 4 --guidance-scale 1 \
  --out fox_flf2v.mp4
```

Both frames are cover-cropped to the clip's canvas, the first deciding its size when the
request names none, so write the prompt as the change from one to the other. The endpoints
are conditioning rather than a paste — the clip ends where the run landed, not on your file
byte for byte — and asking I2V weights for a frame they were never trained to mark costs
some certainty in the middle. ComfyUI drives this same conditioning on these files, at 81
frames by default.

## Examples

*Generated:*

[Example 1 - P2V](https://github.com/user-attachments/assets/b8d1968e-c275-4a13-89f6-721d4eb40e70)

[Example 2 - D2V](https://github.com/user-attachments/assets/431bfb0c-6aae-4eed-bf04-783340c6a517)

## What to send

What a request leaves unset falls back to the **base** releases' numbers:

| Run | `steps` | `flow_shift` | `guidance_scale` |
|---|---|---|---|
| TI2V-5B | 12 | 5 | 5.0 |
| T2V-A14B | 40 | 12 | 4.0 |
| I2V-A14B | 40 | 5 | 3.5 |

The turbo checkpoints are step-distilled, and none of them says so in its header, so a
run on one names its own three numbers: **4 steps, flow shift 5, CFG 1.0** — the same
recipe for FastWan 5B and for both 4-step A14B pairs.

Each half of a pair takes its own LoRA file, named for the mode of the pair it belongs
to; the 4-step pair needs none. At CFG 1.0 the negative prompt is not sent at all.
