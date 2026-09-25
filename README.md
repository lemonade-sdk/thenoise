<div align="center">
  <img src="thenoise/ui/logo2.png" alt="TheNoise" width="280" />
</div>

TheNoise is an open-source image generation / editing engine made specifically to run well on Strix Halo (gfx1151) and other ROCm-capable AMD iGPUs and dGPUs. It is tuned to perform extremely well on the machine it runs on.  

TheNoise loads one model at a time and generates images from text prompts. Editing-capable models - like Qwen-Image 2.1, Qwen-Image-Edit and FLUX.2 Klein - can also edit an existing image from a text instruction (image + prompt → edited image).

TheNoise can be used standalone, from the command line or through a webui or through an OpenAI-compatible server like [Lemonade](https://lemonade-server.ai/docs/dev/backends-reference/#backends), with which it is already integrated.

<details open id="shot-main">
  <summary>Main</summary>
  <img width="2048" height="1066" alt="thenoise-main-screenshot" src="https://github.com/user-attachments/assets/afaf2d89-5857-4f50-995f-06fdf556a3c4" />
</details>

<details id="shot-edit">
  <summary>Edit</summary>
  <img width="2048" height="1066" alt="thenoise-edit" src="https://github.com/user-attachments/assets/17efedda-b887-4f87-b0c0-c151619b19ac" />
</details>

<details id="shot-upscale">
  <summary>Upscale</summary>
  <img width="2048" height="1066" alt="thenoise-upscaler" src="https://github.com/user-attachments/assets/f7ce89b7-fd25-4ad9-a3e4-d5e367530ab7" />
</details>

---
## Features

TheNoise ships: 

- image generation / editing support for major open-weights models,
- a built-in 2× refiner-based (SesquiLSR) upscaler - fast and high-quality upscaling without loading extra model files,
- pixel-space upscalers (Real-ESRGAN) up to 4× - standard ESRGAN-based models,
- film grain and RCAS sharpening as post-processing,
- LoRA support - one or more LoRAs per image, each with its own weight.

## Supported models

| Model | Generate | Edit | Details |
|---|---|---|---|
| **Anima**, small and fast | ✓ | — | [anima](docs/models/anima.md) |
| **Krea 2**, highest image quality | ✓ | — | [krea2](docs/models/krea2.md) |
| **Z-Image / Z-Image-Turbo**, quality at 8 steps | ✓ | — | [zimage](docs/models/zimage.md) |
| **Flux.2 Klein 4B / 9B**, 4 steps, with editing | ✓ | ✓ | [flux2-klein](docs/models/flux2-klein.md) |
| **Qwen-Image / Qwen-Image-Edit**, generation and editing | ✓ | ✓ | [qwen-image](docs/models/qwen-image.md) |
| **Qwen-Image 2.1**, generation and editing in one model | ✓ | ✓ | [qwen-image-2.1](docs/models/qwen-image-2.1.md) |

New models are added over time. PRs adding model support are welcome.

## How does it compare to ComfyUI?

ComfyUI is a general-purpose, node-based framework and remains the better
choice for advanced, customizable workflows. TheNoise is a focused engine, and
it is a good fit when:

- you are running a Strix Halo and would like to start generating images quickly, without having to care about "workflows",
- you prefer a simple command line or a small UI over building and maintaining workflows,
- you want a small, stable image-generation endpoint that other software can call,
- you would rather have an engine optimized for your hardware than a general-purpose one.

## Performance

TheNoise is tuned for the hardware it runs on, and its performance is quite
similar to ComfyUI's - and sometimes even better. The numbers below are
seconds per image on a Strix Halo (gfx1151, 128 GB unified), measured after
a warmup run.

*Text to image (generation):*

| Model & settings | 1024×1024 | 1536x2048 |
|---|---|---|
| Krea 2 Turbo · BF16 · 8 steps | 33.8s | 117.6s |
| Krea 2 Turbo · INT8-ConvRot · 8 steps | 26.7s | 99s |
| Anima Base · 20 steps · CGF 3 | 29.7s | 111s |
| Anima Turbo · 8 steps | 6.6s | 25.1s |
| Z-Image Turbo · 8 steps | 14.3s | 53.8s |
| Flux.2 Klein 9B · INT8-ConvRot · 4 steps | 9.6s | 34.3s |
| Qwen-Image 2512 · BF16 · 4 steps | 9.7s | 34.9s |
| Qwen-Image 2.1 + [Qwen-Image-2.1 Turbo LoRA](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) · BF16 · 6 steps | 15s | 53s |

*Image + simple instruction to edited image (editing):*

| Model & settings | 1024×1024 · KV-cache OFF | 1024×1024 · KV-cache ON |
|---|---|---|
| Flux.2 Klein 9B · INT8-ConvRot · 4 steps | 14.2s | 10.6s |
| Qwen-Image-Edit 2511 · BF16 · 4 steps | 15.8s | 10.6s |
| Qwen-Image 2.1 · BF16 · 4 steps | 15.4s | 9.8s |
| Qwen-Image 2.1 · INT8-ConvRot · 4 steps | 14.4s | 9.2s |

<small>TheNoise 0.9.0, Strix Halo (gfx1151, 128 GB unified)</small>

## Quick start

Install TheNoise and generate your first image in a few commands. The full walkthrough is in [docs/setup.md](docs/setup.md):

```bash
# 1. grab the portable bundle for your GPU from the releases page, extract it
tar -xzf thenoise-<version>-rocm<rocm>-gfx1151-x64.tar.gz
cd thenoise-<version>-rocm<rocm>-gfx1151-x64

# 2. download a model
./bin/python3 scripts/download.py --model anima

# 3. generate
./bin/thenoise generate \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --prompt "a fox walking in the snow" --out fox.png
```

The portable bundle is self-contained: it needs no Python installation, build tools, or administrator rights on the target machine.

## Using TheNoise

| If you want to… | Use |
|---|---|
| generate an image from the command line | the [CLI](docs/cli.md) - `generate`, `edit`, `upscale` |
| work through a browser | the web UI at `http://localhost:8000/` when running `serve` |
| call it from other software | the [HTTP API](docs/api.md) - `/text2image`, `/edit`, `/upscale` |

## Documentation

| | |
|---|---|
| [**Setup**](docs/setup.md) | from a released build to your first image |
| [**CLI reference**](docs/cli.md) | all `generate` / `edit` / `serve` / `upscale` flags, upscaling, LoRAs |
| [**HTTP API**](docs/api.md) | endpoints, request/response reference, curl and Python examples |
| [**Development & Contribution**](docs/development.md) | building from source, tests, portable builds, adding models |

## Acknowledgments

This project incorporates code from:

1. [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner)
2. [SD Scripts](https://github.com/kohya-ss/sd-scripts)
3. [SesquiLSR](https://github.com/LoganBooker/SesquiLSR)

plus smaller snippets from other sources or transitively inherited through the
above codebases.
