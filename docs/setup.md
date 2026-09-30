# Setup

Getting from "downloaded the release" to "generated my first image" in three
steps. Everything here works on a machine with no Python, no build tools, and
no sudo.

> Already have the repo checked out? The [Development & Contribution](development.md)
> page covers installing from source instead.

## Requirements

- **Linux x86_64**
- An **AMD GPU** with ROCm support. Targets:
  - **Strix Halo** (gfx1151) - the primary, fully optimized target
  - gfx1150 / gfx1152 - tested
- **RAM:** 32 GB minimum for the small models (Anima ~5 GB, Z-Image ~21 GB);
  64 GB+ recommended; the 128 GB Strix Halo configuration runs everything.
- **Disk:** see the download sizes on each model's page - plan for model files
  plus a few GB of headroom.

You only need one model to get going. Anima (~5.4 GB) is the fastest way to a
first image; Krea 2 (~35 GB) is the quality pick.

## 1. Get TheNoise

Download the **portable** release asset for your GPU from the
[releases page](https://github.com/lemonade-sdk/thenoise/releases). Each bundle
is a self-contained directory with a standalone CPython, the ROCm build of
PyTorch, all dependencies, TheNoise itself, and a bundled `clang` (so
`torch.compile`/Triton JIT works with no system compiler).

```bash
# single archive
tar -xzf thenoise-<version>-rocm<rocm>-<gfx>-x64.tar.gz
cd thenoise-<version>-rocm<rocm>-<gfx>-x64

# split archive (GitHub's 2 GB asset limit) — concatenate the parts first
cat thenoise-*.part*.tar.gz | tar -xz
```

No installation step: the extracted directory *is* TheNoise.

## 2. Download a model

The bundle ships the model download helper (`scripts/download.py`) with
`huggingface_hub` included. Run it with the bundled Python:

```bash
./bin/python3 scripts/download.py --model anima
```

All models and options (variants, `--int8-convrot`, ...):

| Model | Command | Size | Details |
|---|---|---|---|
| Anima | `--model anima` | ~5.4 GB | [model page](models/anima.md) |
| Krea 2 | `--model krea2` | ~35 GB | [model page](models/krea2.md) |
| Z-Image / Z-Image-Turbo | `--model zimage` | ~21 GB | [model page](models/zimage.md) |
| Ming-Image 0.1 | `--model ming-image` | 15–49 GB | [model page](models/ming-image.md) |
| Flux.2 Klein 4B/9B | `--model klein --variant 9b` | 12–25 GB | [model page](models/flux2-klein.md) |
| Qwen-Image / Edit | `--model qwen-image` | ~40 GB | [model page](models/qwen-image.md) |
| Qwen-Image 2.1 | `--model qwen-image-2.1` | ~32 GB | [model page](models/qwen-image-2.1.md) |
| Real-ESRGAN x4 *(optional, upscaling)* | `--model esrgan` | ~0.7 GB | [CLI: upscaling](cli.md#upscaling) |

Models land in `./models/<model>/` next to the bundle.

## 3. Generate your first image

```bash
./bin/thenoise generate \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --prompt "a fox walking in the snow" \
  --out fox.png
```

**The first generation is slow** - the DiT is compiled with `torch.compile` on
load. Expect a minute or two of compilation (and some normal-looking warnings on
the console); every generation after that runs at full speed with the cached
compiled code. No configuration needed.

That's it. `fox.png` is in your current directory.

## Serve it instead

To get the web UI (and the HTTP API other software can call) instead of one-shot
CLI runs:

```bash
./bin/thenoise serve \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --host 127.0.0.1 --port 8000
```

Then open <http://localhost:8000/> - Generate, Edit (editing-capable models) and
Upscale tabs, driven by whatever model is loaded.

## What next

- [CLI reference](cli.md) - every `generate` / `edit` / `serve` / `upscale` flag
- [HTTP API reference](api.md) - endpoints, request/response formats, curl examples
- [Model pages](models/) - per-model download options, examples, settings
- [Upscaling](cli.md#upscaling) - latent refine + Real-ESRGAN pixel upscaling, up to 8×
- [Development & Contribution](development.md) - building from source, tests, releases

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.compile` warnings on first run | Normal. They only happen while the DiT is being compiled. |
| First generation takes minutes | Normal — see [first-run note](#3-generate-your-first-image). Subsequent runs are fast. |
| `error while loading shared libraries` | Don't run the system Python against the bundle; always use `./bin/thenoise` (or `./bin/python3`). |
| `libatomic.so.1: cannot open shared object file` at startup | Missing `libatomic` runtime library (seen on Fedora 44): install it with your package manager, e.g. `sudo dnf install libatomic`. Details in [#29](https://github.com/lemonade-sdk/thenoise/issues/29). |
| GPU not detected | You're on the wrong bundle for your GPU — check `gfx` in the asset name against your hardware (gfx1151 for Strix Halo). |
