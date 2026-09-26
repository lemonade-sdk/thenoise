# Development & Contribution

Everything for working on TheNoise itself: installing from a git clone, running
the test suite, building the portable release bundles, and contributing.

> End users should follow [Setup](setup.md) instead — the released portable
> bundles need none of this.

## Repository layout

```
thenoise/                  the package
  models/                  per-model adapters (detect + DEFAULT_PREFS + sampling)
  dit/                     per-model DiT code (krea2, qwen_image, qwen_image21, anima, zimage, flux2)
  vae/                     VAE loaders (Qwen-Image, Flux, Flux.2, Wan 2.2)
  samplers/                denoising solvers (euler, er_sde)
  upscale/                 latent + pixel upscalers
  postprocess/             grain, sharpening, Qwen-VAE enhance
  ui/                      web UI served by `serve`
  api.py / cli.py          HTTP API and CLI entry points
scripts/                   download.py — the unified model download helper
build-scripts/             portable bundle build, archive, GPU qualification
tests/                     test suite (runs without real weights)
```

## Setup from source

### 1. Install `uv`

[`uv`](https://github.com/astral-sh/uv) is the only prerequisite - it provides
the Python interpreter and installs every dependency:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

### 2. Clone the repo

```bash
git clone https://github.com/lemonade-sdk/thenoise.git
cd thenoise
```

### 3. Bootstrap the environment

`thenoise.sh` creates the `.venv`, installs the ROCm build of PyTorch, and
installs the project in editable mode. Running it with `--help` does all of
that without needing any model weights yet:

```bash
./thenoise.sh --help
```

This is the slow step - it downloads several GB of ROCm PyTorch wheels.
Subsequent runs skip the torch install (detected via `import torch`).

By default the script autodetects the GPU's architecture. Override with the
`GFX_ARCH` environment variable if needed:

```bash
GFX_ARCH=gfx1151 ./thenoise.sh --help
```

### 4. Download a model

Model downloads use `huggingface_hub`, which lives in the `scripts` extra
(`thenoise.sh` only installs runtime deps). Install it once, then use the
venv's Python (a bare system `python` will not work):

```bash
uv pip install -e ".[scripts]"
.venv/bin/python scripts/download.py --model anima
```

All models and options: see the [model pages](models/anima.md) or
`scripts/download.py --help`.

### 5. Generate an image

```bash
./thenoise.sh generate \
  --dit ./models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
  --vae ./models/anima/split_files/vae/qwen_image_vae.safetensors \
  --text-encoder ./models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
  --prompt "a fox walking in the snow" \
  --out fox.png
```

The first generation is slow because the DiT is compiled with `torch.compile` —
see [First run](cli.md#first-run).

### 6. Dev extras

`thenoise.sh` installs the runtime dependencies only. To run the test suite,
install the dev extras:

```bash
uv pip install -e ".[dev]"
```

To use the model download helper, install the `scripts` extra (or in
addition):

```bash
uv pip install -e ".[scripts]"
```

> **Never run `uv sync`.** It would replace/break the ROCm `torch` build that
> `thenoise.sh` installs directly into the venv. `torch` is intentionally *not*
> listed in `pyproject.toml`. Use `uv pip install` for anything else.

## Running the tests

The test suite is designed to run without real model weights:

```bash
.venv/bin/python -m pytest tests/ -q
```

Run this before pushing or opening a PR.

## Building the portable bundles

CI ([`build-thenoise-rocm.yml`](../.github/workflows/build-thenoise-rocm.yml))
publishes the [portable release
bundles](setup.md#get-thenoise) per GPU target. Locally:

```bash
bash build-scripts/build_portable.sh gfx1151          # assemble the bundle
bash build-scripts/create_portable_archive.sh         # package it as release tar.gz(es)
bash build-scripts/qualify_thenoise.sh --root <bundle> --model-dir <dir>   # GPU smoke test
```

`build_portable.sh` produces a relocatable directory (standalone CPython, ROCm
PyTorch, all deps, bundled `clang`, `bin/thenoise` launcher, plus
`scripts/download.py` + `huggingface-hub` for model downloads). `create_portable_archive.sh`
splits it into ≤1.9 GB parts when needed (GitHub's asset limit).
`qualify_thenoise.sh` runs on a real GPU: import checks, model download, one
end-to-end generation including the first-run `torch.compile`.

## Contributing

PRs are welcome - especially new model support. A model addition typically
touches:

1. **`thenoise/models/<name>.py`** — the adapter: `detect()` (checkpoint
   fingerprint), `DEFAULT_PREFS`, `CAPABILITIES`, sampling loop
2. **`thenoise/dit/<name>/`** — model weights layout, loading, tokenizer configs
3. **`scripts/download.py`** — a `ModelSpec` entry + artifact registry for the
   Hugging Face files
4. **`docs/models/<name>.md`** — presentation, specs, examples, download and
   usage instructions
5. **`tests/`** — detection/loading tests that run without real weights
6. **`pyproject.toml`** — the new `thenoise.dit.<name>` package (if any)

General rules:

- Keep the one-model-at-a-time, explicit-flags design - no hidden config, no
  workflow layer.
- Anything a request/checkpoint/model can set goes through the
  `request → checkpoint marker → model default` preference chain
  (`DiffusionModel.pref`).
- Don't add `torch` to `pyproject.toml`.
- Verify with `.venv/bin/python -m pytest tests/ -q` before opening the PR.
