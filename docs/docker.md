# Docker / Compose

Running TheNoise in a container on ROCm: the same engine and the same flags,
with the lifecycle (start, stop, restart on boot, hand the GPU to another
service) managed by Docker or Compose instead of a shell.

The image is the [portable bundle](setup.md) on a base layer: the build stage
runs `build-scripts/build_portable.sh`, the script that produces the release
bundles, and the runtime stage copies the result onto a bare Ubuntu. So the
interpreter, the ROCm build of PyTorch, the `clang` Triton JIT-compiles with and
the `bin/thenoise` launcher are the ones CI builds and GPU-qualifies — there is
no system ROCm and no system Python in the image, and the host needs neither.

**What this does not buy you:** performance. The container adds no measurable
overhead, but it does add a 10 GB image, a device-passed GPU, and one more
thing between you and an error message. If you only ever run TheNoise by hand on
the machine it runs on, use the portable bundle.

## Requirements

- Linux x86_64, Docker with BuildKit (default since 23.0) or Docker Compose v2.
- The **amdgpu kernel driver on the host**, exposing `/dev/kfd` and `/dev/dri`.
  No ROCm user-space install is needed on the host — it is in the image.
- A GPU whose target matches the image's `GFX_ARCH` (`gfx1151` for Strix Halo).

## Get the image

Released versions are published per GPU target:

```bash
docker pull ghcr.io/lemonade-sdk/thenoise:gfx1151        # latest release
docker pull ghcr.io/lemonade-sdk/thenoise:0.12.0-gfx1151 # a specific version
```

Only a release moves `:gfx1151`. You may also see `:edge-gfx1151` in the registry
— that is the last image someone built by hand to check something, and it is not
a release.

Building from a checkout pulls several GB of wheels, so expect ten to thirty
minutes and ~25 GB of free disk:

```bash
docker build -f docker/Dockerfile --build-arg GFX_ARCH=gfx1151 -t thenoise:local .
```

`GFX_ARCH` is an exact target — `gfx1151`, `gfx1150`, `gfx1152`, `gfx1030`, …
It selects which GPU's code objects the ROCm wheels install, so an image built
for one target does not run on another.

## Compose

[`docker/docker-compose.yml`](docker-compose.yml) is the example to copy. It
defines one service per model (TheNoise loads one at a time) and a `download`
service for filling the model directory:

```bash
# build once, or delete the build: block and pull from ghcr.io instead
docker compose -f docker/docker-compose.yml build

# download a model into the mounted models directory (--profile because the
# download service is deliberately not part of `up`)
docker compose -f docker/docker-compose.yml --profile tools run --rm download \
  --model anima --out /models/anima

# serve it: web UI at http://localhost:8000/
docker compose -f docker/docker-compose.yml up -d anima

docker compose -f docker/docker-compose.yml logs -f anima
docker compose -f docker/docker-compose.yml down
```

By default the models come from the repo's own `models/` directory, so a
`scripts/download.py` run on the host and the container share one copy of
everything. `THENOISE_MODELS_DIR` and `THENOISE_OUTPUT_DIR` point them apart.

## docker run

The Compose file is a convenience; everything is one `docker run`. Model paths
are the container paths of the checkpoints you mounted:

```bash
docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add render --group-add video \
  --security-opt seccomp=unconfined --ipc=host \
  -v "$PWD/models:/models" -v thenoise-cache:/cache -v "$PWD/output:/output" \
  -p 8000:8000 \
  thenoise:local serve \
    --dit /models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
    --vae /models/anima/split_files/vae/qwen_image_vae.safetensors \
    --text-encoder /models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors
```

A one-shot generation instead of a server, writing `fox.png` into the mounted
output directory:

```bash
docker run --rm \
  --device /dev/kfd --device /dev/dri \
  --group-add render --group-add video \
  --security-opt seccomp=unconfined --ipc=host \
  -v "$PWD/models:/models" -v thenoise-cache:/cache -v "$PWD/output:/output" \
  thenoise:local generate \
    --dit /models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors \
    --vae /models/anima/split_files/vae/qwen_image_vae.safetensors \
    --text-encoder /models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors \
    --prompt "a fox walking in the snow" --out /output/fox.png
```

The entrypoint is the launcher, so the whole
[CLI](cli.md) is available this way. `docker run thenoise:local --help` works,
and with no arguments at all the container starts `serve`.

## Volumes

| Path | What goes there |
|---|---|
| `/models` | Checkpoints — the layout `scripts/download.py` produces. Read-write so downloads can land here; mount `:ro` once you stop downloading. |
| `/output` | Working directory, so a `--out fox.png` with no directory ends up somewhere you mounted. |
| `/cache` | Compiled-kernel caches. Without a volume here every new container re-runs `torch.compile`, which is minutes per model. |

The image sets the cache locations so they all land in `/cache`, and you can
override any of them:

| Variable | Default | Caches |
|---|---|---|
| `MIOPEN_USER_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR` | `/cache/miopen` | MIOpen kernel tuning + kernels |
| `TRITON_CACHE_DIR` | `/cache/triton` | Triton kernels |
| `TORCHINDUCTOR_CACHE_DIR` | `/cache/inductor` | `torch.compile` graphs |
| `TORCH_EXTENSIONS_DIR` | `/cache/torch-extensions` | JIT-compiled C++ extensions |
| `HF_HOME` | `/cache/huggingface` | anything fetched from the Hub at runtime |

## Startup, shutdown, switching models

- **Readiness.** `serve` loads the model *before* it binds the port, so the
  moment anything answers on it the model is ready. `/health` answers `200` with
  the loaded model's name and capabilities, which is what the healthcheck and
  tools like llama-swap poll. Loading a 35 GB model takes minutes; the Compose
  example gives it a generous `start_period`.
- **The bind address.** `serve` defaults to `127.0.0.1`, which inside a
  container is reachable from nothing. The container's entrypoint appends
  `--host 0.0.0.0` to any `serve` that does not pass `--host`, since a
  container's network namespace is its own. TheNoise has no authentication:
  publish port 8000 only where you would publish an unauthenticated service, and
  use `--host 127.0.0.1` to keep it inside the container.
- **Shutdown.** `docker stop` sends `SIGTERM`; uvicorn stops accepting and lets
  the request in flight finish. After `stop_grace_period` Docker sends
  `SIGKILL`, which reclaims the GPU immediately. A generation takes tens of
  seconds, so a swap that must happen now is a cancelled request either way —
  set `stop_grace_period` to what you are willing to wait.
- **Swapping.** Only one container can use the GPU well at a time, so swap
  rather than share: `docker compose stop anima && docker compose up -d krea2`.
  With llama-swap, give it the `docker run` as the model's `cmd` and let it own
  the lifecycle:

  ```yaml
  # llama-swap config.yaml — healthCheckTimeout must outlast a model load
  healthCheckTimeout: 900

  models:
    thenoise-anima:
      cmd: |
        docker run --rm -i
        --network host
        --device /dev/kfd --device /dev/dri
        --group-add render --group-add video
        --security-opt seccomp=unconfined --ipc=host
        -v /srv/models:/models -v thenoise-cache:/cache
        ghcr.io/lemonade-sdk/thenoise:gfx1151
        serve --port ${PORT}
        --dit /models/anima/split_files/diffusion_models/anima-turbo-v1.0.safetensors
        --vae /models/anima/split_files/vae/qwen_image_vae.safetensors
        --text-encoder /models/anima/split_files/text_encoders/qwen_3_06b_base.safetensors
      proxy: http://127.0.0.1:${PORT}
      checkEndpoint: /health
      ttl: 300
      # llama-swap force-kills after this: keep it above the container's own
      # stop grace, otherwise the switch is a SIGKILL and not a good-bye.
      unloadTimeout: 40
  ```

## Debugging inside the container

The libraries the bundle ships are registered with the loader, so a bare
`--entrypoint` run works without reproducing any environment:

```bash
# does the container see the GPU?
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add render --group-add video --security-opt seccomp=unconfined \
  --entrypoint python3 thenoise:local \
  -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'

# a shell
docker run --rm -it --entrypoint bash thenoise:local
# and inside it: thenoise --help, python3, /opt/thenoise/scripts/download.py --help
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Permission denied` on `/dev/kfd` | The device nodes are group-owned — `render` on most distros, `video` on others — so pass both `--group-add render --group-add video`. Especially when running with `user:`, where DAC applies. |
| `no HIP devices found` while the host is fine | Wrong `GFX_ARCH` for this GPU, or the amdgpu driver is not loaded on the host. |
| `libgomp.so.1: cannot open shared object file` | A base-image change dropped the library; the runtime stage installs `libgomp1` and `libatomic1` on purpose. [Open an issue](https://github.com/lemonade-sdk/thenoise/issues). |
| Container reports `unhealthy` but generates images | The healthcheck assumes port 8000 and starts counting during the model load. Pass the URL to it (`test: ["CMD", "healthcheck.sh", "http://127.0.0.1:8123/health"]`) and raise `start_period`. |
| Every start takes minutes before the first image | `/cache` is not a volume (or is empty): see [volumes](#volumes). |
| `docker compose run … download` writes nothing where you look | `--out` is required: the download helper's default is a path relative to the working directory, which here is `/output`. And the service has a profile, so `run` needs `--profile tools`. |
| `no space left on device` during the build | The bundle is ~10 GB and the image that contains it another ~10 GB; `docker builder prune` frees the previous attempts. |
| Files the container wrote are root-owned on the host | The container runs as root. Set `user: "1000:1000"` on the service (keeping `group_add`) and make `/models`, `/output` and the cache volume writable by that uid — `/cache` is written during every first run. |

**GPU qualification:** CI builds this image without a GPU and checks that the
launcher comes up and that `import torch` works. A real generation inside the
container is the part only you can run — the `docker run` above with `generate`
and a small model is enough, and a first run that compiles and produces a PNG
means the image is good on your hardware.
