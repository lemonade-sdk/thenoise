# syntax=docker/dockerfile:1
#
# TheNoise on ROCm, as a container.
#
# The build stage runs build-scripts/build_portable.sh — the same script that
# produces (and whose output CI GPU-qualifies) the release bundles — so the
# image carries the identical self-contained stack: a standalone CPython, the
# ROCm torch wheels for one gfx target, the clang Triton JIT compiles with, the
# unversioned `libfoo.so` symlinks the ROCm wheels omit, and `bin/thenoise`.
# The runtime stage copies that directory onto a bare Ubuntu: no system ROCm,
# no system Python, and no second copy of the environment setup — the launcher
# still owns LD_LIBRARY_PATH, CC and the MIOpen/AOTriton knobs.
#
# Build from the repo root (~10 GB, tens of minutes — the ROCm wheels are big):
#
#   docker build -f docker/Dockerfile --build-arg GFX_ARCH=gfx1151 -t thenoise:local .
#
# Prebuilt images are on GHCR (see .github/workflows/build-docker-image.yml),
# tagged per GPU target: ghcr.io/lemonade-sdk/thenoise:gfx1151.

ARG GFX_ARCH=gfx1151

# ------------------------------------------------------------------ builder --
FROM ubuntu:24.04 AS build
ARG GFX_ARCH

# gcc is needed only here, for Triton's hip_utils precompile; the image itself
# JIT-compiles kernels with the clang the wheels ship.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

COPY . /src
ENV THENOISE_ROOT=/opt/thenoise
RUN bash /src/build-scripts/build_portable.sh "${GFX_ARCH}"

# ------------------------------------------------------------------- runtime --
FROM ubuntu:24.04 AS runtime

# libgomp1: libtorch_cpu.so is linked against libgomp.so.1 and no wheel vendors
# it. libatomic1: the ROCm tracer libraries are linked against it — the same
# missing library portable users hit on Fedora (#29).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates libatomic1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/thenoise /opt/thenoise
ENV PATH=/opt/thenoise/bin:${PATH}

# The launcher exports the bundled library paths itself. Registering them with
# the dynamic loader as well is what makes `docker run --entrypoint python3`
# (the usual way to debug a container) import torch without first reproducing
# the launcher's environment.
RUN for d in /opt/thenoise/lib/python3.*/site-packages/_rocm_sdk_*/lib \
             /opt/thenoise/lib/python3.*/site-packages/_rocm_sdk_*/lib/llvm/lib \
             /opt/thenoise/lib/python3.*/site-packages/torch/lib; do \
        { [ -d "$d" ] && echo "$d"; } || true; \
    done > /etc/ld.so.conf.d/thenoise.conf \
 && ldconfig \
 && python3 -c "import torch, torchvision, fastapi, thenoise; print('torch', torch.__version__)"

# MIOpen, Triton, Inductor and the Hub each cache under $HOME or /tmp, which in
# a container means the writable layer a new container starts without. Pointed
# at one /cache volume, so a rebuilt container keeps its compiled kernels.
ENV MIOPEN_USER_DB_PATH=/cache/miopen \
    MIOPEN_CUSTOM_CACHE_DIR=/cache/miopen \
    TRITON_CACHE_DIR=/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    TORCH_EXTENSIONS_DIR=/cache/torch-extensions \
    HF_HOME=/cache/huggingface

RUN mkdir -p /cache /models /output
WORKDIR /output

COPY --chmod=0755 docker/entrypoint.sh docker/healthcheck.sh /usr/local/bin/
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
