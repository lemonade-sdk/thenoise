# This is a lightweight editable development environment, not the production
# image in docker/Dockerfile (which builds a multi-GB portable release bundle).
FROM mcr.microsoft.com/devcontainers/base:ubuntu24.04

# Keep uv versioned; do not use uv sync (see AGENTS.md).
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /usr/local/bin/

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git pkg-config \
      libatomic1 libgomp1 libgl1 libglib2.0-0t64 \
 && rm -rf /var/lib/apt/lists/*

# Both configs build with .devcontainer/ as their context.
COPY gpu-entrypoint.sh /usr/local/bin/thenoise-devcontainer-entrypoint
RUN chmod 0755 /usr/local/bin/thenoise-devcontainer-entrypoint \
 && mkdir -p /workspaces/thenoise/.venv /home/vscode/.cache \
 && chown -R vscode:vscode /workspaces/thenoise /home/vscode/.cache

WORKDIR /workspaces/thenoise
# The entrypoint needs root to map GPU GIDs; VS Code runs as vscode.
USER root
ENTRYPOINT ["/usr/local/bin/thenoise-devcontainer-entrypoint"]
CMD ["sleep", "infinity"]
