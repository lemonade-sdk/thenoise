#!/usr/bin/env bash
set -euo pipefail

# Device node GIDs are host-dependent. Map them inside the container BEFORE
# VS Code starts its non-root "vscode" session, rather than assuming the
# container's "render" or "video" GID matches the host.
if [ "$(id -u)" -eq 0 ]; then
  for node in /dev/kfd /dev/dri/renderD* /dev/dri/card*; do
    [ -c "$node" ] || continue
    gid="$(stat -c '%g' "$node")"
    if [ "$gid" -eq 0 ]; then
      echo "Warning: $node is root-group-owned; configure host device permissions for non-root GPU access." >&2
      continue
    fi
    if ! getent group "$gid" >/dev/null; then
      groupadd -g "$gid" "thenoise-gpu-$gid"
    fi
    group="$(getent group "$gid" | cut -d: -f1)"
    usermod -aG "$group" vscode
  done

  # These are named Docker volumes, not directories from the host checkout.
  # Only change mount roots to avoid re-chowning a large venv/cache on restart.
  chown vscode:vscode /workspaces/thenoise/.venv /home/vscode/.cache
fi

exec "$@"
