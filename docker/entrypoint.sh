#!/bin/bash
# Container entrypoint: turn the container's command line into launcher
# arguments. All environment setup belongs to the bundle's bin/thenoise
# (LD_LIBRARY_PATH, the bundled clang, the MIOpen/AOTriton knobs) — this script
# deliberately sets none of it.
set -euo pipefail

LAUNCHER=/opt/thenoise/bin/thenoise

# `docker run thenoise` with no arguments means "start the server".
if [ $# -eq 0 ]; then
  set -- serve
fi

# serve binds 127.0.0.1, which nothing outside this network namespace can
# reach, so a container listens on every interface unless told otherwise.
# Pass `--host 127.0.0.1` to keep it loopback-only inside the container.
if [ "$1" = serve ]; then
  host_given=no
  for arg in "$@"; do
    case "$arg" in
      --host | --host=*) host_given=yes ;;
    esac
  done
  if [ "$host_given" = no ]; then
    set -- "$@" --host 0.0.0.0
  fi
fi

exec "$LAUNCHER" "$@"
