#!/bin/bash
# Readiness probe. `serve` loads its model before it binds, so a listening
# socket already means "ready to generate" — this only turns a 200 from /health
# into an exit code. Pass the URL as $1 when serving on another port.
set -euo pipefail

URL="${1:-http://127.0.0.1:8000/health}"

/opt/thenoise/bin/python3 - "$URL" <<'PY'
import json, sys, urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
        ok = json.load(response).get("status") == "ok"
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY
