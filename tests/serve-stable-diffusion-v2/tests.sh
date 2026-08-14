#!/usr/bin/env bash
set -euxo pipefail

# Notebook only documents the local `serve run` as a terminal step, so drive
# main.py + query.py directly. Skips the prod `anyscale service deploy` path.
uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match

serve run main:stable_diffusion_app --non-blocking
trap 'serve shutdown -y || true' EXIT

# First request blocks through GPU provisioning, the replicas' runtime_env build
# (the lock carries torch + the CUDA stack) and the SDXL load, so retry for ~15 min.
# Well inside the 2700s test budget in BUILD.yaml.
ok=false
for i in $(seq 1 30); do
  python query.py || true
  if python -c "
import os
from PIL import Image
assert os.path.getsize('image.png') > 1000
Image.open('image.png').verify()
" 2>/dev/null; then
    ok=true
    echo "query.py produced a valid PNG on attempt ${i}."
    break
  fi
  echo "Attempt ${i}: endpoint not ready yet; retrying in 30s."
  sleep 30
done

[ "$ok" = "true" ] || { echo "serve-stable-diffusion did not return a valid image in time." >&2; exit 1; }
