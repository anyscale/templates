#!/usr/bin/env bash
set -euo pipefail

RAY_VERSION="${1:?Usage: fetch-ray-depsets.sh <ray-version> <python-short>}"
PYTHON_SHORT="${2:?Usage: fetch-ray-depsets.sh <ray-version> <python-short>}"

DEST_DIR="/tmp/ray-deps"
LOCK_FILE="ray_img_py${PYTHON_SHORT}.lock"
URL="https://raw.githubusercontent.com/ray-project/ray/ray-${RAY_VERSION}/python/deplocks/ray_img/${LOCK_FILE}"

mkdir -p "$DEST_DIR"

# fallback to use requirements.txt for ray versions that don't have ray_img_py${PYTHON_SHORT}.lock
FALLBACK_URL="https://raw.githubusercontent.com/ray-project/ray/releases/${RAY_VERSION}/python/requirements.txt"

echo "Fetching Ray lock: $URL"
if ! curl -fsSL "$URL" -o "$DEST_DIR/$LOCK_FILE" 2>/dev/null; then
  echo "Not found, falling back to: $FALLBACK_URL"
  curl -fsSL "$FALLBACK_URL" -o "$DEST_DIR/$LOCK_FILE"
fi
echo "Saved to $DEST_DIR/$LOCK_FILE"
