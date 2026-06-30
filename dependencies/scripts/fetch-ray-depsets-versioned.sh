#!/usr/bin/env bash
# Like fetch-ray-depsets.sh, but writes the fetched lock to a version-stamped
# path (/tmp/ray-deps/ray_<ray-version>_img_py<python-short>.lock) so it does
# not collide with concurrent fetches of a different Ray version's lock for
# the same Python (raydepsets v0.0.1 runs all pre_hooks before any compile,
# so two entries that share the unversioned /tmp/ray-deps/ray_img_py<X>.lock
# path stomp on each other when one of them targets a different Ray version).
set -euo pipefail

RAY_VERSION="${1:?Usage: fetch-ray-depsets-versioned.sh <ray-version> <python-short>}"
PYTHON_SHORT="${2:?Usage: fetch-ray-depsets-versioned.sh <ray-version> <python-short>}"

DEST_DIR="/tmp/ray-deps"
LOCK_FILE="ray_${RAY_VERSION}_img_py${PYTHON_SHORT}.lock"
URL="https://raw.githubusercontent.com/ray-project/ray/ray-${RAY_VERSION}/python/deplocks/ray_img/ray_img_py${PYTHON_SHORT}.lock"

mkdir -p "$DEST_DIR"

FALLBACK_URL="https://raw.githubusercontent.com/ray-project/ray/releases/${RAY_VERSION}/python/requirements.txt"

echo "Fetching Ray lock: $URL"
if ! curl -fsSL "$URL" -o "$DEST_DIR/$LOCK_FILE" 2>/dev/null; then
  echo "Not found, falling back to: $FALLBACK_URL"
  curl -fsSL "$FALLBACK_URL" -o "$DEST_DIR/$LOCK_FILE"
fi
echo "Saved to $DEST_DIR/$LOCK_FILE"
