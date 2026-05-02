#!/usr/bin/env bash
set -euo pipefail

RAY_VERSION="${1:?Usage: fetch-ray-constraints.sh <ray-version> <python-version>}"
PYTHON_VERSION="${2:?Usage: fetch-ray-constraints.sh <ray-version> <python-version>}"

DEST_DIR="/tmp/ray-deps"
DEST_FILE="requirements_compiled_py${PYTHON_VERSION}.txt"
BASE_URL="https://raw.githubusercontent.com/ray-project/ray/ray-${RAY_VERSION}/python"

mkdir -p "$DEST_DIR"

if [[ -f "$DEST_DIR/$DEST_FILE" ]]; then
  echo "Already fetched: $DEST_DIR/$DEST_FILE"
  exit 0
fi

# Try version-specific constraints first, fall back to generic
URL="${BASE_URL}/requirements_compiled_py${PYTHON_VERSION}.txt"
echo "Fetching Ray constraints: $URL"
if ! curl -fsSL "$URL" -o "$DEST_DIR/$DEST_FILE" 2>/dev/null; then
  FALLBACK_URL="${BASE_URL}/requirements_compiled.txt"
  echo "Version-specific constraints not found, falling back to: $FALLBACK_URL"
  curl -fsSL "$FALLBACK_URL" -o "$DEST_DIR/$DEST_FILE"
fi

echo "Saved to $DEST_DIR/$DEST_FILE"
