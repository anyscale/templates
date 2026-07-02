#!/usr/bin/env bash
set -euo pipefail

RAY_VERSION="${1:?Usage: fetch-ray-llm-depsets.sh <ray-version> <python-short> <cuda-variant>}"
PYTHON_SHORT="${2:?Usage: fetch-ray-llm-depsets.sh <ray-version> <python-short> <cuda-variant>}"
CUDA_VARIANT="${3:?Usage: fetch-ray-llm-depsets.sh <ray-version> <python-short> <cuda-variant>}"

DEST_DIR="/tmp/ray-deps"
# Remote filename is version-agnostic (it lives under the ray-${RAY_VERSION} tag);
# the local copy is namespaced by Ray version so concurrent multi-version builds
# don't clobber each other's LLM lock in the shared /tmp/ray-deps dir.
REMOTE_FILE="rayllm_py${PYTHON_SHORT}_${CUDA_VARIANT}.lock"
LOCK_FILE="rayllm_${RAY_VERSION}_py${PYTHON_SHORT}_${CUDA_VARIANT}.lock"
URL="https://raw.githubusercontent.com/ray-project/ray/ray-${RAY_VERSION}/python/deplocks/llm/${REMOTE_FILE}"

mkdir -p "$DEST_DIR"

echo "Fetching Ray LLM lock: $URL"
curl -fsSL "$URL" -o "$DEST_DIR/$LOCK_FILE"
echo "Saved to $DEST_DIR/$LOCK_FILE"
