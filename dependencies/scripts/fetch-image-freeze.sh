#!/usr/bin/env bash
# Fetch the package list of a published Anyscale base image.
#
# docs.anyscale.com serves one JSON per image variant ({image_name, pip[], conda,
# debian}); index.json maps image name -> filename. The `pip` array is a freeze of
# the real image, which is the only accurate description of it: ray-llm images are
# assembled with `pip install --no-deps`, so their package set is not something a
# resolver can reproduce (the image ships opencv 4.13 with numpy 1.26.4 and vllm
# 0.22 with protobuf 4.25.8 — both violate the packages' declared requirements).
#
# Usage: fetch-image-freeze.sh <image-name> <dest-file>
#   e.g. fetch-image-freeze.sh anyscale/ray-llm:2.56.0-py312-cu130 \
#          dependencies/images/ray-llm-2.56.0-py312-cu130.freeze.txt
set -euo pipefail

IMAGE="${1:?Usage: fetch-image-freeze.sh <image-name> <dest-file>}"
DEST="${2:?Usage: fetch-image-freeze.sh <image-name> <dest-file>}"
BASE="https://docs.anyscale.com/base-images"

FILENAME=$(curl -fsSL "$BASE/index.json" | IMAGE="$IMAGE" python3 -c '
import json, os, sys
image = os.environ["IMAGE"]
print(next((e["filename"] for e in json.load(sys.stdin) if e.get("imageName") == image), ""))')

if [ -z "$FILENAME" ]; then
  echo "Image not found in $BASE/index.json: $IMAGE" >&2
  exit 1
fi

mkdir -p "$(dirname "$DEST")"
{
  echo "# pip freeze of $IMAGE"
  echo "# Fetched from $BASE/$FILENAME — regenerate with dependencies/scripts/fetch-image-freeze.sh"
  curl -fsSL "$BASE/$FILENAME" | python3 -c '
import json, sys
print("\n".join(json.load(sys.stdin)["pip"]))'
} > "$DEST"

echo "Fetched $(grep -c '==' "$DEST") packages for $IMAGE -> $DEST"
