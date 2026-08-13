#!/usr/bin/env bash
# Fetch a published Anyscale base image's package list into a freeze file.
#
# docs.anyscale.com serves {image_name, pip[], conda, debian} per image variant;
# index.json maps image name -> filename.
#
# Usage: fetch-image-freeze.sh <image-name> <dest-file>
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
  echo "# Regenerate with dependencies/scripts/fetch-image-freeze.sh"
  curl -fsSL "$BASE/$FILENAME" | python3 -c '
import json, sys
print("\n".join(json.load(sys.stdin)["pip"]))'
} > "$DEST"

echo "Fetched $(grep -c '==' "$DEST") packages for $IMAGE -> $DEST"
