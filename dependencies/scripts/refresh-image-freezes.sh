#!/usr/bin/env bash
# Refresh the committed image freezes for a Ray version.
#
# An image that isn't published yet is skipped rather than failing the run: base
# locks land before every variant exists, and the next scheduled run picks it up.
#
# Usage: refresh-image-freezes.sh <ray-version>
set -euo pipefail

VERSION="${1:?Usage: refresh-image-freezes.sh <ray-version>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LIST="$ROOT/dependencies/images/tracked-images.txt"

[ -f "$LIST" ] || { echo "Missing $LIST" >&2; exit 1; }

missing=0
while IFS= read -r line; do
  case "$line" in ''|'#'*) continue ;; esac
  image="${line//\{version\}/$VERSION}"
  dest="$ROOT/dependencies/images/$(echo "${image#anyscale/}" | tr ':' '-').freeze.txt"
  if ! "$ROOT/dependencies/scripts/fetch-image-freeze.sh" "$image" "$dest"; then
    echo "::warning::no published freeze for $image yet — skipping"
    missing=$((missing + 1))
  fi
done < "$LIST"

echo "Refreshed image freezes for Ray $VERSION ($missing skipped)"
