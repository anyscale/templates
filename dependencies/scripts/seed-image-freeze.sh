#!/usr/bin/env bash
# Seed a lock with the base image's package list before compiling.
#
# uv reads the -o file as resolution preferences, so a package listed there keeps
# its version unless a requirement forces otherwise — the same result as
# `pip install` in a workspace on that image. Seeding from the freeze rather than
# the committed lock makes it a function of freeze + requirements.txt only.
#
# Usage: seed-image-freeze.sh <freeze-file> <lock-output-path>
set -euo pipefail

FREEZE="${1:?Usage: seed-image-freeze.sh <freeze-file> <lock-output-path>}"
OUTPUT="${2:?Usage: seed-image-freeze.sh <freeze-file> <lock-output-path>}"

[ -f "$FREEZE" ] || { echo "Image freeze not found: $FREEZE" >&2; exit 1; }

mkdir -p "$(dirname "$OUTPUT")"
cp "$FREEZE" "$OUTPUT"
echo "Seeded $OUTPUT with $(grep -c '==' "$FREEZE") packages from $(basename "$FREEZE")"
