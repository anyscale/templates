#!/usr/bin/env bash
# Seed a template's lock with the base image's package list before compiling.
#
# uv treats an existing -o file as resolution *preferences*: a package listed there
# keeps its version unless a requirement forces otherwise. Seeding with a freeze of
# the image therefore reproduces what a user gets from `pip install` in a workspace
# on that image — image versions stay put, and only what the template genuinely
# needs moves. That is why these templates don't need image versions hand-copied
# into requirements.txt.
#
# Seeding from the image (rather than from the previously committed lock) is what
# makes it deterministic: the result is a function of freeze + requirements.txt,
# not of whatever happened to be committed last.
#
# Usage: seed-image-freeze.sh <freeze-file> <lock-output-path>
set -euo pipefail

FREEZE="${1:?Usage: seed-image-freeze.sh <freeze-file> <lock-output-path>}"
OUTPUT="${2:?Usage: seed-image-freeze.sh <freeze-file> <lock-output-path>}"

[ -f "$FREEZE" ] || { echo "Image freeze not found: $FREEZE" >&2; exit 1; }

mkdir -p "$(dirname "$OUTPUT")"
cp "$FREEZE" "$OUTPUT"
echo "Seeded $OUTPUT with $(grep -c '==' "$FREEZE") packages from $(basename "$FREEZE")"
