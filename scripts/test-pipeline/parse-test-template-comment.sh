#!/bin/bash
# Parse a /test-template comment body and emit one GH-Actions output to
# $GITHUB_OUTPUT:
#   templates — space-separated template names
#
# Inputs:
#   COMMENT_BODY  — the PR comment body
#   GITHUB_OUTPUT — path to the GH Actions outputs file (set by the runner)
#
# Validation:
#   - non-empty list of names
#   - max 3 templates per comment (cost cap)
#   - each name matches ^[a-zA-Z0-9_-]+$
#   - each name appears in BUILD.yaml's `- name:` entries

set -euo pipefail

: "${COMMENT_BODY:?COMMENT_BODY env var required}"
: "${GITHUB_OUTPUT:?GITHUB_OUTPUT env var required}"

MAX_TEMPLATES=3

TEMPLATES=$(
  echo "$COMMENT_BODY" \
    | tr -d '\r' \
    | awk '/^\/test-template / { for (i=2; i<=NF; i++) printf "%s ", $i; exit }' \
    | xargs
)

if [ -z "$TEMPLATES" ]; then
  echo "::error::No template names provided"
  exit 1
fi

COUNT=$(echo "$TEMPLATES" | wc -w | xargs)
if [ "$COUNT" -gt "$MAX_TEMPLATES" ]; then
  echo "::error::Too many templates ($COUNT); max $MAX_TEMPLATES per /test-template comment"
  exit 1
fi

VALID=$(awk '/^- name:/ {print $3}' BUILD.yaml)
for t in $TEMPLATES; do
  if [[ ! "$t" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "::error::Invalid template name: $t"
    exit 1
  fi
  if ! grep -qxF "$t" <<< "$VALID"; then
    echo "::error::Template '$t' not found in BUILD.yaml"
    exit 1
  fi
done

echo "templates=$TEMPLATES" >> "$GITHUB_OUTPUT"
