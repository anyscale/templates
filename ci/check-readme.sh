#!/bin/bash
# Pre-commit hook: each template's README.md must match what nbconvert
# produces from its sibling README.ipynb. Verifier; never mutates.

set -euo pipefail

if ! command -v jupyter-nbconvert >/dev/null 2>&1; then
  echo "ERROR: jupyter-nbconvert not found. Install with: pip install nbconvert" >&2
  exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Pre-commit passes either README.ipynb or README.md; normalize to the .ipynb.
# Leading `(` in patterns avoids parser confusion with $() closing.
NOTEBOOKS=$(
  for f in "$@"; do
    case "$f" in
      (*/README.ipynb) printf '%s\n' "$f" ;;
      (*/README.md)    printf '%s\n' "${f%.md}.ipynb" ;;
    esac
  done | sort -u
)
[ -z "$NOTEBOOKS" ] && exit 0

EXIT=0
while IFS= read -r NOTEBOOK; do
  [ -z "$NOTEBOOK" ] || [ ! -f "$NOTEBOOK" ] && continue
  DIR=$(dirname "$NOTEBOOK")
  EXPECTED="$DIR/README.md"
  REGEN_CMD="jupyter-nbconvert --to markdown $NOTEBOOK --output-dir $DIR"

  if [ ! -f "$EXPECTED" ]; then
    echo "::error::$NOTEBOOK has no sibling README.md. Generate: $REGEN_CMD" >&2
    EXIT=1; continue
  fi

  jupyter-nbconvert --to markdown "$NOTEBOOK" --output README --output-dir "$TMP" >/dev/null 2>&1
  if ! diff -q "$EXPECTED" "$TMP/README.md" >/dev/null 2>&1; then
    echo "::error::$EXPECTED out of sync with $NOTEBOOK. Regenerate: $REGEN_CMD" >&2
    echo "Diff (committed vs regenerated, first 50 lines):" >&2
    diff "$EXPECTED" "$TMP/README.md" 2>&1 | head -50 >&2 || true
    EXIT=1
  fi
  rm -f "$TMP/README.md"
done <<< "$NOTEBOOKS"

exit $EXIT
