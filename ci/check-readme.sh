#!/bin/bash
# Pre-commit hook: each template's README.md must match what nbconvert
# produces from its sibling README.ipynb. The notebook is the source of
# truth; the .md is generated. This hook FAILS if they drift — it does
# not modify the working tree.
#
# Pre-commit passes the changed files matching the configured `files`
# regex as arguments. We accept either README.ipynb or README.md and
# normalize both to the sibling README.ipynb path.

set -euo pipefail

if ! command -v jupyter-nbconvert >/dev/null 2>&1; then
  cat >&2 <<EOF
ERROR: \`jupyter-nbconvert\` not found. Install nbconvert with one of:

    pip install nbconvert        # plain pip (in an active venv)
    uv tool install nbconvert    # uv (puts jupyter-nbconvert on PATH globally)

EOF
  exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# Collect unique README.ipynb paths from the changed files
NOTEBOOKS=$(
  for f in "$@"; do
    if [[ "$f" == */README.ipynb ]]; then
      printf '%s\n' "$f"
    elif [[ "$f" == */README.md ]]; then
      printf '%s\n' "${f%.md}.ipynb"
    fi
  done | sort -u
)

if [ -z "$NOTEBOOKS" ]; then
  exit 0
fi

EXIT=0
while IFS= read -r NOTEBOOK; do
  [ -z "$NOTEBOOK" ] && continue
  if [ ! -f "$NOTEBOOK" ]; then
    continue  # notebook was deleted; nothing to check against
  fi

  DIR=$(dirname "$NOTEBOOK")
  EXPECTED="$DIR/README.md"

  if [ ! -f "$EXPECTED" ]; then
    cat >&2 <<EOF

$NOTEBOOK has no sibling README.md.
Every README.ipynb must have a generated README.md alongside it. Generate it with:

    jupyter-nbconvert --to markdown $NOTEBOOK --output-dir $DIR

then stage README.md and commit again.

EOF
    EXIT=1
    continue
  fi

  jupyter-nbconvert --to markdown "$NOTEBOOK" --output README --output-dir "$TMP" >/dev/null 2>&1

  if ! diff -q "$EXPECTED" "$TMP/README.md" >/dev/null 2>&1; then
    cat >&2 <<EOF

$EXPECTED is out of sync with $NOTEBOOK.

The .ipynb is the source of truth; README.md must be regenerated from it.
Install nbconvert if needed (\`pip install nbconvert\` or \`uv tool install nbconvert\`),
then run:

    jupyter-nbconvert --to markdown $NOTEBOOK --output-dir $DIR

and re-stage README.md and commit again.

Diff (committed vs regenerated, first 50 lines):
EOF
    diff "$EXPECTED" "$TMP/README.md" 2>&1 | head -50 >&2 || true
    echo >&2
    EXIT=1
  fi

  rm -f "$TMP/README.md"
done <<< "$NOTEBOOKS"

exit $EXIT
