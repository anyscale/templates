#!/usr/bin/env bash
set -euo pipefail

# Repo root = nearest ancestor dir with BUILD.yaml, so this works from any cwd.
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
while [[ "$REPO_ROOT" != "/" && ! -f "$REPO_ROOT/BUILD.yaml" ]]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/BUILD.yaml" ]] || { echo "repo root not found: no BUILD.yaml above $0" >&2; exit 1; }

RAYDEPSETS_BASE_URL="https://github.com/ray-project/raydepsets/releases/download/v0.0.1"
RAYDEPSETS_BIN="/tmp/raydepsets"

# Select platform-specific binary
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64)  RAYDEPSETS_URL="${RAYDEPSETS_BASE_URL}/raydepsets-linux-x86_64" ;;
  Darwin-arm64)  RAYDEPSETS_URL="${RAYDEPSETS_BASE_URL}/raydepsets-darwin-arm64" ;;
  *)             echo "Unsupported platform: $(uname -s)-$(uname -m)" >&2; exit 1 ;;
esac

# Fetch raydepsets binary if not already present
if [[ ! -x "$RAYDEPSETS_BIN" ]]; then
  echo "Fetching raydepsets binary..."
  curl -fsSL "$RAYDEPSETS_URL" -o "$RAYDEPSETS_BIN"
  chmod +x "$RAYDEPSETS_BIN"
fi

echo "Running raydepsets..."
"$RAYDEPSETS_BIN" build \
  "$REPO_ROOT/dependencies/template.depsets.yaml" \
  --workspace-dir "$REPO_ROOT" \
  "$@"
