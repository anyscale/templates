#!/usr/bin/env bash
# Cursor Cloud Agent install script — runtime auth + repo-tree wiring.
#
# Split between this script and .cursor/Dockerfile:
#   - Dockerfile (image build, cached): all deterministic, version-pinned,
#     secret-free tooling — gh, docker, gcloud, anyscale CLI, pre-commit,
#     buildkite-mcp-server, ANYSCALE_HOST, etc.
#   - install.sh (every VM start): per-run auth using Cursor secrets, plus
#     setup that depends on the mounted repo (rayapp pinned by
#     download_rayapp.sh, pre-commit hook, sideloaded skills).
#
# Required secrets (Cursor → My Secrets):
#   ANYSCALE_GH_TOKEN              gh write fallback on anyscale/templates
#   ANYSCALE_CLI_TOKEN             anyscale CLI auth + skills install
#   GCP_TEMPLATE_REGISTRY_SA_KEY   docker push to us-docker.pkg.dev
#   BUILDKITE_API_TOKEN            Buildkite MCP server (Dockerfile-baked)
set -euo pipefail

# --- pre-commit hooks (auto-fire on git commit; idempotent) ---
# Non-fatal: pre-commit refuses if core.hooksPath is set (Cursor sets it).
# The agent can still run `pre-commit run --all-files` by hand.
if [ -f .pre-commit-config.yaml ] && [ -d .git ]; then
  pre-commit install \
    || echo "WARN: pre-commit install skipped — run 'pre-commit run --all-files' manually before committing."
fi

# --- rayapp (version pinned via repo's download_rayapp.sh; lives in the
# repo so kept here rather than baked into the image) ---
if [ -f download_rayapp.sh ] && ! command -v rayapp &>/dev/null; then
  bash download_rayapp.sh
  mv rayapp /usr/local/bin/rayapp
fi

# --- Auth: gh ---
: "${ANYSCALE_GH_TOKEN:?secret ANYSCALE_GH_TOKEN is empty/unset; add it in Cursor → Cloud Agents → My Secrets}"
echo "$ANYSCALE_GH_TOKEN" | gh auth login --with-token

# --- Auth: gcloud (for docker push to GCP artifact registry; soft — only needed for custom images) ---
if [ -n "${GCP_TEMPLATE_REGISTRY_SA_KEY:-}" ]; then
  echo "$GCP_TEMPLATE_REGISTRY_SA_KEY" > /tmp/gcp-sa.json
  gcloud auth activate-service-account --key-file=/tmp/gcp-sa.json
  gcloud auth configure-docker us-docker.pkg.dev --quiet
else
  echo "WARN: GCP_TEMPLATE_REGISTRY_SA_KEY not set — custom-image rebuild + push to GCP will fail."
fi

# --- Auth: anyscale CLI + skills install (/ask, /fix, /run, /inspect).
# -f overwrites locally-stale skills with the latest published.
# After install, prune any other skill dirs so ~/.claude/skills/ mirrors
# what the CLI laid down — keeps the env hermetic across runs. ---
if [ -n "${ANYSCALE_CLI_TOKEN:-}" ]; then
  mkdir -p ~/.anyscale
  cat > ~/.anyscale/credentials.json <<EOF
{"cli_token": "$ANYSCALE_CLI_TOKEN"}
EOF
  anyscale skills install -p claude-code -y -f
  for d in ~/.claude/skills/*/; do
    [ -d "$d" ] || continue
    case "$(basename "$d")" in
      anyscale-platform-ask|anyscale-platform-fix|anyscale-platform-run|anyscale-platform-inspect) ;;
      *) echo "Removing unrecognized skill: $d"; rm -rf "$d" ;;
    esac
  done
  ls ~/.claude/skills/
else
  echo "WARN: ANYSCALE_CLI_TOKEN not set — preflight will fail."
fi

# --- Cloud-agent MCP config: merge workspace .cursor/mcp.json into user-scope
# ~/.cursor/mcp.json. Workspace-scope is read by Cursor IDE only; cloud agents
# read user-scope at session boot. Merge rather than overwrite so any
# pre-existing MCPs (e.g. ones Cursor itself populates) survive — workspace
# entries win on key collision. ---
if [ -f .cursor/mcp.json ]; then
  mkdir -p ~/.cursor
  python3 - <<'PY'
import json, pathlib

home = pathlib.Path.home() / ".cursor" / "mcp.json"
ws = pathlib.Path(".cursor") / "mcp.json"

merged = {}
if home.exists():
    try:
        merged = json.loads(home.read_text())
    except json.JSONDecodeError:
        merged = {}

ws_cfg = json.loads(ws.read_text())
merged.setdefault("mcpServers", {})
merged["mcpServers"].update(ws_cfg.get("mcpServers", {}))

home.write_text(json.dumps(merged, indent=2) + "\n")
print(f"Merged MCP servers from {ws} into {home}")
PY
fi

# --- Preflight: validate the env this script just set up. Same script the
# agent re-runs at task start (see AGENTS.md → Cursor Cloud → Preconditions). ---
bash .cursor/preflight.sh
