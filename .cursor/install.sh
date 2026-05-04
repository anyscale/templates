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
# Required secrets (set in Cursor → My Secrets, exposed as env vars):
#   ANYSCALE_CLI_TOKEN             For the anyscale CLI
#   GCP_TEMPLATE_REGISTRY_SA_KEY   GCP SA JSON for docker push to us-docker.pkg.dev
#   BUILDKITE_API_TOKEN            Read by the Buildkite MCP server (Dockerfile-baked).
#
# GitHub auth: relies on Cursor's native GH App auth (no PAT). Skills are
# pulled from the anyscale-debug-agent repo that Cursor pre-clones via
# .cursor/environment.json's repositoryDependencies field.
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

# --- Auth: gcloud (for docker push to GCP artifact registry; soft — only needed for custom images) ---
if [ -n "${GCP_TEMPLATE_REGISTRY_SA_KEY:-}" ]; then
  echo "$GCP_TEMPLATE_REGISTRY_SA_KEY" > /tmp/gcp-sa.json
  gcloud auth activate-service-account --key-file=/tmp/gcp-sa.json
  gcloud auth configure-docker us-docker.pkg.dev --quiet
else
  echo "WARN: GCP_TEMPLATE_REGISTRY_SA_KEY not set — custom-image rebuild + push to GCP will fail."
fi

# --- Auth: anyscale CLI (soft — only needed for templates that invoke the anyscale CLI) ---
if [ -n "${ANYSCALE_CLI_TOKEN:-}" ]; then
  mkdir -p ~/.anyscale
  cat > ~/.anyscale/credentials.json <<EOF
{"cli_token": "$ANYSCALE_CLI_TOKEN"}
EOF
else
  echo "WARN: ANYSCALE_CLI_TOKEN not set — rayapp and anyscale CLI commands will fail."
fi

# --- Sideload required skills (/ask, /fix, /run, /inspect) from the
# anyscale-debug-agent repo. We don't clone it ourselves — Cursor pre-clones
# it via .cursor/environment.json's repositoryDependencies. Find that
# pre-clone and rsync .claude/skills/ from it. If we can't locate the
# pre-clone, fail loudly with a filesystem dump so we can adjust the search
# list (or fall back to a token-based clone). ---
DEBUG_AGENT_SRC=""
for p in \
    "$(pwd)/../anyscale-debug-agent" \
    /workspace/anyscale-debug-agent \
    /repos/anyscale-debug-agent \
    "$HOME/anyscale-debug-agent" \
    "$HOME/repositories/anyscale-debug-agent" \
    "$HOME/workspaces/anyscale-debug-agent"; do
  if [ -d "$p/.claude/skills" ]; then
    DEBUG_AGENT_SRC="$p"
    echo "Found anyscale-debug-agent pre-clone at: $DEBUG_AGENT_SRC"
    break
  fi
done

if [ -z "$DEBUG_AGENT_SRC" ]; then
  echo "ERROR: anyscale-debug-agent pre-clone not found at any expected path."
  echo "Filesystem inspection (looking for it):"
  find / -maxdepth 6 -name 'anyscale-debug-agent' -type d 2>/dev/null || true
  echo "Parent of CWD ($(pwd)/..):"
  ls -la "$(pwd)/.." 2>/dev/null || true
  exit 1
fi

mkdir -p ~/.claude/skills
# rsync --delete keeps ~/.claude/skills/ exactly mirrored to upstream — drops
# any locally-stale skill that was removed from anyscale-debug-agent.
rsync -a --delete "$DEBUG_AGENT_SRC/.claude/skills/" ~/.claude/skills/
echo "User-scope skills:"
ls ~/.claude/skills/

# --- Preflight: validate the env this script just set up. Same script the
# agent re-runs at task start (see AGENTS.md → Cursor Cloud → Preconditions). ---
bash .cursor/preflight.sh
