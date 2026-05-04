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
#   ANYSCALE_DEBUG_AGENT_GH_TOKEN  GitHub PAT — needs read on anyscale/anyscale-debug-agent
#                                   (skills clone) AND push/PR/comment/label on anyscale/templates
#                                   (gh fallback when Cursor's default auth lacks PR permissions).
#   ANYSCALE_CLI_TOKEN             For the anyscale CLI
#   GCP_TEMPLATE_REGISTRY_SA_KEY   GCP SA JSON for docker push to us-docker.pkg.dev
#   BUILDKITE_API_TOKEN            Read by the Buildkite MCP server (Dockerfile-baked).
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
: "${ANYSCALE_DEBUG_AGENT_GH_TOKEN:?secret ANYSCALE_DEBUG_AGENT_GH_TOKEN is empty/unset; add it in Cursor → Cloud Agents → My Secrets}"
echo "$ANYSCALE_DEBUG_AGENT_GH_TOKEN" | gh auth login --with-token

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
  echo "WARN: ANYSCALE_CLI_TOKEN not set — anyscale CLI commands will fail."
fi

# --- Sideload required skills (/ask, /fix, /run, /inspect) from
# anyscale-debug-agent. Required for the /template update flow — without
# /fix the agent cannot iterate on CI failures. Last in the script so a
# clone failure can't cascade into losing earlier auth/creds setup; the
# script still exits non-zero on failure. Sparse + shallow + blobless:
# fetches only .claude/skills/ (~1.4M) instead of the full 210M repo.
#
# Direct `git clone` with the token in the URL, not `gh repo clone`:
# newer `gh` doesn't auto-configure git's credential helper on
# `gh auth login --with-token`, so the underlying git clone fires
# anonymously and GitHub returns 404 on private repos. Embedding the
# token in the URL bypasses that auth chain entirely. ---
rm -rf /tmp/debug-agent
git clone --depth 1 --single-branch --no-checkout --filter=blob:none \
  "https://x-access-token:${ANYSCALE_DEBUG_AGENT_GH_TOKEN}@github.com/anyscale/anyscale-debug-agent.git" \
  /tmp/debug-agent
git -C /tmp/debug-agent sparse-checkout set .claude/skills
git -C /tmp/debug-agent checkout
mkdir -p ~/.claude/skills
cp -r /tmp/debug-agent/.claude/skills/* ~/.claude/skills/
echo "User-scope skills:"
ls ~/.claude/skills/
# Clean up the cloned dir — the .git/config has the token embedded and
# we've already extracted what we need.
rm -rf /tmp/debug-agent
