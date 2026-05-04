#!/usr/bin/env bash
# Cursor Cloud Agent install script.
# Idempotent: runs on every VM startup. Cached via Cursor's snapshot.
# Required secrets (set in Cursor → My Secrets, exposed as env vars):
#   ANYSCALE_DEBUG_AGENT_GH_TOKEN  GitHub PAT — needs read on anyscale/anyscale-debug-agent
#                                   (skills clone) AND push/PR/comment/label on anyscale/templates
#                                   (gh fallback when Cursor's default auth lacks PR permissions).
#   ANYSCALE_CLI_TOKEN             For the anyscale CLI
#   GCP_TEMPLATE_REGISTRY_SA_KEY   GCP SA JSON for docker push to us-docker.pkg.dev
set -euo pipefail

# --- CLIs ---
if ! command -v gh &>/dev/null; then
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y gh
fi

# --- Python tooling (versions pinned to match this repo's CI) ---
# --break-system-packages: Ubuntu 24.04's Python 3.12 enforces PEP 668 and
# refuses `pip install --user` without it. Safe in this single-purpose
# container — there's no user Python env to break.
python3 -m pip install --user --no-warn-script-location --break-system-packages \
  pre-commit==3.8.0 nbconvert==7.17.1 anyscale==0.26.87 \
  pyyaml==6.0.3 pydantic==2.13.3
export PATH="$HOME/.local/bin:$PATH"
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc 2>/dev/null \
  || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# --- Persist env for subsequent agent shells. ~/.bashrc only fires for
# interactive shells (Ubuntu's stock ~/.bashrc early-returns when $PS1 is
# unset), so login shells need /etc/profile.d/ to pick up these exports.
# Written early so partial install failures still leave env persisted.
# Idempotent: tee overwrites. ---
sudo tee /etc/profile.d/cursor-env.sh > /dev/null <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
export PATH="$PATH:$HOME/google-cloud-sdk/bin"
export ANYSCALE_HOST="https://console.anyscale-staging.com"
EOF

# --- pre-commit hooks (auto-fire on git commit; idempotent) ---
# Non-fatal: pre-commit refuses if core.hooksPath is set (and other niche
# git configs). The agent can still run `pre-commit run --all-files` by hand.
if [ -f .pre-commit-config.yaml ] && [ -d .git ]; then
  pre-commit install \
    || echo "WARN: pre-commit install skipped — run 'pre-commit run --all-files' manually before committing."
fi

# --- rayapp (version pinned via repo's download_rayapp.sh) ---
if [ -f download_rayapp.sh ] && ! command -v rayapp &>/dev/null; then
  bash download_rayapp.sh
  mkdir -p "$HOME/.local/bin"
  mv rayapp "$HOME/.local/bin/rayapp"
fi

if ! command -v gcloud &>/dev/null; then
  curl -sSL https://sdk.cloud.google.com > /tmp/gcloud-install.sh
  bash /tmp/gcloud-install.sh --disable-prompts --install-dir="$HOME"
  echo 'export PATH=$PATH:$HOME/google-cloud-sdk/bin' >> ~/.bashrc
fi
export PATH=$PATH:$HOME/google-cloud-sdk/bin

if ! command -v docker &>/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y docker.io fuse-overlayfs iptables
fi

# --- Buildkite MCP server (binary; the cursor cloud automation's MCP config
# invokes `buildkite-mcp-server stdio` and the binary picks up BUILDKITE_API_TOKEN
# from the agent's env, populated by the Cursor secret). ---
if ! command -v buildkite-mcp-server &>/dev/null; then
  BK_MCP_VER=$(curl -fsSL https://api.github.com/repos/buildkite/buildkite-mcp-server/releases/latest \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['tag_name'])")
  curl -fsSL "https://github.com/buildkite/buildkite-mcp-server/releases/download/${BK_MCP_VER}/buildkite-mcp-server_Linux_x86_64.tar.gz" \
    | tar -xz -C "$HOME/.local/bin" buildkite-mcp-server
  chmod +x "$HOME/.local/bin/buildkite-mcp-server"
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
# Point at staging to match CI (test-template uses anyscale_cli_token_ci_staging).
export ANYSCALE_HOST="https://console.anyscale-staging.com"
grep -qxF 'export ANYSCALE_HOST="https://console.anyscale-staging.com"' ~/.bashrc 2>/dev/null \
  || echo 'export ANYSCALE_HOST="https://console.anyscale-staging.com"' >> ~/.bashrc

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
