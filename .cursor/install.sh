#!/usr/bin/env bash
# Cursor Cloud Agent install script.
# Idempotent: runs on every VM startup. Cached via Cursor's snapshot.
# Required secrets (set in Cursor → My Secrets, exposed as env vars):
#   ANYSCALE_DEBUG_AGENT_GH_TOKEN  GitHub PAT with read on anyscale/anyscale-debug-agent
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
python3 -m pip install --user --no-warn-script-location \
  pre-commit==3.8.0 nbconvert==7.17.1 anyscale==0.26.87 \
  pyyaml==6.0.3 pydantic==2.13.3
export PATH="$HOME/.local/bin:$PATH"
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc 2>/dev/null \
  || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# --- pre-commit hooks (auto-fire on git commit; idempotent) ---
if [ -f .pre-commit-config.yaml ] && [ -d .git ]; then
  pre-commit install
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

# --- Auth: gh (hard requirement — used to clone debug-agent skills) ---
: "${ANYSCALE_DEBUG_AGENT_GH_TOKEN:?secret ANYSCALE_DEBUG_AGENT_GH_TOKEN is empty/unset; add it in Cursor → Cloud Agents → My Secrets}"
echo "$ANYSCALE_DEBUG_AGENT_GH_TOKEN" | gh auth login --with-token

# --- Sideload private skills from anyscale-debug-agent ---
rm -rf /tmp/debug-agent
GH_TOKEN="$ANYSCALE_DEBUG_AGENT_GH_TOKEN" gh repo clone anyscale/anyscale-debug-agent /tmp/debug-agent
mkdir -p ~/.claude/skills
cp -r /tmp/debug-agent/skills/* ~/.claude/skills/
echo "User-scope skills:"
ls ~/.claude/skills/

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
