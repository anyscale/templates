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

pip install -q --upgrade anyscale

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

# --- Auth: gh ---
echo "$ANYSCALE_DEBUG_AGENT_GH_TOKEN" | gh auth login --with-token

# --- Sideload private skills from anyscale-debug-agent ---
rm -rf /tmp/debug-agent
gh repo clone anyscale/anyscale-debug-agent /tmp/debug-agent
mkdir -p ~/.claude/skills
cp -r /tmp/debug-agent/skills/* ~/.claude/skills/
echo "User-scope skills:"
ls ~/.claude/skills/

# --- Auth: gcloud (for docker push to GCP artifact registry) ---
if [ -n "${GCP_TEMPLATE_REGISTRY_SA_KEY:-}" ]; then
  echo "$GCP_TEMPLATE_REGISTRY_SA_KEY" > /tmp/gcp-sa.json
  gcloud auth activate-service-account --key-file=/tmp/gcp-sa.json
  gcloud auth configure-docker us-docker.pkg.dev --quiet
fi

# --- Auth: anyscale CLI ---
if [ -n "${ANYSCALE_CLI_TOKEN:-}" ]; then
  mkdir -p ~/.anyscale
  cat > ~/.anyscale/credentials.json <<EOF
{"cli_token": "$ANYSCALE_CLI_TOKEN"}
EOF
fi
