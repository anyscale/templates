# Cursor Cloud Agent base image for anyscale/templates.
#
# Split between this Dockerfile and .cursor/install.sh:
#   - Dockerfile: deterministic, version-pinned, secret-free tooling. Built
#     once by Cursor, cached as image layers. Adding/bumping tools here
#     triggers an automatic image rebuild on the next agent run.
#   - install.sh: runtime auth (per-run secrets) + repo-tree-dependent
#     setup (pre-commit hook, rayapp pinned via repo's download_rayapp.sh,
#     skills install via 'anyscale skills install'). Runs on every VM start.
#
# This Dockerfile also puts environment.json into "Manual Dockerfile" mode
# so Cursor honors the install command instead of its agent-driven setup.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# --- Base OS packages ---
# rsync + openssh-client are needed by `anyscale workspace push` (which
# `rayapp test` calls). python-is-python3 makes `python` resolve, which
# `update_deps.sh` and the depset/raydepsets scripts need.
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl wget ca-certificates sudo gnupg lsb-release \
      git python3 python3-pip python-is-python3 \
      rsync openssh-client \
 && rm -rf /var/lib/apt/lists/*

# --- gh CLI ---
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update -qq && apt-get install -y gh \
 && rm -rf /var/lib/apt/lists/*

# --- docker.io (for byod custom image flow via publish-custom-image.sh) ---
RUN apt-get update -qq && apt-get install -y docker.io fuse-overlayfs iptables \
 && rm -rf /var/lib/apt/lists/*

# --- Python tooling (pinned via requirements-dev.txt, the single source of
# truth shared with CI and human contributors). System-wide install so
# binaries land in /usr/local/bin (already on PATH). --break-system-packages:
# Ubuntu 24.04's Python 3.12 enforces PEP 668; safe in this single-purpose
# container. ---
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN python3 -m pip install --no-warn-script-location --break-system-packages \
      -r /tmp/requirements-dev.txt \
 && rm /tmp/requirements-dev.txt

# --- gcloud SDK (installed at /opt/google-cloud-sdk; bin added to PATH below) ---
RUN curl -sSL https://sdk.cloud.google.com > /tmp/gcloud-install.sh \
 && bash /tmp/gcloud-install.sh --disable-prompts --install-dir=/opt \
 && rm /tmp/gcloud-install.sh

# --- Buildkite MCP server (pinned; the cursor cloud automation's MCP config
# invokes `buildkite-mcp-server stdio` and the binary reads BUILDKITE_API_TOKEN
# from the agent's env, populated by the Cursor secret). ---
ARG BK_MCP_VER=v1.0.0
RUN curl -fsSL "https://github.com/buildkite/buildkite-mcp-server/releases/download/${BK_MCP_VER}/buildkite-mcp-server_Linux_x86_64.tar.gz" \
      | tar -xz -C /usr/local/bin buildkite-mcp-server \
 && chmod +x /usr/local/bin/buildkite-mcp-server

# --- Bake env into the image so ALL processes see it (login, non-login,
# interactive, non-interactive). Replaces the prior /etc/profile.d/ + ~/.bashrc
# dance, which depended on shell mode. ---
ENV PATH=/opt/google-cloud-sdk/bin:$PATH
ENV ANYSCALE_HOST=https://console.anyscale-staging.com

WORKDIR /workspace
