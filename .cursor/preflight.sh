#!/usr/bin/env bash
# Cursor Cloud preflight — see AGENTS.md → Cursor Cloud → Preconditions.

set -uo pipefail

failures=()

# 1. Companion skills (rsync'd from Cursor's pre-clone of anyscale-debug-agent)
for s in ask fix run inspect; do
  if [[ ! -f "$HOME/.claude/skills/$s/SKILL.md" ]]; then
    failures+=("missing skill: ~/.claude/skills/$s/SKILL.md — install.sh couldn't find Cursor's anyscale-debug-agent pre-clone (check .cursor/environment.json repositoryDependencies)")
  fi
done

# 2. Environment variables
for var in ANYSCALE_CLI_TOKEN GCP_TEMPLATE_REGISTRY_SA_KEY BUILDKITE_API_TOKEN; do
  if [[ -z "${!var:-}" ]]; then
    failures+=("missing env var: $var (team-scope, non-empty)")
  fi
done

# 3. Auth verified — gh uses Cursor's native GH App auth, no PAT.
if ! gh auth status >/dev/null 2>&1; then
  failures+=("gh auth: 'gh auth status' failed — Cursor's native GH App auth not wired")
fi

if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | grep -q .; then
  failures+=("gcloud auth: no active account — service-account activation failed (check GCP_TEMPLATE_REGISTRY_SA_KEY)")
fi

if [[ ! -f "$HOME/.docker/config.json" ]] \
    || ! grep -q "us-docker.pkg.dev" "$HOME/.docker/config.json" 2>/dev/null; then
  failures+=("docker auth: us-docker.pkg.dev not configured in ~/.docker/config.json — install.sh's 'gcloud auth configure-docker' step likely failed; custom-image push will fail")
fi

# Pinned to staging — Cursor Cloud agent must never run against prod.
if ! ANYSCALE_HOST=https://console.anyscale-staging.com anyscale cloud list >/dev/null 2>&1; then
  failures+=("anyscale auth: 'anyscale cloud list' against staging failed — check ANYSCALE_CLI_TOKEN is a staging token")
fi

# Buildkite MCP server starts with a stale token and only 401s mid-task —
# probe the API directly to surface that here.
if [[ -n "${BUILDKITE_API_TOKEN:-}" ]] \
    && ! curl -sf -H "Authorization: Bearer $BUILDKITE_API_TOKEN" \
         https://api.buildkite.com/v2/access-token >/dev/null 2>&1; then
  failures+=("buildkite auth: BUILDKITE_API_TOKEN rejected by api.buildkite.com — token expired or revoked")
fi

# 4. Tools
if ! command -v rayapp >/dev/null 2>&1; then
  failures+=("rayapp not on PATH — install.sh's download_rayapp.sh step likely failed")
fi

if ! command -v buildkite-mcp-server >/dev/null 2>&1; then
  failures+=("buildkite-mcp-server not on PATH — Dockerfile bake step regressed; Buildkite MCP integration will fail to start")
fi

if [[ ${#failures[@]} -gt 0 ]]; then
  echo "Preflight FAILED:" >&2
  for f in "${failures[@]}"; do
    echo "  - $f" >&2
  done
  exit 1
fi

echo "Preflight OK: skills, secrets, and auth verified."
