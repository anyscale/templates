#!/usr/bin/env bash
# Preflight checks for Cursor Cloud agent runs on anyscale/templates.
# Hard-exits with a list of missing preconditions if any check fails.
# See AGENTS.md → Cursor Cloud → Preconditions.

set -uo pipefail

failures=()

# 1. Companion skills (cloned from anyscale/anyscale-debug-agent by install.sh)
for s in ask fix run inspect; do
  if [[ ! -f "$HOME/.claude/skills/$s/SKILL.md" ]]; then
    failures+=("missing skill: ~/.claude/skills/$s/SKILL.md (clone of anyscale/anyscale-debug-agent failed? Check ANYSCALE_GH_TOKEN)")
  fi
done

# 2. Environment variables
for var in ANYSCALE_GH_TOKEN ANYSCALE_CLI_TOKEN GCP_TEMPLATE_REGISTRY_SA_KEY BUILDKITE_API_TOKEN; do
  if [[ -z "${!var:-}" ]]; then
    failures+=("missing env var: $var (team-scope, non-empty)")
  fi
done

# 3. Auth verified
if ! GH_TOKEN="${ANYSCALE_GH_TOKEN:-}" gh auth status >/dev/null 2>&1; then
  failures+=("gh auth: 'gh auth status' failed with ANYSCALE_GH_TOKEN — check token validity/scopes")
fi

# Verifies the token has access to anyscale/templates specifically. Catches
# tokens that are valid but not SSO-authorized for the org — `gh auth status`
# won't flag those, but every PR write would 404.
if ! GH_TOKEN="${ANYSCALE_GH_TOKEN:-}" gh api /repos/anyscale/templates >/dev/null 2>&1; then
  failures+=("gh repo access: ANYSCALE_GH_TOKEN can't fetch anyscale/templates — likely missing SSO authorization for the org")
fi

if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | grep -q .; then
  failures+=("gcloud auth: no active account — service-account activation failed (check GCP_TEMPLATE_REGISTRY_SA_KEY)")
fi

# Without this, `docker push us-docker.pkg.dev/...` fails with a confusing
# auth error mid custom-image rebuild. install.sh runs `gcloud auth
# configure-docker us-docker.pkg.dev` to wire it up.
if [[ ! -f "$HOME/.docker/config.json" ]] \
    || ! grep -q "us-docker.pkg.dev" "$HOME/.docker/config.json" 2>/dev/null; then
  failures+=("docker auth: us-docker.pkg.dev not configured in ~/.docker/config.json — install.sh's 'gcloud auth configure-docker' step likely failed; custom-image push will fail")
fi

# Pinned to staging — Cursor Cloud agent must never run against prod. The
# Dockerfile sets ANYSCALE_HOST=staging globally; this re-asserts it locally
# in case it's been overridden.
if ! ANYSCALE_HOST=https://console.anyscale-staging.com anyscale cloud list >/dev/null 2>&1; then
  failures+=("anyscale auth: 'anyscale cloud list' against staging failed — check ANYSCALE_CLI_TOKEN is a staging token")
fi

# Live token check — the Buildkite MCP server starts fine with a stale token
# and only 401s mid-task, so we probe the API directly here.
if [[ -n "${BUILDKITE_API_TOKEN:-}" ]] \
    && ! curl -sf -H "Authorization: Bearer $BUILDKITE_API_TOKEN" \
         https://api.buildkite.com/v2/access-token >/dev/null 2>&1; then
  failures+=("buildkite auth: BUILDKITE_API_TOKEN rejected by api.buildkite.com — token expired or revoked")
fi

# 4. Tools (Dockerfile-baked except rayapp, which install.sh downloads)
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
