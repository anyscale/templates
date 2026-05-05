#!/usr/bin/env bash
# Cursor Cloud preflight — see AGENTS.md → Cursor Cloud → Preconditions.

set -uo pipefail

failures=()

# Append a failure entry consisting of a summary line and an indented block of
# captured stderr/stdout output, so summaries and their underlying errors stay
# paired in the final report.
add_failure_with_output() {
  local summary="$1" output="$2" indented
  indented=$(printf '%s\n' "$output" | sed 's/^/        /')
  failures+=("$summary"$'\n'"      Raw error:"$'\n'"$indented")
}

# 1. Companion skills
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
if out=$(GH_TOKEN="${ANYSCALE_GH_TOKEN:-}" gh auth status 2>&1); then :; else
  add_failure_with_output \
    "gh auth: 'gh auth status' failed with ANYSCALE_GH_TOKEN — check token validity/scopes" \
    "$out"
fi

# `gh auth status` validates the token but won't flag tokens that are valid
# yet not SSO-authorized for the org — every PR write would 404.
if out=$(GH_TOKEN="${ANYSCALE_GH_TOKEN:-}" gh api /repos/anyscale/templates 2>&1); then :; else
  add_failure_with_output \
    "gh repo access: ANYSCALE_GH_TOKEN can't fetch anyscale/templates — likely missing SSO authorization for the org" \
    "$out"
fi

gcloud_out=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>&1)
if ! echo "$gcloud_out" | grep -q .; then
  add_failure_with_output \
    "gcloud auth: no active account — service-account activation failed (check GCP_TEMPLATE_REGISTRY_SA_KEY)" \
    "$gcloud_out"
fi

if [[ ! -f "$HOME/.docker/config.json" ]] \
    || ! grep -q "us-docker.pkg.dev" "$HOME/.docker/config.json" 2>/dev/null; then
  failures+=("docker auth: us-docker.pkg.dev not configured in ~/.docker/config.json — install.sh's 'gcloud auth configure-docker' step likely failed; custom-image push will fail")
fi

# Pinned to staging — Cursor Cloud agent must never run against prod.
# --no-interactive prevents the CLI from prompting (would hang in CI).
if out=$(ANYSCALE_HOST=https://console.anyscale-staging.com \
         anyscale cloud list --no-interactive 2>&1); then :; else
  add_failure_with_output \
    "anyscale auth: 'anyscale cloud list' against staging failed — check ANYSCALE_CLI_TOKEN is a staging token" \
    "$out"
fi

# Buildkite MCP server starts with a stale token and only 401s mid-task —
# probe the API directly to surface that here.
if [[ -n "${BUILDKITE_API_TOKEN:-}" ]]; then
  if out=$(curl -sS -H "Authorization: Bearer $BUILDKITE_API_TOKEN" \
           https://api.buildkite.com/v2/access-token 2>&1); then
    if ! echo "$out" | grep -q '"uuid"'; then
      add_failure_with_output \
        "buildkite auth: BUILDKITE_API_TOKEN rejected by api.buildkite.com — token expired or revoked" \
        "$out"
    fi
  else
    add_failure_with_output \
      "buildkite auth: 'curl https://api.buildkite.com/v2/access-token' failed" \
      "$out"
  fi
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
