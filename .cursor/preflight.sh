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

# Verifies the token can do PR write operations (create/edit/comment/label)
# on this repo. .permissions.push from the repo response reflects the calling
# user's role — Write or higher returns true; everything PR-write needs that.
# Catches: invalid token, missing SSO authorization, user not a collaborator,
# token revoked.
# Limitation: for fine-grained PATs this reflects the user's role, not the
# token's configured scopes. A token belonging to an admin user but scoped
# narrowly would still report push:true. No API closes that gap without an
# actual write.
push_perm=$(GH_TOKEN="${ANYSCALE_GH_TOKEN:-}" gh api /repos/anyscale/templates --jq '.permissions.push' 2>/dev/null)
if [[ "$push_perm" != "true" ]]; then
  failures+=("gh repo access: ANYSCALE_GH_TOKEN doesn't grant push to anyscale/templates (PR create/edit/comment/label will fail) — check SSO authorization and repo permissions")
fi

if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | grep -q .; then
  failures+=("gcloud auth: no active account — service-account activation failed (check GCP_TEMPLATE_REGISTRY_SA_KEY)")
fi

if ! anyscale cloud list >/dev/null 2>&1; then
  failures+=("anyscale auth: 'anyscale cloud list' failed — check ANYSCALE_CLI_TOKEN")
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

if [[ ${#failures[@]} -gt 0 ]]; then
  echo "Preflight FAILED:" >&2
  for f in "${failures[@]}"; do
    echo "  - $f" >&2
  done
  exit 1
fi

echo "Preflight OK: skills, secrets, and auth verified."
