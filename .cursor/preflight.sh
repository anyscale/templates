#!/usr/bin/env bash
# Preflight checks for Cursor Cloud agent runs on anyscale/templates.
# Hard-exits with a list of missing preconditions if any check fails.
# See AGENTS.md → Cursor Cloud → Preconditions.

set -uo pipefail

failures=()

# 1. Companion skills (cloned from anyscale/anyscale-debug-agent by install.sh)
for s in ask fix run inspect; do
  if [[ ! -f "$HOME/.claude/skills/$s/SKILL.md" ]]; then
    failures+=("missing skill: ~/.claude/skills/$s/SKILL.md (clone of anyscale/anyscale-debug-agent failed?)")
  fi
done

# 2. Cursor team-scope secrets
for var in ANYSCALE_DEBUG_AGENT_GH_TOKEN ANYSCALE_CLI_TOKEN GCP_TEMPLATE_REGISTRY_SA_KEY BUILDKITE_API_TOKEN; do
  if [[ -z "${!var:-}" ]]; then
    failures+=("missing env var: $var (Cursor team-scope secret not provisioned)")
  fi
done

# 3. Auth verified
if ! GH_TOKEN="${ANYSCALE_DEBUG_AGENT_GH_TOKEN:-}" gh auth status >/dev/null 2>&1; then
  failures+=("gh auth: 'gh auth status' failed with ANYSCALE_DEBUG_AGENT_GH_TOKEN — check token validity/scopes")
fi

if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | grep -q .; then
  failures+=("gcloud auth: no active account — service-account activation failed (check GCP_TEMPLATE_REGISTRY_SA_KEY)")
fi

if ! anyscale cloud list >/dev/null 2>&1; then
  failures+=("anyscale auth: 'anyscale cloud list' failed — check ANYSCALE_CLI_TOKEN")
fi

if [[ ${#failures[@]} -gt 0 ]]; then
  echo "Preflight FAILED:" >&2
  for f in "${failures[@]}"; do
    echo "  - $f" >&2
  done
  exit 1
fi

echo "Preflight OK: skills, secrets, and auth verified."
