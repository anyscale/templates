#!/usr/bin/env bash
# Trigger Cursor Cloud "template-updater" agents to bump templates to a Ray version.
# One agent -> one template -> one draft PR (each runs the non-interactive workflow
# .claude/skills/template/workflows/bump-ray-version.md, per AGENTS.md "Cursor Cloud").
#
# Credentials are read from the environment — NEVER hardcoded, never committed.
# Stored locally for now (a GitHub Actions / repo-secrets path can come later):
#   CURSOR_TEMPLATE_UPDATER_AUTH_TOKEN     (required to launch)  Cursor API key, sent as Bearer.
#                                          Falls back to CURSOR_API_KEY if unset.
#   CURSOR_TEMPLATE_UPDATER_WEBHOOK        (optional)  https URL Cursor POSTs statusChange callbacks to.
#   CURSOR_TEMPLATE_UPDATER_WEBHOOK_SECRET (optional)  shared secret for the X-Webhook-Signature HMAC.
#   CURSOR_AGENTS_URL  (optional)  default https://api.cursor.com/v0/agents  (v0 supports webhooks).
#   TEMPLATES_REPO     (optional)  default https://github.com/anyscale/templates
#
# Template selection: pass names explicitly, or --all to expand to every *maintained*
# BUILD.yaml entry. `maintained: false` entries (archived templates) are always skipped;
# a named entry that is unmaintained or absent from BUILD.yaml is skipped with a warning.
#
# SAFE BY DEFAULT: without --execute the script only PREVIEWS (prints the payloads
# it would POST, makes zero API calls). Launching real agents requires an explicit
# --execute — so a mistyped or swallowed flag can never launch anything.
#
# Usage:
#   export CURSOR_TEMPLATE_UPDATER_AUTH_TOKEN=...          # from your vault
#   ci/trigger-cursor-bump.sh [-v 2.56.0] [-r main] [--all] [--exclude a,b,c] [--list] [--dry-run] [--execute] [<template>...]
#
#   # Preview (no --execute → no launch, no creds needed):
#   ci/trigger-cursor-bump.sh -v 2.56.0 job-intro object-detection-video-processing skyrl
#   ci/trigger-cursor-bump.sh -v 2.56.0 --all --exclude job-intro,object-detection-video-processing,skyrl --list
#   # Test batch (one per image-URI case) — add --execute to actually launch:
#   ci/trigger-cursor-bump.sh -v 2.56.0 job-intro object-detection-video-processing skyrl --execute
#   # Fanout (all maintained, minus the already-done test batch):
#   ci/trigger-cursor-bump.sh -v 2.56.0 --all --exclude job-intro,object-detection-video-processing,skyrl --execute
set -euo pipefail

RAY_VERSION="2.56.0"
REF="main"
DRY_RUN=0
LIST_ONLY=0
EXECUTE=0
USE_ALL=0
EXCLUDE_CSV=""
CURSOR_AGENTS_URL="${CURSOR_AGENTS_URL:-https://api.cursor.com/v0/agents}"
TEMPLATES_REPO="${TEMPLATES_REPO:-https://github.com/anyscale/templates}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_YAML="$SCRIPT_DIR/../BUILD.yaml"

usage() { grep '^#' "$0" | sed 's/^# \?//'; }

# Positional template names accumulate here (newline-separated); flags may appear
# before, after, or interspersed with them.
POSITIONAL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--ray-version) RAY_VERSION="$2"; shift 2 ;;
    -r|--ref)         REF="$2"; shift 2 ;;
    --all)            USE_ALL=1; shift ;;
    --exclude)        EXCLUDE_CSV="$2"; shift 2 ;;
    --list)           LIST_ONLY=1; shift ;;
    --dry-run)        DRY_RUN=1; shift ;;
    --execute)        EXECUTE=1; shift ;;
    -h|--help)        usage; exit 0 ;;
    --)               shift; while [[ $# -gt 0 ]]; do POSITIONAL="$(printf '%s\n%s' "$POSITIONAL" "$1")"; shift; done ;;
    -*)               echo "unknown flag: $1" >&2; exit 2 ;;
    *)                POSITIONAL="$(printf '%s\n%s' "$POSITIONAL" "$1")"; shift ;;
  esac
done
POSITIONAL="$(printf '%s\n' "$POSITIONAL" | sed '/^$/d')"

command -v jq >/dev/null || { echo "error: jq required" >&2; exit 2; }
[[ -f "$BUILD_YAML" ]] || { echo "error: BUILD.yaml not found at $BUILD_YAML" >&2; exit 2; }

# Classify every BUILD.yaml entry as maintained/unmaintained. Default is
# maintained; a `maintained: false` line in the entry flips it (the validator
# enforces that archive/ entries carry it). Emits "<state>\t<name>" per entry.
classify() {
  awk '
    function flush() { if (n != "") print (m == "false" ? "unmaintained" : "maintained") "\t" n }
    /^-[[:space:]]*name:/ {
      flush(); n = $0; sub(/^-[[:space:]]*name:[[:space:]]*/, "", n); m = "true"; next
    }
    /^[[:space:]]+maintained:/ {
      v = $0; sub(/^[[:space:]]*maintained:[[:space:]]*/, "", v); m = (v ~ /^false/) ? "false" : "true"
    }
    END { flush() }
  ' "$BUILD_YAML"
}
CLASSIFIED="$(classify)"
MAINTAINED_NAMES="$(printf '%s\n' "$CLASSIFIED" | awk -F'\t' '$1 == "maintained" { print $2 }')"
ALL_NAMES="$(printf '%s\n' "$CLASSIFIED" | awk -F'\t' '{ print $2 }')"

# is_in NEEDLE NEWLINE-LIST  -> exit 0 if NEEDLE is an exact line of the list.
is_in() { printf '%s\n' "$2" | grep -qxF -- "$1"; }

# Resolve the requested set: --all (every maintained entry) or explicit names.
if [[ "$USE_ALL" == 1 ]]; then
  [[ -z "$POSITIONAL" ]] || { echo "error: pass template names OR --all, not both" >&2; exit 2; }
  REQUESTED="$MAINTAINED_NAMES"
else
  [[ -n "$POSITIONAL" ]] || { echo "error: pass at least one template name, or --all" >&2; exit 2; }
  REQUESTED="$POSITIONAL"
fi

# --exclude a,b,c -> newline list.
EXCLUDE_LIST=""
[[ -n "$EXCLUDE_CSV" ]] && EXCLUDE_LIST="$(printf '%s' "$EXCLUDE_CSV" | tr ',' '\n' | sed '/^$/d')"

# Final launch list: drop excluded / unknown / unmaintained / duplicate names.
FINAL=""
while IFS= read -r t; do
  [[ -n "$t" ]] || continue
  if [[ -n "$EXCLUDE_LIST" ]] && is_in "$t" "$EXCLUDE_LIST"; then
    echo "skip (excluded): $t" >&2; continue
  fi
  if ! is_in "$t" "$ALL_NAMES"; then
    echo "skip (not in BUILD.yaml): $t" >&2; continue
  fi
  if ! is_in "$t" "$MAINTAINED_NAMES"; then
    echo "skip (maintained: false): $t" >&2; continue
  fi
  is_in "$t" "$FINAL" && continue   # dedupe
  FINAL="$(printf '%s\n%s' "$FINAL" "$t")"
done <<< "$REQUESTED"
FINAL="$(printf '%s\n' "$FINAL" | sed '/^$/d')"

COUNT="$(printf '%s\n' "$FINAL" | sed '/^$/d' | wc -l | tr -d ' ')"
[[ "$COUNT" -ge 1 ]] || { echo "error: no templates to launch after filtering" >&2; exit 2; }

# --list: print the resolved set and exit (no creds, no launch).
if [[ "$LIST_ONLY" == 1 ]]; then
  echo "resolved $COUNT template(s):" >&2
  printf '%s\n' "$FINAL"
  exit 0
fi

# SAFE BY DEFAULT: only --execute performs real API calls; anything else previews.
# This makes a mistyped or swallowed flag fail closed (preview) instead of launching.
if [[ "$EXECUTE" != 1 ]]; then
  [[ "$DRY_RUN" == 1 ]] || echo "note: preview only — no agents launched. Re-run with --execute to launch for real." >&2
  DRY_RUN=1
fi

# Auth is only needed to actually POST.
TOKEN="${CURSOR_TEMPLATE_UPDATER_AUTH_TOKEN:-${CURSOR_API_KEY:-}}"
[[ "$DRY_RUN" == 1 || -n "$TOKEN" ]] || {
  echo "error: set CURSOR_TEMPLATE_UPDATER_AUTH_TOKEN (or CURSOR_API_KEY) to launch" >&2; exit 2
}

# Webhook (optional). Redact the values under --dry-run so nothing sensitive prints.
WH_URL="${CURSOR_TEMPLATE_UPDATER_WEBHOOK:-}"
WH_SECRET="${CURSOR_TEMPLATE_UPDATER_WEBHOOK_SECRET:-}"
if [[ "$DRY_RUN" == 1 ]]; then
  [[ -n "$WH_URL" ]]    && WH_URL='[redacted CURSOR_TEMPLATE_UPDATER_WEBHOOK]'
  [[ -n "$WH_SECRET" ]] && WH_SECRET='[redacted CURSOR_TEMPLATE_UPDATER_WEBHOOK_SECRET]'
fi

echo "== $([[ "$DRY_RUN" == 1 ]] && echo 'DRY RUN — ')launching $COUNT agent(s): Ray $RAY_VERSION, ref $REF ==" >&2
[[ -n "$WH_URL" ]] \
  && echo "   webhook: on" >&2 \
  || echo "   webhook: off (set CURSOR_TEMPLATE_UPDATER_WEBHOOK to enable statusChange callbacks)" >&2

launched=0
while IFS= read -r tmpl; do
  [[ -n "$tmpl" ]] || continue
  prompt="You are the non-interactive \`template-updater\` Cursor Cloud agent. \
Run the Ray-version bump workflow at .claude/skills/template/workflows/bump-ray-version.md \
for template '${tmpl}', target Ray version ${RAY_VERSION}. \
Follow AGENTS.md 'Cursor Cloud' (preflight, cursor/ branch naming, GH_TOKEN=\$ANYSCALE_GH_TOKEN on gh writes, \
ray-update + cursor-cloud PR labels). One template, one draft PR. Do not read or act on PR comments. \
Stop-and-report on any blocked precondition."

  payload="$(jq -n \
    --arg text "$prompt" --arg repo "$TEMPLATES_REPO" --arg ref "$REF" \
    --arg whurl "$WH_URL" --arg whsecret "$WH_SECRET" \
    '{prompt: {text: $text}, source: {repository: $repo, ref: $ref}}
     + (if $whurl == "" then {}
        else {webhook: ({url: $whurl} + (if $whsecret == "" then {} else {secret: $whsecret} end))} end)')"

  if [[ "$DRY_RUN" == 1 ]]; then
    echo "-- DRY RUN: $tmpl --" >&2   # marker to stderr; stdout stays pure JSON (jq-able)
    printf '%s\n' "$payload" | jq .
    continue
  fi

  echo "-- launching: $tmpl (Ray $RAY_VERSION) --"
  # `|| true`: a single curl failure (network blip, 5xx) must not abort the
  # whole batch under `set -e` — fall through to the unexpected-response branch.
  resp="$(curl -sS -X POST "$CURSOR_AGENTS_URL" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$payload" || true)"
  if printf '%s' "$resp" | jq -e . >/dev/null 2>&1; then
    printf '%s' "$resp" | jq -r '"  agent: \(.id // "?")   status: \(.status // "?")   url: \(.target.url // .url // "?")"'
    launched=$((launched + 1))
  else
    echo "  unexpected response: $resp" >&2
  fi
done <<< "$FINAL"

if [[ "$DRY_RUN" == 1 ]]; then
  echo "== dry run complete: $COUNT payload(s) ==" >&2
else
  echo "== launched $launched/$COUNT agent(s) ==" >&2
fi
