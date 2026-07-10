#!/usr/bin/env bash
# idle_sweep_loop.sh — self-scheduling idle-workspace sweeper.
#
# Companion to ../recipes/slack-idle-workspace-sweep.md. Runs one "tick" every
# $INTERVAL seconds: invokes headless Claude with the fixed tick-prompt (which runs
# idle_sweep.py DRY-RUN, reasons, and posts a Slack digest). Falls back to running
# the script directly if `claude` isn't installed/authed, so it works either way.
#
# Launch ONCE so it survives SSH disconnect (does NOT survive a workspace restart —
# that's why the host workspace needs idle-termination disabled):
#   setsid bash idle_sweep_loop.sh > /mnt/user_storage/idle-sweep/loop.log 2>&1 < /dev/null &
#
# Testing:    INTERVAL=60   (default here) — ticks every minute so you can watch it.
# Production:  INTERVAL=3600 — hourly. Just export INTERVAL=3600 before launching.
set -uo pipefail

CLOUD="${CLOUD:-aws-public-us-west-2}"
INTERVAL="${INTERVAL:-60}"                     # seconds between ticks (60=testing, 3600=hourly)
STATE="${STATE:-/mnt/user_storage/idle-sweep}"
SCRIPT="${SCRIPT:-$STATE/idle_sweep.py}"
PROMPT="${PROMPT:-$STATE/tick-prompt.txt}"
MODE="${MODE:-claude}"                         # claude | script

# Delivery is optional. Precedence: SLACK_WEBHOOK_URL (curl) > SLACK_CHANNEL_EMAIL (SMTP) > log-only.
# With neither set, the digest just goes to run.log — add SLACK_WEBHOOK_URL later to turn on Slack.

mkdir -p "$STATE"
LOG="$STATE/run.log"

post_slack() {  # $1 = digest -> Slack (webhook > email > log-only)
  if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
    local payload
    payload=$(python3 -c 'import json,sys; print(json.dumps({"text": sys.stdin.read()}))' <<<"$1")
    curl -sf -X POST -H 'Content-type: application/json' --data "$payload" \
      "$SLACK_WEBHOOK_URL" >/dev/null || echo "[warn] slack webhook post failed" | tee -a "$LOG"
  elif [ -n "${SLACK_CHANNEL_EMAIL:-}" ]; then
    printf '%s' "$1" | python3 "$STATE/slack_email.py" \
      || echo "[warn] slack email post failed" | tee -a "$LOG"
  else
    echo "[delivery=log-only] digest logged above; set SLACK_WEBHOOK_URL to enable Slack" | tee -a "$LOG"
  fi
}

echo "[start] $(date -u +%FT%TZ) cloud=$CLOUD interval=${INTERVAL}s mode=$MODE" | tee -a "$LOG"

while true; do
  ts=$(date -u +%FT%TZ)
  echo "[tick] $ts" | tee -a "$LOG"
  if [ "$MODE" = claude ] && command -v claude >/dev/null 2>&1; then
    claude -p "$(cat "$PROMPT")" \
      --allowedTools "Bash(anyscale *),Bash(python3 *),Bash(curl *),Read" \
      >> "$LOG" 2>&1 \
      || { echo "[warn] $ts claude tick failed -> script fallback" | tee -a "$LOG"
           out=$(python3 "$SCRIPT" "$CLOUD" 2>&1); echo "$out" >> "$LOG"; post_slack "$out"; }
  else
    out=$(python3 "$SCRIPT" "$CLOUD" 2>&1)
    echo "$out" >> "$LOG"
    post_slack "$out"
  fi
  sleep "$INTERVAL"
done
