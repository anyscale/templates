#!/bin/bash
set -euo pipefail

: "${TEMPLATE_NAMES:?TEMPLATE_NAMES env var is required}"

TEMPLATES="$TEMPLATE_NAMES"

# Repo root = nearest ancestor dir with BUILD.yaml (robust to where this script lives).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/BUILD.yaml" ]; do
  ROOT="$(dirname "$ROOT")"
done
[ -f "$ROOT/BUILD.yaml" ] || { echo "repo root not found: no BUILD.yaml above ${BASH_SOURCE[0]}" >&2; exit 1; }
PYPROJECT="$ROOT/pyproject.toml"
ANYSCALE_VERSION=$(grep -oE 'anyscale==[0-9][0-9A-Za-z.+!-]*' "$PYPROJECT" | head -1 | cut -d= -f3)
: "${ANYSCALE_VERSION:?could not read anyscale pin from $PYPROJECT}"

for t in $TEMPLATES; do
  case "$t" in
    *[!a-zA-Z0-9_-]*)
      echo "Invalid template name: $t" >&2
      exit 1
      ;;
  esac
done

# BUILD.yaml's `timeout_in_sec` bounds the *test*; the job timeout bounds everything
# around it too -- workspace create, cluster start, image pull, dep install. So the
# job's must be the looser of the two, or it fires first and a slow-but-healthy run
# reads as a template bug. A flat 75 did exactly that to the two templates that
# budget more: biotech_boltz_screening at 90min and e2e-rag-deepdive at 120.
#
# Stdlib only, on purpose: this renders on a bare agent, and adding a pyyaml install
# to the fan-out step to read one integer is a poor trade.
STARTUP_ALLOWANCE_MIN=30
template_timeout_min() {
  python3 - "$ROOT/BUILD.yaml" "$1" "$STARTUP_ALLOWANCE_MIN" <<'PY'
import re, sys

build, want, allowance = sys.argv[1], sys.argv[2], int(sys.argv[3])
name, budget = None, None
for line in open(build):
    entry = re.match(r"^- name:\s*(\S+)", line)
    if entry:
        name = entry.group(1)
        continue
    seconds = re.match(r"^\s+timeout_in_sec:\s*(\d+)", line)
    if seconds and name == want:
        budget = int(seconds.group(1))
        break
# No entry, or none declared: keep what every step used before.
print(max(75, -(-budget // 60) + allowance) if budget else 75)
PY
}

echo "steps:"
for t in $TEMPLATES; do
  TIMEOUT="$(template_timeout_min "$t")"
  cat <<STEP
  - label: "Test template: $t"
    env:
      TEMPLATE_NAME: "$t"
    commands:
      - |
        set -euo pipefail
        export ANYSCALE_CLI_TOKEN="\$\$(aws --region=us-west-2 secretsmanager get-secret-value --secret-id \$\$ANYSCALE_CLI_TOKEN_SECRET_NAME | jq -r .SecretString)"
        export ANYSCALE_HOST="https://console.anyscale.com"
        bash download_rayapp.sh
        sudo apt-get update && sudo apt-get install -y rsync ca-certificates && sudo update-ca-certificates
        sudo pip install anyscale==${ANYSCALE_VERSION}
        LOG=/tmp/rayapp-\$\$TEMPLATE_NAME.log
        : > "\$\$LOG"
        # Watch for "Workspace created successfully id: expwrk_..." (always
        # printed by anyscale CLI), then build the canonical workspace URL
        # from cloud_id/project_id and post a buildkite annotation.
        # set +eo pipefail: grep returns non-zero when the log is still empty.
        (
          set +eo pipefail
          while :; do
            WS_ID=\$\$(grep 'Workspace created successfully id:' "\$\$LOG" 2>/dev/null \\
              | grep -oE 'expwrk_[a-z0-9]+' | head -1)
            if [ -n "\$\$WS_ID" ]; then
              JSON=\$\$(anyscale workspace_v2 get --id "\$\$WS_ID" -j 2>/dev/null)
              CLOUD_ID=\$\$(echo "\$\$JSON" | jq -r '.cloud_id // empty')
              PROJECT_ID=\$\$(echo "\$\$JSON" | jq -r '.project_id // empty')
              if [ -n "\$\$CLOUD_ID" ] && [ -n "\$\$PROJECT_ID" ]; then
                URL="\$\$ANYSCALE_HOST/\$\$CLOUD_ID/\$\$PROJECT_ID/workspaces/\$\$WS_ID"
                printf '**%s** workspace: %s\n' "\$\$TEMPLATE_NAME" "\$\$URL" \\
                  | buildkite-agent annotate --style info --context "ws-\$\$TEMPLATE_NAME"
                break
              fi
            fi
            sleep 2
          done
        ) &
        WATCHER_PID=\$\$!
        set +e
        ./rayapp test \$\$TEMPLATE_NAME 2>&1 | tee "\$\$LOG"
        EXIT=\$\${PIPESTATUS[0]}
        set -e
        kill "\$\$WATCHER_PID" 2>/dev/null || true
        wait "\$\$WATCHER_PID" 2>/dev/null || true
        exit \$\$EXIT
    timeout_in_minutes: ${TIMEOUT}
    agents:
      queue: small
    retry:
      automatic:
        # Agent-level deaths only -- the job never returned a status of its own, so
        # there is nothing in it to read. A command that *did* return stays red, even
        # when the cause was infra: a PyPI 502 surfaces as an ordinary exit 1, and
        # retrying that hides an outage behind a green tick. It also tripled the wall
        # clock on genuinely broken templates, which is how build 641 read as one
        # 78-minute run instead of two dead attempts and a good one.
        - exit_status: -1
          limit: 2
        - exit_status: 255
          limit: 2
    plugins:
      - docker#v5.9.0:
          image: "830883877497.dkr.ecr.us-west-2.amazonaws.com/anyscale/forge:241125"
          propagate-aws-auth-tokens: true
          mount-buildkite-agent: true
          shell: ["/bin/bash", "-e", "-c"]
          environment:
            - "BUILDKITE"
            - "BUILDKITE_PIPELINE_ID"
            - "TEMPLATE_NAME"
            - "ANYSCALE_CLI_TOKEN_SECRET_NAME"
STEP
done
