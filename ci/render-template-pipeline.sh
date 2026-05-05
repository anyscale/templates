#!/bin/bash
set -euo pipefail

: "${TEMPLATE_NAMES:?TEMPLATE_NAMES env var is required}"

TEMPLATES="$TEMPLATE_NAMES"

# Single source of truth for the forge-image anyscale pin. Script-relative
# path (CWD-independent); awk so a missing line gives empty output rather
# than tripping set -e before the :? fires.
REQ_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../requirements-dev.txt"
ANYSCALE_VERSION=$(awk -F= '/^anyscale==/{print $3}' "$REQ_FILE")
: "${ANYSCALE_VERSION:?could not read anyscale pin from $REQ_FILE}"

for t in $TEMPLATES; do
  case "$t" in
    *[!a-zA-Z0-9_-]*)
      echo "Invalid template name: $t" >&2
      exit 1
      ;;
  esac
done

echo "steps:"
for t in $TEMPLATES; do
  cat <<STEP
  - label: "Test template: $t"
    env:
      TEMPLATE_NAME: "$t"
    commands:
      - |
        set -euo pipefail
        export ANYSCALE_CLI_TOKEN="\$\$(aws --region=us-west-2 secretsmanager get-secret-value --secret-id \$\$ANYSCALE_CLI_TOKEN_SECRET_NAME | jq -r .SecretString)"
        export ANYSCALE_HOST="https://console.anyscale-staging.com"
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
    timeout_in_minutes: 75
    agents:
      queue: small
    retry:
      automatic: true
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
