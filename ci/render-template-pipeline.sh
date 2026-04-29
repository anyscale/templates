#!/bin/bash
set -euo pipefail

: "${TEMPLATE_NAMES_JSON:?TEMPLATE_NAMES_JSON env var is required}"

TEMPLATES=$(echo "$TEMPLATE_NAMES_JSON" | jq -r '.[]')

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
        sudo apt-get update && sudo apt-get install -y rsync
        sudo pip install anyscale==0.26.87
        LOG=/tmp/rayapp-\$\$TEMPLATE_NAME.log
        : > "\$\$LOG"
        # Background watcher: post the workspace URL annotation as soon as
        # anyscale CLI prints the "View and update dependencies here:" line.
        # Disable -e/pipefail in the subshell — grep returns non-zero when the
        # log doesn't yet contain the line, which would otherwise kill the watcher.
        (
          set +eo pipefail
          while :; do
            URL=\$\$(grep 'View and update dependencies here:' "\$\$LOG" 2>/dev/null \\
              | grep -oE 'https://console\.anyscale[^ ]+/workspaces/expwrk_[a-z0-9]+' \\
              | head -1)
            if [ -n "\$\$URL" ]; then
              printf '**%s** workspace: %s\n' "\$\$TEMPLATE_NAME" "\$\$URL" \\
                | buildkite-agent annotate --style info --context "ws-\$\$TEMPLATE_NAME"
              break
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
