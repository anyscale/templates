#!/usr/bin/env python3
"""Emit the publish-templates sub-pipeline based on drift + `go` label."""

from __future__ import annotations

import json
import os
import sys


def _empty_pipeline() -> dict:
    return {"steps": []}


def _trigger_steps(drifted: list[str], pr_number: str, head_sha: str, head_ref: str) -> dict:
    steps = []
    for name in drifted:
        steps.append({
            "label": f":rocket: Publish {name}",
            "trigger": "tmpl-publish",
            "build": {
                # build.sh clones templates itself; `branch` is metadata only.
                # Setting it to PR head ref scopes cancel_running_branch_builds
                # on tmpl-publish to same-PR predecessors.
                "branch": head_ref,
                "message": f"Publish {name} for PR #{pr_number}",
                # env drives tmpl-publish's `if:` guards (Buildkite conditionals
                # don't support meta_data). meta_data still needed by build.sh.
                "env": {
                    "TMPL_NAME": name,
                    "AUTO_PUBLISH_DEV": "true",
                    "AUTO_PUBLISH_STAGING": "true",
                },
                "meta_data": {
                    "tmpl-name": name,
                    "tmpl-branch": head_ref,
                    "tmpl-commit": head_sha,
                    "auto_publish_dev": "true",
                    "auto_publish_staging": "true",
                },
            },
        })
    return {"steps": steps}


def _wait_for_label_pipeline(drifted: list[str], pr_number: str) -> dict:
    n = len(drifted)
    plural = "" if n == 1 else "s"
    # Defeats manual-Unblock-without-`go` bypass. Requires $GH_TOKEN on the
    # publish-templates pipeline env.
    verify_cmd = f"""set -euo pipefail

if ! LABELS=$(gh api repos/anyscale/templates/issues/{pr_number}/labels --jq '.[].name'); then
  buildkite-agent annotate --style error "Failed to fetch PR labels (gh api error — check GH_TOKEN)."
  exit 1
fi

if ! echo "$LABELS" | grep -qx go; then
  buildkite-agent annotate --style error "Manual unblock without 'go' label is not allowed."
  exit 1
fi

echo "go label present; allowing publish"
"""
    return {
        "steps": [
            {
                "block": f":pause_button: Apply 'go' label on PR to publish {n} template{plural}",
                "key": "wait-for-go-label",
                "prompt": (
                    f"This PR drifts {n} template{plural} ({', '.join(drifted)}). "
                    f"Apply the `go` label on the PR to publish them. The build "
                    f"will re-trigger automatically."
                ),
            },
            {
                "label": ":lock: Verify 'go' label",
                "key": "verify-go-label",
                "depends_on": "wait-for-go-label",
                "command": verify_cmd,
                "agents": {"queue": "small"},
            },
        ],
    }


def main() -> int:
    drifted = json.loads(os.environ.get("DRIFTED", "[]"))
    go_present = os.environ.get("GO_PRESENT", "false") == "true"
    pr_number = os.environ.get("BUILDKITE_PULL_REQUEST", "")
    pr_head_sha = os.environ.get("BUILDKITE_COMMIT", "")
    pr_head_ref = os.environ.get("BUILDKITE_BRANCH", "")

    if not drifted:
        pipeline = _empty_pipeline()
    elif go_present:
        pipeline = _trigger_steps(drifted, pr_number, pr_head_sha, pr_head_ref)
    else:
        pipeline = _wait_for_label_pipeline(drifted, pr_number)

    json.dump(pipeline, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
