#!/usr/bin/env python3
"""Emit the dynamic publish-templates sub-pipeline.

Reads {drifted_templates, go_present, pr_number, pr_head_sha, pr_head_ref}
from buildkite-agent meta-data. Emits one of three pipelines to stdout:

  1. drift == 0                       -> empty (build short-circuits green)
  2. drift  > 0 AND go_present=true   -> N parallel `trigger:` -> tmpl-publish
  3. drift  > 0 AND go_present!=true  -> placeholder block + verify (re-checks
                                         the `go` label to defeat manual
                                         Unblock-without-`go` bypass)
"""

from __future__ import annotations

import json
import subprocess
import sys

import yaml


def _meta(key: str, default: str = "") -> str:
    # buildkite-agent exits 100 when the key is missing; treat that as "use default".
    result = subprocess.run(
        ["buildkite-agent", "meta-data", "get", key],
        capture_output=True,
        text=True,
    )
    if result.returncode == 100:
        return default
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)
    return result.stdout.strip()


def _empty_pipeline() -> dict:
    return {"steps": []}


def _trigger_steps(drifted: list[str], pr_number: str, head_sha: str, head_ref: str) -> dict:
    steps = []
    for name in drifted:
        steps.append({
            "label": f":rocket: Publish {name}",
            "trigger": "tmpl-publish",
            "build": {
                # `branch` is metadata only — build.sh clones templates itself.
                # Setting it to the PR head ref namespaces child builds per-PR
                # so `cancel_running_branch_builds` on tmpl-publish cancels
                # only same-PR predecessors, not cross-PR concurrent publishes.
                "branch": head_ref,
                "message": f"Publish {name} for PR #{pr_number}",
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
    # Verify step re-runs the label check via `gh api` after the block clears.
    # Catches the "maintainer clicks Unblock without applying `go`" bypass —
    # the build goes red instead of silently succeeding with zero children.
    verify_cmd = (
        f"set -euo pipefail\n"
        f"if ! gh api repos/anyscale/templates/issues/{pr_number}/labels "
        f"--jq '.[].name' | grep -qx go; then\n"
        f'  echo "::error::Manual unblock without \'go\' label is not allowed" >&2\n'
        f"  exit 1\n"
        f"fi\n"
        f'echo "go label present; allowing publish"\n'
    )
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
                # GH_TOKEN must be provisioned in the publish-templates pipeline
                # settings (Buildkite agents don't have a GH token by default).
                "command": verify_cmd,
                "agents": {"queue": "small"},
            },
        ],
    }


def main() -> int:
    drifted_raw = _meta("drifted_templates", "[]")
    go_present = _meta("go_present", "false")
    pr_number = _meta("pr_number", "")
    pr_head_sha = _meta("pr_head_sha", "")
    pr_head_ref = _meta("pr_head_ref", "")

    drifted = json.loads(drifted_raw) if drifted_raw else []

    if not drifted:
        pipeline = _empty_pipeline()
    elif go_present == "true":
        pipeline = _trigger_steps(drifted, pr_number, pr_head_sha, pr_head_ref)
    else:
        pipeline = _wait_for_label_pipeline(drifted, pr_number)

    yaml.safe_dump(pipeline, sys.stdout, sort_keys=False, default_flow_style=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
