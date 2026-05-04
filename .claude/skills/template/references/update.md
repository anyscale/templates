# Update a template to a new Ray release

Bump an Anyscale console template to a new Ray version. Opens a PR, validates via CI, iterates via `/fix` on failure.

## Tasks

Create at session start:

1. `update-version` — Bump image (+ GCP publish if custom) and any version strings
2. `open-pr` — Commit, push, open PR
3. `validate-ci` — Comment `/test-template <id>` on PR, wait for CI
4. `fix` — If CI fails, spawn `/fix`, push new commit, re-validate (cap 2 retries)
5. `report` — Session report

---

## Step 1: Update version

Get latest Ray version with `pip index versions ray`. Pick the right image case from SKILL.md ("Image URI cases") and apply it to `BUILD.yaml`.

Grep and update any remaining version strings in template code.

## Step 2: Open PR

1. Commit: `Update <template-name> to Ray <version>`
2. Push to `update/<template-name>/ray-<version>`
3. PR title: `[ray-update-<version>] Update <template-name> to Ray <version>`. Body: what changed and why.
4. Apply the `ray-update` label.

## Step 3: Validate via CI

Comment `/test-template <template-id>` on the PR, wait for CI.

- **PASSED** → Step 5
- **FAILED** → Step 4

## Step 4: Fix

Spawn `/fix` subagent (explicitly authorized):

```
/fix templates/<template-name>

CI failed on PR #<num>. <paste relevant CI error>

Fix minimally — code, notebooks, Dockerfiles, configs. Don't touch BUILD.yaml (orchestrator owns it).
Iterate until `tests/<template-name>/tests.sh` passes. For custom images, rebuild as needed (See `anyscale image build` CLI for fast iteration) and report the working URI and Dockerfile.
Notes: .claude/.artifacts/<template-name>/update-ray-<version>/notes-fix-<timestamp-epoch>.md
```

After `/fix` returns:
1. If a new image was built, run `.claude/skills/template/scripts/publish-custom-image.sh <dockerfile-dir> <image-name> <ray-version>` to push to GCP; update BUILD.yaml `byod.docker_image` with the printed URI.
2. Follow instructions at `format.md` to normalize.
3. Commit and push fixes to the PR branch.
4. Back to Step 3.

Cap: 2 CI retries, then stop and report what was tried.

## Step 5: Report

Write to `.claude/.artifacts/<template-name>/update-ray-<version>/notes-session-<timestamp-epoch>.md`: what changed, issues, fixes.
