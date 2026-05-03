# Update a template to a new Ray release

Bump an Anyscale console template to a new Ray version. Opens a PR, validates via CI, iterates via `/fix` on failure.

## Tasks

Create at session start:

1. `update-version` — Bump image (+ GCP publish if custom) and any version strings
2. `open-pr` — Commit, push, open PR
3. `validate-ci` — Comment `/test-template <id>` on PR, wait for CI
4. `fix` — If CI fails, classify and respond — spawn `/fix` for agent-fixable failures, hand off for infrastructure failures
5. `report` — Session report

---

## Step 1: Update version

Get latest Ray version: `pip index versions ray`.

**Verify the new image tag exists** — a non-existent tag fails only after a slow workspace boot:

```bash
curl -sf "https://hub.docker.com/v2/repositories/anyscale/ray/tags/<tag>/" >/dev/null \
  || { echo "anyscale/ray:<tag> does not exist"; exit 1; }
```

For third-party images, use the upstream registry's equivalent.

Apply the bump per case (taxonomy in SKILL.md "Image URI cases"):

- **Anyscale base:** set `image_uri` to `anyscale/ray:<new-tag>`.
- **Anyscale custom on GCP:** bump Dockerfile `FROM` → run `.claude/skills/template/scripts/publish-custom-image.sh <dockerfile-dir> <name> <ray-version>` to publish to GCP (use the entry's `name` field as `<image-name>` — validator requires `<registry>/<name>:<ray-version>`) → update `byod.docker_image` and `ray_version` in `BUILD.yaml`.
- **Third-party:** same repo, pick the latest tag with the highest Ray version, update `byod.docker_image` and `ray_version`. Don't swap to `anyscale/ray`.

Grep and update any remaining version strings in template content.

## Step 2: Open PR

1. Commit: `Update <template-name> to Ray <version>`
2. Push the commit. In Cursor Cloud you're already on a `cursor/...` branch — push to that. Outside Cursor, any branch name works.
3. Open the PR against `main`. Title: `[ray-update-<version>] Update <template-name> to Ray <version>`. Body: what changed and why.
4. Apply the `ray-update` label. If running in Cursor Cloud, also apply `cursor-cloud` (see AGENTS.md "PR labels").

## Step 3: Validate via CI

Comment `/test-template <template-id>` on the PR, wait for CI.

- **PASSED** → Step 5
- **FAILED** → Step 4

## Step 4: Fix

Classify the CI failure first:

- **Agent-fixable** — template code/notebook/Dockerfile/config bug, BUILD.yaml schema error, image build error. Spawn `/fix` and iterate (below).
- **Infrastructure** — workspace creation timeout, Anyscale staging API errors, auth/SSO errors, GitHub Actions runner errors. **Don't retry.** Run the local test (`rayapp test <template-name>`, see `local-testing.md`); if it passes, trust the local result, post a PR comment summarizing the CI infra failure, and hand off to a human. The PR stays open.

For agent-fixable failures, spawn `/fix` subagent (explicitly authorized):

```
/fix templates/<template-name>

CI failed on PR #<num>. <paste relevant CI error>

Fix minimally — code, notebooks, Dockerfiles, configs. Don't touch BUILD.yaml (orchestrator owns it).
Iterate until `rayapp test <template-name>` passes locally (see local-testing.md). For custom images, rebuild as needed (See `anyscale image build` CLI for fast iteration) and report the working URI and Dockerfile.
Notes: .claude/.artifacts/<template-name>/update-ray-<version>/notes-fix-<timestamp-epoch>.md
```

After `/fix` returns:
1. If a new image was built, run `.claude/skills/template/scripts/publish-custom-image.sh <dockerfile-dir> <image-name> <ray-version>` to push to GCP; update BUILD.yaml `byod.docker_image` with the printed URI.
2. Follow instructions at `format.md` to normalize.
3. Commit and push fixes to the PR branch.
4. Back to Step 3.

## Step 5: Report

Write to `.claude/.artifacts/<template-name>/update-ray-<version>/notes-session-<timestamp-epoch>.md`: what changed, issues, fixes.
