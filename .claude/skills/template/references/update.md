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

**IMPORTANT: Verify the new image tag exists to catch errors early**:

```bash
curl -sf "https://hub.docker.com/v2/repositories/anyscale/ray/tags/<tag>/" >/dev/null \
  || { echo "anyscale/ray:<tag> does not exist"; exit 1; }
```

For third-party images, use the upstream registry's equivalent.

Apply the bump per case (taxonomy in SKILL.md "Image URI cases"):

- **Anyscale base:** set `image_uri` to `anyscale/ray:<new-tag>`.
- **Anyscale custom on GCP:** bump Dockerfile `FROM` → ensure the docker daemon is up (`sudo service docker start`) → run `.claude/skills/template/scripts/publish-custom-image.sh <dockerfile-dir> <name> <ray-version>` to publish to GCP (use the entry's `name` field as `<image-name>` — validator requires `<registry>/<name>:<ray-version>`) → update `byod.docker_image` and `ray_version` in `BUILD.yaml`.
- **Third-party:** same repo, pick the tag with the highest Ray version **≤ the requested version** (upstream may lag — using the closest version below the request is acceptable). Update `byod.docker_image` and `ray_version` to that tag. Don't swap to `anyscale/ray`.

Grep and update any remaining version strings in template content.

## Step 2: Open PR

All `gh` write commands below need `GH_TOKEN=$ANYSCALE_DEBUG_AGENT_GH_TOKEN` (Cursor's default auth can't write to this repo — see AGENTS.md "GitHub write operations").

1. Commit: `Update <template-name> to Ray <version>`
2. Push the commit. In Cursor Cloud you're already on a `cursor/...` branch — push to that. Outside Cursor, use `update/<template-name>/ray-<version>`.
3. Open the PR (Draft) with title `[ray-update-<version>] Update <template-name> to Ray <version>` and body following the outline below. `GH_TOKEN=$ANYSCALE_DEBUG_AGENT_GH_TOKEN gh pr create --base main --title '...' --body-file <body.md> --draft` (or use `--body "$(cat <<'EOF' ... EOF)"`).
4. Apply both labels: `GH_TOKEN=$ANYSCALE_DEBUG_AGENT_GH_TOKEN gh pr edit --add-label ray-update --add-label cursor-cloud` (per AGENTS.md "PR labels").

**Keep the PR body concise — short bullets, no prose, no boilerplate.** Outline (omit **Fix iterations** if `/fix` wasn't invoked):

```markdown
## Summary
Bump <template-name> to Ray <ray-version>.

## Changes
- <free-form bullets: BUILD.yaml entry bump, in-template version strings, Dockerfile FROM, etc.>

## Fix iterations
<short summary of what was iterated on>. Full notes: `.claude/.artifacts/<template-name>/update-ray-<version>/notes-session-<timestamp-epoch>.md`.

## Tests / validation
- **Local:** `rayapp test <template-name>` — passed.
- **CI:** Buildkite build #N — <passed | skipped due to infra failure: <reason + link>>.
```

## Step 3: Validate via CI

`GH_TOKEN=$ANYSCALE_DEBUG_AGENT_GH_TOKEN gh pr comment <pr-number> --body '/test-template <template-id>'`.

The `/test-template` GitHub Action only dispatches a Buildkite job (`template-test` pipeline). **Monitor the Buildkite build via the Buildkite MCP** — that's where the workspace creation, image pull, and test logs live. `gh pr checks` only shows the dispatch step.

- **PASSED** (Buildkite build green) → Step 5
- **FAILED** → Step 4 (read the Buildkite logs to classify)

## Step 4: Fix

Classify the CI failure first (read the Buildkite logs via the MCP — `gh pr checks` won't show the test failure, only the dispatch):

- **Agent-fixable** — template code/notebook/Dockerfile/config bug, BUILD.yaml schema error, image build error. Spawn `/fix` and iterate (below).
- **Infrastructure** — workspace creation timeout, Anyscale staging API errors, auth/SSO errors, Buildkite/GitHub Actions runner errors. **Don't retry.** Run the local test (`rayapp test <template-name>`, see `rayapp-local-testing.md`); if it passes, trust the local result, post a PR comment summarizing the CI infra failure (`GH_TOKEN=$ANYSCALE_DEBUG_AGENT_GH_TOKEN gh pr comment <pr-number> --body '...'`), and hand off to a human. The PR stays open.

For agent-fixable failures, spawn `/fix` subagent (explicitly authorized):

```
/fix templates/<template-name>

CI failed on PR #<num>. <paste relevant CI error>

Fix minimally — code, notebooks, Dockerfiles, configs...
Iterate until `rayapp test <template-name>` passes locally (see rayapp-local-testing.md). For custom images, rebuild as needed (See `anyscale image build` CLI for fast iteration) and report the working URI and Dockerfile.
Notes: .claude/.artifacts/<template-name>/update-ray-<version>/notes-fix-<timestamp-epoch>.md
```

After `/fix` returns:
1. If a new image was built, run `.claude/skills/template/scripts/publish-custom-image.sh <dockerfile-dir> <image-name> <ray-version>` to push to GCP; update BUILD.yaml `byod.docker_image` with the printed URI.
2. Follow instructions at `format.md` to normalize.
3. Commit and push fixes to the PR branch.
4. Back to Step 3.

## Step 5: Report

PR description (Step 2) is the canonical session report. Only write an extensive `.claude/.artifacts/<template-name>/update-ray-<version>/notes-session-<timestamp-epoch>.md` if `/fix` was invoked.
