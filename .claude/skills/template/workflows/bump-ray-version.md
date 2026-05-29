# Ray-version bump (non-interactive)

**Audience: the automated `template-updater` Cursor cloud agent.** The trigger supplies the target template `<name>` (one bump + one PR per template). Deterministic, **no prompts** — use defaults. On a missing required input or a blocked precondition, **stop and report** — to the PR if one exists, else to the run log (stdout) — and never guess. A human can run this manually, but it asks no questions.

Cursor environment quirks — `GH_TOKEN=$ANYSCALE_GH_TOKEN` on `gh` writes, `cursor/...` branch naming, `pre-commit` not auto-firing under `core.hooksPath`, and the PR labels — live in **AGENTS.md "Cursor Cloud"**. Follow them there; this runbook does not restate them.

## Setup — preflight

Run `bash .cursor/preflight.sh`. It verifies the companion skills (incl. `/anyscale-platform-fix`), secrets, and auth. On a non-zero exit, handle per **AGENTS.md "Cursor Cloud → Preconditions"** (report stderr + stop) — at this point no PR exists yet, so that report lands in the run/CI output.

## 1. Bump the image

Get the latest Ray version: `pip index versions ray`. Then apply per case (taxonomy: SKILL.md "Image URI cases") — **verify the target tag exists before committing to it.** The anyscale base check:

```bash
curl -sf "https://hub.docker.com/v2/repositories/anyscale/ray/tags/<tag>/" >/dev/null \
  || { echo "anyscale/ray:<tag> not published yet"; exit 1; }
```

- **Anyscale base** — run the check above. Published → set `cluster_env.image_uri` to `anyscale/ray:<new-tag>`. **Not published yet → stop and report** (base images lag the Ray release by a few days); rerun once it lands.
- **Anyscale custom on GCP** — the base is the Dockerfile `FROM` (an `anyscale/ray` tag); verify it with the same check. **Not published yet → stop and report.** Otherwise bump the `FROM` → ensure docker is up (`sudo service docker start`) → run `.claude/skills/template/scripts/push-custom-image-to-gcp.sh <dockerfile-dir> <name> <ray-version>` (use the entry's `name`; the validator requires `<registry>/<name>:<ray-version>`) → update `cluster_env.byod.{docker_image,ray_version}` with the printed URI.
- **Third-party** — query the upstream registry; pick the highest available tag with Ray **≤ requested** (upstream may lag — closest-below is acceptable). Update `cluster_env.byod.{docker_image,ray_version}`. Do **not** swap to `anyscale/ray`. **No tag at or below the requested version → stop and report.**

Then bump `BUILD.yaml` `ray_version` and grep/update any in-template version strings.

## 2. Open the PR

1. Validate locally (`../references/conventions.md` "Validate locally") before committing.
2. Commit: `Update <name> to Ray <version>`.
3. Push to the PR branch (branch naming: AGENTS.md).
4. Open a **draft** PR — title `[ray-update-<version>] Update <name> to Ray <version>`, concise body (outline below). Apply the `ray-update` + `cursor-cloud` labels (AGENTS.md "PR labels").

Body outline (omit **Fix iterations** if `/anyscale-platform-fix` wasn't used):

```markdown
## Summary
Bump <name> to Ray <version>.

## Changes
- <BUILD.yaml image bump, Dockerfile FROM, in-template version strings, etc.>

## Fix iterations
<what was iterated on, if any>

## Tests / validation
- **Local:** `rayapp test <name>` — passed.
- **CI:** Buildkite build #N — <passed | skipped: infra failure + link>.
```

## 3. Validate via CI

Comment `/test-template <name>` on the PR; follow `../references/testing-template.md` for dispatch + Buildkite-MCP monitoring. Green → step 5. Failure → step 4.

## 4. Fix on failure

Triage and recover per `../references/testing-template.md` "Recovery" (agent-fixable → `/anyscale-platform-fix`; infra → don't retry). **No human to hand off to:** post the summary to the PR, leave it open as a draft, and end the run. **Cap the fix loop** — if it isn't green after ~2 fix→test cycles, treat it as infra (summarize + stop) rather than looping. **Ray-bump delta:** when `/anyscale-platform-fix` rebuilds a custom image, re-run `.claude/skills/template/scripts/push-custom-image-to-gcp.sh` and update `cluster_env.byod.{docker_image,ray_version}` before re-pushing, then return to step 3.

## 5. Mark ready + start the publish

When all checks are green, mark the PR ready for review, **start a `tmpl-publish` build** (`../references/publish-to-backend.md`), and **add its Buildkite build link to the PR body** so reviewers can drive it. The build's manual gates govern the actual dev→staging→prod rollout — done after the PR merges, since the pipeline ships templates `main`.
