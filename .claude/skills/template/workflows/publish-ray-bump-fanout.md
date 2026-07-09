# Publish a Ray-bump fanout (agent team)

**Audience: you are the team leader.** A human hands you a target Ray `<version>` after a bump fanout (per-template PRs, label `ray-update`). You review and land those bumps, then fan out **worker subagents** — ≤8 templates each — that drive every merged template through `tmpl-publish` to prod. Workers publish; you coordinate and report. Interactive at the edges (confirm scope, hand off failures), autonomous in between.

This runs the per-template flow in `../references/publish-to-backend.md` at fleet scale — it does not restate the pipeline mechanics or its **Failure handling** rubric (retry vs. skip).

## Roles & concurrency

- **Leader (you):** discover → review + land → fan out → aggregate. Owns all merges and the human hand-off list.
- **Worker** (subagent, `general-purpose`; Bash + Read + Buildkite MCP): one batch of ≤8 templates, published **sequentially**.
- **Concurrency = parallel across workers, sequential within a worker.** Keep total in-flight builds at ~8–10. More has driven `test-template` to fail on cluster-capacity contention (worker-group startup timeouts) — spurious failures that read like template bugs. 54 templates → 7 workers → ≤7 concurrent builds, safely under the line.

## 1. Discover & scope

1. Inputs: target Ray `<version>` (required); optional explicit template list (default: every template bumped to `<version>` this release).
2. `gh pr list --label ray-update --state open --search "<version>"` → the bumps to review + land. Add any already-merged `<version>` bumps the human named (the straggler case: publish-only, they skip step 2).
3. Present the vetted plan — N to merge, M to publish, anomalies held — and get **one go-ahead** before landing or publishing. Prod is outward-facing; confirm once, then run autonomously.

## 2. Review & land (leader, sequential)

Per open bump PR:
- **Triage-review the diff.** A clean bump touches only: the image (`cluster_env.image_uri` tag / BYOD `docker_image` + `ray_version`, or `Dockerfile FROM`), `templates/<name>/python_depset.lock`, `BUILD.yaml` `ray_version`, and in-template version strings — target version correct, required checks green.
- **Anomaly** — a stray file, an unexpected edit, red CI, or a diff that doesn't read as a pure bump → **hold for the human**; never publish a bump you couldn't vet.
- **Clean** → squash-merge **and delete the branch** (`gh pr merge --squash --delete-branch`, to keep the remote uncluttered). **Space the merges (~5s) and back off on any write error** — bulk approve+merge trips GitHub's secondary rate limit.

The vetted templates are now on `main` (the pipeline publishes `main` HEAD).

## 3. Fan out

1. Partition the vetted templates into batches of ≤8; create a scratch results dir and note its path.
2. Spawn `ceil(n/8)` workers with the Agent tool (`general-purpose`, background), each given: its batch, `<version>`, the results-dir path, and the **Worker contract** below.
3. Poll the results dir for progress while they run.

## 4. Worker contract

Hand each worker this. It first loads the Buildkite MCP tools (deferred — `ToolSearch` "buildkite"; authenticate if prompted) and reads `../references/publish-to-backend.md` (incl. its **Failure handling** section). Then, for **each** template in its batch, **sequentially**:

1. **Pre-publish review** — confirm the merged bump actually landed (`git show origin/main -- templates/<name> BUILD.yaml`): image tag → `<version>`, lock recompiled if the template ships one. Wrong → record `SKIPPED:review:<why>`, next. Never publish a bad artifact.
2. **Publish** per `publish-to-backend.md`: trigger → unblock `input-tmpl-name` (fields) → **wait `build-template` + `test-template` pass** → dev → staging → prod. Invariant: **never unblock a publish gate before `test-template` is green.**
3. **On any failure, any stage** → classify per **Failure handling** in `publish-to-backend.md`: transient/infra → bounded retry; genuine → capture the failing job's log tail + build URL, record `SKIPPED:<stage>:<reason>`, next template.
4. **Record** one line per template to `<results-dir>/<name>.json` as it goes — `{status: PUBLISHED|SKIPPED, stage, reason, build_url}`. On entry, skip any template already `PUBLISHED` (idempotent resume after a kill).
5. **Return** a compact per-template summary to the leader.

Sharp edge: unblocking `input-tmpl-name` needs a *fields* payload; if the MCP can't send unblock fields, fall back to a REST `curl` (`PUT .../jobs/{id}/unblock` with `BUILDKITE_API_TOKEN`) for that one call.

## 5. Aggregate & report

Collect the workers' summaries (and the results dir):
- ✅ **Published** (n)
- ⏭️ **Skipped — needs human**: template · stage · reason · log link
- 🔍 **Held at review**: template · anomaly (step 2)
- Totals + wall-clock.

The skipped + held lists are the human hand-off. Re-invoking is safe — already-`PUBLISHED` templates are skipped.
