# Publish the template artifact to the backend

Publishes a template's content artifact to the dev → staging → production backends (S3) via the
**`tmpl-publish` Buildkite pipeline**: https://buildkite.com/anyscale/tmpl-publish. The pipeline
lives in `anyscale/product`, is triggered against its `master`, and publishes the content on the
templates repo's `main`.

This is **distinct from `/register-template`** (product-gallery registration, in anyscale/product's
`console-template-plugin`) — different job. `<name>` is the template's `name` in `BUILD.yaml`. To publish several templates, run one
independent flow per template in parallel.

## Pipeline stages

`init` → **`input-tmpl-name`** (manual input) → `build-template` + `test-template` →
**`block-publish-dev`** → `publish-dev` → **`block-publish-staging`** → `publish-staging` →
**`block-publish-production`** → `publish-production`.

Bold = manual gates you Unblock. **Never unblock a publish step until this pipeline's
`test-template` job is green.** Note: that in-pipeline `test-template` job is *not* the PR-comment
`/test-template` from `testing-template.md` — this one runs as a stage of the publish build itself.

`input-tmpl-name` fields:
- `tmpl-name=<name>`
- `tmpl-branch=main`
- `tmpl-commit=HEAD`

## Run it (Buildkite MCP)

1. **Trigger build:** `org_slug=anyscale`, `pipeline_slug=tmpl-publish`, `branch=master`,
   `message=<name>`. Runs against the latest `tmpl-publish` pipeline on `master`. (The templates
   content published is set by the `tmpl-*` input fields above — latest `main`.) Init runs ~10–60s.
2. **Unblock `input-tmpl-name`** with the three fields above.
3. Wait `build-template` (~2–3 min) **and** `test-template` (~5–10 min, up to ~45–60 for some) →
   passed. On any stage failure, triage per **Failure handling** below. Then **unblock `block-publish-dev`**.
4. Wait `publish-dev` → passed (~3–5 min). **Unblock `block-publish-staging`**.
5. Wait `publish-staging` → passed (~3–5 min). **Unblock `block-publish-production`**.
6. Wait `publish-production` → passed (~3–5 min).

**Anyscale auth errors** in `build-template` / `test-template` usually mean a prod/staging env or token mismatch — see `run-tests-locally-with-rayapp.md` "Auth errors?".

## Failure handling — retry or skip?

On a job reaching a terminal non-passed state at **any** stage, classify before acting — bias to **retry the ambiguous; skip only the clearly template-caused or the reproducible**:

| Class | Fingerprint | Action |
|---|---|---|
| **Transient / infra** | Buildkite API 429/5xx, timeouts, DNS/connection reset; job `broken`, lost agent, runner disconnect; **cluster capacity** — "worker group startup timed out", "insufficient cluster resources", instance/quota-launch timeout, spot reclaim; image-pull backoff / registry 5xx; Anyscale control-plane 5xx or a one-off token-refresh blip; **≥2 templates failing at the same instant with the same error** (shared infra event) | **Retry** |
| **Genuine** | `test-template` assertion / notebook failure that **reproduces on retry**; `build-template` failure from the template's deps or Dockerfile (depset conflict, package not found, real compile error); a publish stage where the **backend rejects** the artifact (manifest/schema invalid, missing field); a persistent auth/config error you can't fix; **any failure identical across the full retry budget** | **Skip + report** |

**Retry budget** per template per stage: ~2 `retry_job` + 1 full re-trigger (a fresh build re-resolves `main` HEAD + the latest pipeline), backoff 30–60s. API/network transients (429/5xx/timeout) → just retry the call; they don't consume the budget. **Cross-template correlation wins** — several templates failing together at one moment is one infra event; retry, don't skip. (In the 2.56.0 fanout, three templates hit `exit -1` at the same instant under 49-way concurrency; a sequential re-run passed all three — capacity, not bugs.)

**Skip = hand off, don't fix.** On a genuine failure, capture the failing job's log tail + build URL, record it, and move on — repairing a broken bump is a human's call (or a `../workflows/bump-ray-version.md` fix loop), out of scope for a publish run.

## Updates (re-publishing an existing template)

An update is just another run: trigger a fresh build (step 1) with the same `message=<name>` and
input fields. Prefer a fresh trigger over `rebuild_build` — it re-resolves `main` HEAD (your latest
merged content) and the current pipeline, whereas a rebuild replays the original build's older commits.

## Publish without the test gate (events / urgent fixes)

For an event template that must ship or be fixed faster than the test pipeline (~5–60 min). `rayapp` treats `test` as optional and the BUILD.yaml validator requires it only under `templates/`, so an entry pointing at `archive/` publishes with no `test-template` stage:

1. Put the template under `archive/` and point its `BUILD.yaml` entry's `dir` + `compute_config` there, with **no `test` field** (`templates/` demands a test; `archive/` is exempt). The move: `../workflows/archive-template.md`.
2. Publish as above — `test-template` has nothing to run, so the fix ships immediately.
3. After the event: restore it to `templates/` **with a test**, or retire it (`../workflows/archive-template.md`).

Use sparingly — it bypasses the test gate. Anything not under event-time pressure stays in `templates/`, tested.
