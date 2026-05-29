# Publish the template artifact to the backend

Publishes a template's content artifact to the dev → staging → production backends (S3) via the
**`tmpl-publish` Buildkite pipeline**: https://buildkite.com/anyscale/tmpl-publish. The pipeline
lives in `anyscale/product`, is triggered against its `master`, and publishes the content on the
templates repo's `main`.

This is **distinct from `/publish-template`** (the product-gallery registration plugin) — different
job. `<name>` is the template's `name` in `BUILD.yaml`. To publish several templates, run one
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
   `commit=f2a547b88525845df7cf99636a420a0f5523ad07`, `message=<name>`. That commit is the **pinned
   `anyscale/product` pipeline** — every template publish runs against it so builds are reproducible;
   it changes only when the pipeline is updated product-side. (The templates content published is
   still latest `main` — see the `tmpl-*` input fields above.) Init runs ~10–60s.
2. **Unblock `input-tmpl-name`** with the three fields above.
3. Wait `build-template` (~2–3 min) **and** `test-template` (~5–10 min, up to ~45–60 for some) →
   passed. If `test-template` fails, `retry_job` and wait again (retry at least once before giving
   up). Then **unblock `block-publish-dev`**.
4. Wait `publish-dev` → passed (~3–5 min). **Unblock `block-publish-staging`**.
5. Wait `publish-staging` → passed (~3–5 min). **Unblock `block-publish-production`**.
6. Wait `publish-production` → passed (~3–5 min).

**Anyscale auth errors** in `build-template` / `test-template` usually mean a prod/staging env or token mismatch — see `run-tests-locally-with-rayapp.md` "Auth errors?".

## Updates (re-publishing an existing template)

Prefer rebuilding the existing build over a fresh trigger: find it via `list_builds` filtered by
`message=<name>`, then `rebuild_build` — it re-runs against the same pinned product commit (step 1),
keeping publishes reproducible. **If no build matches** (first publish), trigger a fresh build
(step 1). The input fields and gates (above) still apply.
