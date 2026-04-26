# Publish Template

Publish a template to dev/staging/predeploy/production via the `anyscale/tmpl-publish` Buildkite pipeline.

Use the Buildkite MCP. `<template-id>` must match the template's `name` in `BUILD.yaml`. If multiple templates are requested, run one independent flow per template in parallel.

1. **Trigger build**: `org_slug=anyscale`, `pipeline_slug=tmpl-publish`, `branch=master`, `commit=HEAD`, `message=<template-id>`. Init runs ~10–60s.

2. **Unblock `input-tmpl-name`** with fields:
   - `tmpl-name=<template-id>`
   - `tmpl-branch=main`
   - `tmpl-commit=HEAD`

3. Wait `build-template` (~2–3 min) AND `test-template` (~5–10 min, up to ~45–60 min for some templates) → passed. **Do not unblock any publish step unless `test-template` is passed.** If `test-template` fails, `retry_job` and wait again — retry at least once before giving up. Then **unblock `block-publish-dev`**.

4. Wait `publish-dev` → passed (~3–5 min). **Unblock `block-publish-staging`**.

5. Wait `publish-staging` → passed (~3–5 min). **Unblock `block-publish-production`**.

6. Wait `publish-production` → passed (~3–5 min).

## Updates

For subsequent publishes of the same template, rebuild an existing build instead of step 1. Find it via `list_builds` filtered by `message=<template-id>`; canonical match is product commit `f2a547b88525845df7cf99636a420a0f5523ad07` + templates `main`. Steps 2–6 still apply.
