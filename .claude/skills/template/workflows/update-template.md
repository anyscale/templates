# Update a template

Interactive flow for changing an existing template's content, config, or tests. **Not** a Ray-version bump — that's `bump-ray-version.md`. The template is already in the gallery, so there is **no product PR** — you just re-publish the artifact via `tmpl-publish`. (Exception: changing gallery metadata — title, description, labels, icon — is a product-side `workspace-templates.yaml` change instead; use `/register-template`.)

## 1. Confirm scope

Ask which template (`<name>`) and what's changing — notebook/code, compute config, test, or dependencies. Keep the change minimal and focused.

## 2. Make the change

Edit files under `templates/<name>/` (and `configs/<name>/` or `tests/<name>/` as needed). Author README content in `README.ipynb` (README convention in `../references/conventions.md`). If the change touches **dependencies** (`requirements.txt`), recompile and commit the `python_depset.lock` per `../references/depsets.md` — `check-depsets` fails on a stale lock and nothing local catches it. If the change touches the image: identify the case in SKILL.md "Image URI cases" (for custom-GCP, rebuild and push via `.claude/skills/template/scripts/push-custom-image-to-gcp.sh`). For a Ray-**version** bump specifically, use `bump-ray-version.md` — it owns the full per-case bump procedure.

## 3. Format

Apply `../references/conventions.md`. On updates, pay extra attention to **URLs alive** — links rot over time (check described in `../references/conventions.md`).

## 4. Validate

Commit on a branch and open a PR against `main`. Run `/test-template`, get it green. Dispatch, monitoring, and failure recovery: `../references/testing-template.md`.

## 5. Republish

**Merge the green PR to `main` first** (the pipeline publishes templates `main`), then re-publish the artifact (dev → staging → prod) via `../references/publish-to-backend.md`. (Urgent event fix that can't wait on tests? Re-publish *test-free* via an `archive/` entry — see "Publish without the test gate" there.)
