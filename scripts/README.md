# `scripts/` — repo automation scripts

Helper scripts for this repo's automation, grouped by concern. The `hooks/` scripts run
**locally** on `git commit` (via pre-commit) as well as in CI.

Scripts that need the repo root find it by walking up to the directory containing `BUILD.yaml`,
so they work from any working directory and are safe to relocate. The exceptions are
`parse-test-template-comment.sh` and `check-depsets.py`, which operate on the current directory
(the CI runner's checkout root).

## `hooks/` — content-quality gates (pre-commit + premerge CI)

| Script | What it does | Invoked by |
|---|---|---|
| `clear-notebook-outputs.py` | Strips outputs + `execution_count` from `*.ipynb` (**mutating** — re-stage after). Avoids diff churn and secret leaks. | pre-commit (`clear-notebook-outputs`) |
| `check-image-urls.py` | Fails if an image ref in `*.ipynb`/`*.md` is relative — relative URLs break the console gallery. | pre-commit (`check-image-urls`) |
| `check-readme.sh` | Verifies `README.md` byte-matches `nbconvert` of `README.ipynb` (never mutates). | pre-commit (`check-readme`) |
| `validate-build-yaml.py` | Validates the `BUILD.yaml` schema and that referenced paths exist (`--no-network` runs it offline). | pre-commit (`check-build-yaml`); `.github/workflows/premerge.yaml` |
| `check-depsets.py` | Verifies a PR's dependency locks are current — scoped to the changed templates (skip / scoped / full), retrying transient index errors. | `.github/workflows/premerge.yaml` (CI only) |

## `test-pipeline/` — `/test-template` PR comment → Buildkite

| Script | What it does | Invoked by |
|---|---|---|
| `parse-test-template-comment.sh` | Parses a `/test-template <id>…` PR comment into a validated, ≤3-name list (checked against `BUILD.yaml`) and emits a GitHub-Actions output. | `.github/workflows/test-template.yaml` |
| `render-template-pipeline.sh` | Renders the Buildkite `template-test` pipeline YAML for the requested templates (`TEMPLATE_NAMES`). | `.buildkite/pipeline.template-test.yaml` |

## `ray-bump/` — Ray-version bump automation

| Script | What it does | Invoked by |
|---|---|---|
| `latest-depset-version.py` | Resolves the newest "complete" Ray version from `dependencies/depsets/` (both base + LLM locks present). | `.github/workflows/ray-bump-fanout.yaml` |
| `prepare-base-locks.py` | Recompiles + stages the base depset locks for a new Ray version (the workflow opens the PR); exits "needs human" on a matrix change. | `.github/workflows/ray-base-locks.yaml` |
| `trigger-cursor-bump.py` | Fans the "Template update" Cursor automation out over maintained `BUILD.yaml` entries (one draft PR per template); previews unless `--execute`. | `.github/workflows/ray-bump-fanout.yaml`; run by hand for a manual fanout (see `AGENTS.md`) |
