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
| `check-dep-delivery.py` | Four checks CI would otherwise miss: `depset-config`, `lock-installed`, `bare-pip`, `pin-style`. System reference: the `/template` skill's `references/dependencies.md`. | pre-commit (`check-dep-delivery`) |

## `test-pipeline/` — `/test-template` PR comment → Buildkite

| Script | What it does | Invoked by |
|---|---|---|
| `parse-test-template-comment.sh` | Parses a `/test-template <id>…` PR comment into a validated, ≤3-name list (checked against `BUILD.yaml`) and emits a GitHub-Actions output. | `.github/workflows/test-template.yaml` |
| `render-template-pipeline.sh` | Renders the Buildkite `template-test` pipeline YAML for the requested templates (`TEMPLATE_NAMES`). | `.buildkite/pipeline.template-test.yaml` |

## `depsets/` — the image freezes templates lock against

Both maintain `dependencies/`, which holds only data: the freezes and `template.depsets.yaml`.

| Script | What it does | Invoked by |
|---|---|---|
| `refresh-image-freezes.py` | Fetches a published image's package list from `docs.anyscale.com/base-images` into `dependencies/images/<image>.freeze.txt`. Whole tracked list by default, or just the images named; a named image that fails is an error, one from the list is a skip. | `.github/workflows/ray-version-prep.yaml`; run by hand to add one image |
| `seed-image-freeze.sh` | Writes a lock's seed — the image freeze plus the previous lock's pins for packages the image doesn't ship — to the lock's own path before uv compiles over it. | `pre_hook` on every entry in `dependencies/template.depsets.yaml` |

## `ray-bump/` — Ray-version bump automation

| Script | What it does | Invoked by |
|---|---|---|
| `depset_versions.py` | Which Ray versions have a freeze of every image in `dependencies/images/tracked-images.txt`, and where each freeze lives. Importable, and the CLI both workflows resolve their target version with — no args prints the newest complete version, `--require <v>` validates one, and either exits non-zero so callers fail closed. | `.github/workflows/ray-bump-fanout.yaml`; `.github/workflows/ray-version-prep.yaml`; imported by the two scripts below and `depsets/refresh-image-freezes.py` |
| `prepare-ray-version.py` | Stages a new Ray version's `build_arg_sets` (the workflow fetches the freezes and opens the PR); exits "needs human" when the image matrix moves. | `.github/workflows/ray-version-prep.yaml` |
| `trigger-cursor-bump.py` | Fans the "Template update" Cursor automation out over maintained `BUILD.yaml` entries (one draft PR per template); previews unless `--execute`. | `.github/workflows/ray-bump-fanout.yaml`; run by hand for a manual fanout (see `AGENTS.md`) |

`test_prepare_ray_version.py` covers `prepare-ray-version.py` and `depset_versions.py`; pre-commit runs it.
