# Agent guidance for anyscale/templates

Anyscale console templates. For any template-related work (bump Ray, format, publish, debug a test failure), use the `template` skill (`/template`) — canonical entry point for procedures and references.

## Companion skills

The `/template` update flow leans on companion skills `/ask`, `/fix`, `/run`, `/inspect` (from `anyscale/anyscale-debug-agent`). `/fix` in particular drives the CI iteration loop — without it you cannot reliably diagnose and fix a broken template. Strongly recommended for any update work. Tip: wrap `/fix` in a subagent to keep its debug output out of your main context.

For Cursor Cloud, these skills are a hard precondition (see Cursor Cloud → Preconditions).

## Dev lifecycle

Manage production templates (BUILD.yaml entries). No services to run.

Local: edit → `pre-commit run --all-files` → push. Pre-commit covers lint, formatting, codebase conventions. `rayapp build all` mirrors CI's build job locally.

Per-template tests — comment `/test-template <id> [<id>...]` (up to 3, parallel) on the PR to dispatch the Buildkite `template-test` pipeline (workspace + actual test run). For local iteration before pushing, `rayapp test <id>` runs the same flow (see `references/rayapp-local-testing.md` in the `/template` skill for setup).

PR labels (apply all that fit):
- `cursor-cloud` — origin: Cursor Cloud agent.
- `ray-update` — content: Ray version bump.

## Cursor Cloud

The `template-updater` Cursor Cloud agent owns Ray-version bumps end-to-end (open PR → CI → fix-loop) on every major/minor Ray release.

### Preconditions (HARD EXIT IF MISSING)

Run `bash .cursor/preflight.sh` before any task. It checks:

- **Companion skills** present at `~/.claude/skills/{ask,fix,run,inspect}` (rsync'd by `.cursor/install.sh` from the `anyscale/anyscale-debug-agent` pre-clone Cursor sets up via `.cursor/environment.json` `repositoryDependencies`).
- **Cursor secrets** (team-scope, all three non-empty):
  - `ANYSCALE_CLI_TOKEN`
  - `GCP_TEMPLATE_REGISTRY_SA_KEY`
  - `BUILDKITE_API_TOKEN`
- **Auth verified:** `gh auth status` (Cursor's native GH App auth), `gcloud auth list`, and `anyscale cloud list` all succeed.

**If preflight exits non-zero, post its stderr as a PR comment and stop — don't attempt the task.**

### Setup

Use `.cursor/Dockerfile` and `.cursor/install.sh` as the canonical environment setup. Run `bash .cursor/install.sh`. If anything fails, read the files and reproduce their steps yourself.

### Cursor-specific quirks

- **Branch naming:** Cursor auto-assigns `cursor/...`. (Outside Cursor, use `update/<template-name>/ray-<version>`.)
- **`pre-commit install` doesn't auto-fire:** Cursor sets `core.hooksPath`, which causes pre-commit to skip its hook install. Run `pre-commit run --all-files` manually before committing.
