# Agent guidance for anyscale/templates

Anyscale console templates. For any template-related work (bump Ray, format, publish, debug a test failure), use the `template` skill (`/template`) — canonical entry point for procedures and references.

## Required skills

The `/template` update flow requires companion skills `/ask`, `/fix`, `/run`, `/inspect` (from `anyscale/anyscale-debug-agent`). `/fix` in particular drives the CI iteration loop — **without it you cannot fix a broken template and should not try to do so**. Tip: wrap `/fix` in a subagent to keep its debug output out of your main context.

## Dev lifecycle

~48 production templates (BUILD.yaml entries). No services to run.

Local: edit → `pre-commit run --all-files` → push. Pre-commit covers lint, formatting, codebase conventions. `rayapp build all` mirrors CI's build job locally.

Per-template tests — comment `/test-template <id> [<id>...]` (up to 3, parallel) on the PR to dispatch the Buildkite `template-test` pipeline (workspace + actual test run). For local iteration before pushing, `rayapp test <id>` runs the same flow (see `references/rayapp-local-testing.md` in the `/template` skill for setup).

PR labels (apply all that fit):
- `cursor-cloud` — origin: Cursor Cloud agent.
- `ray-update` — content: Ray version bump.

## Cursor Cloud

The `template-updater` Cursor Cloud agent owns Ray-version bumps end-to-end (open PR → CI → fix-loop) on every major/minor Ray release.

### Setup

Use `.cursor/Dockerfile` and `.cursor/install.sh` as the canonical environment setup. Run `bash .cursor/install.sh`. If anything fails, read the files and reproduce their steps yourself.

### Required Cursor secrets

Already provisioned at team scope:
- `ANYSCALE_DEBUG_AGENT_GH_TOKEN` — skills clone + `gh` write fallback on this repo (see quirks below).
- `ANYSCALE_CLI_TOKEN`
- `GCP_TEMPLATE_REGISTRY_SA_KEY`
- `BUILDKITE_API_TOKEN`

### Cursor-specific quirks

- **Branch naming:** Cursor auto-assigns `cursor/...`. (Outside Cursor, use `update/<template-name>/ray-<version>`.)
- **`pre-commit install` doesn't auto-fire:** Cursor sets `core.hooksPath`, which causes pre-commit to skip its hook install. Run `pre-commit run --all-files` manually before committing.
- **GitHub write operations:** Cursor's default GitHub App auth can't write to this repo. **Always prefix `gh` write commands** (`gh pr create`, `gh pr edit`, `gh pr comment`, `gh issue comment`, `gh pr review`) with `GH_TOKEN=$ANYSCALE_DEBUG_AGENT_GH_TOKEN`. Read-only `gh` calls work without the prefix.
