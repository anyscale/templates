# Agent guidance for anyscale/templates

Anyscale console templates. For any template-related work (bump Ray, format, publish, debug a test failure), use the `template` skill (`/template`) — canonical entry point for procedures and references.

## Required skills

The `/template` update flow requires companion skills `/ask`, `/fix`, `/run`, `/inspect` (from `anyscale/anyscale-debug-agent`). `/fix` in particular drives the CI iteration loop — **without it you cannot fix a broken template and should not try to do so**. Tip: wrap `/fix` in a subagent to keep its debug output out of your main context.

## CI

**Invariant** — `.github/workflows/test-template.yaml` only runs when a PR comment matches `/test-template <template-id> [<template-id>...]` (up to 3, fanned out in parallel). After any push to a PR, comment to trigger or re-trigger validation.

**CI runs on Buildkite, not GitHub.** The GitHub Action above only dispatches the Buildkite `template-test` pipeline; the actual workspace creation, image pull, and test run happen there. To monitor a build or read failure logs, use the **Buildkite MCP** (`mcp__buildkite__*` tools, authenticated via `BUILDKITE_API_TOKEN`). `gh pr checks` only shows the dispatch status, not the test result.

## PR labels

Apply all that fit:
- `cursor-cloud` — **origin marker:** any PR opened by a Cursor Cloud agent.
- `ray-update` — **content marker:** any PR bumping a template's Ray version.

## Quick command reference

~48 production templates (the BUILD.yaml entries). No services to run.

Dev loop: edit → `pre-commit run --all-files` → push → CI validates. The pre-commit hooks cover trailing whitespace, README auto-generation (from `README.ipynb` via `jupyter nbconvert`), and BUILD.yaml schema validation.

Beyond pre-commit, `rayapp build all` builds all templates locally (non-self-closing `<img>` warnings are benign).

**Caveat:** `generate-readme` is non-deterministic across Python/jupyter versions (CI uses 3.9). If pre-commit passes locally but fails CI with "files were modified by this hook", treat as infra failure in `/template`'s triage.

## Cursor Cloud

The `template-updater` Cursor Cloud agent owns Ray-version bumps end-to-end (open PR → CI → fix-loop) on every major/minor Ray release.

### Setup

Use `.cursor/Dockerfile` and `.cursor/install.sh` as the canonical environment setup. `install.sh` handles auth and the skills sideload (cloning `anyscale/anyscale-debug-agent` into `~/.claude/skills/`). Run `bash .cursor/install.sh` from the repo root verbatim — don't infer or replicate its steps, the script is the source of truth. If anything fails, read the files and reproduce their steps yourself.

### Required Cursor secrets

Already provisioned at team scope:
- `ANYSCALE_DEBUG_AGENT_GH_TOKEN` — skills clone + `gh` write fallback on this repo (see quirks below).
- `ANYSCALE_CLI_TOKEN`
- `GCP_TEMPLATE_REGISTRY_SA_KEY`
- `BUILDKITE_API_TOKEN` — used by the Buildkite MCP (see "CI" above).

### Cursor-specific quirks

- **Branch naming:** Cursor auto-assigns `cursor/...`. (Outside Cursor, use `update/<template-name>/ray-<version>`.)
- **`pre-commit install` doesn't auto-fire:** Cursor sets `core.hooksPath`, which causes pre-commit to skip its hook install. Run `pre-commit run --all-files` manually before committing.
- **GitHub write operations:** Cursor's default GitHub App auth can't write to this repo. **Always prefix `gh` write commands** (`gh pr create`, `gh pr edit`, `gh pr comment`, `gh issue comment`, `gh pr review`) with `GH_TOKEN=$ANYSCALE_DEBUG_AGENT_GH_TOKEN`. Read-only `gh` calls work without the prefix.
