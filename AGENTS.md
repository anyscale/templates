# Agent guidance for anyscale/templates

Anyscale console templates. For any template-related work (bump Ray, format, publish, debug a test failure), use the `template` skill (`/template`) — canonical entry point for procedures and references.

Cloud agents **require** companion skills `/ask`, `/fix`, `/run`, `/inspect` — sideloaded by `.cursor/install.sh` from `anyscale/anyscale-debug-agent` into `~/.claude/skills/`. These are not optional: the `/template` update flow uses `/fix` to iterate on CI failures and cannot complete without it. If `~/.claude/skills/` is empty after install.sh, the `ANYSCALE_DEBUG_AGENT_GH_TOKEN` token lacks access — fix the token before retrying. Tip: wrap `/fix` in a subagent to keep its debug output out of your main context.

**CI invariant** — `.github/workflows/test-template.yaml` only runs when a PR comment matches `/test-template <template-id> [<template-id>...]` (up to 3, fanned out in parallel). After any push to a PR, comment to trigger or re-trigger validation.

**Known automation** — the `template-updater` Cursor automation owns Ray-version bumps end-to-end (open PR → CI → fix-loop) on every major/minor Ray release. Its PRs use label `ray-update` and branches `update/*/ray-*`.

## For Cursor Cloud Agent setup

Use `.cursor/Dockerfile` and `.cursor/install.sh` as the canonical environment setup. Run `bash .cursor/install.sh` from the repo root verbatim — don't infer or replicate its steps, the script is the source of truth.

Required Cursor secrets (already provisioned at team scope):
- `ANYSCALE_DEBUG_AGENT_GH_TOKEN`
- `ANYSCALE_CLI_TOKEN`
- `GCP_TEMPLATE_REGISTRY_SA_KEY`

If any issues, just read `.cursor/Dockerfile` and `.cursor/install.sh` and reproduce their steps yourself.

## Cursor Cloud quick reference

- **No services to run.** This is a content repository of ~73 self-contained templates. There are no backend servers, databases, or long-running processes. Dev loop: edit → `pre-commit run --all-files` → push → CI validates.
- **Lint:** `pre-commit run --all-files`. The `pre-commit install` git hook won't activate because Cursor sets `core.hooksPath`; run manually before committing.
- **Build all templates:** `rayapp build all` (non-self-closing `<img>` warnings are benign).
- **Validate `BUILD.yaml`:** `python3 ci/validate_build_yaml.py --no-network` — schema + path check, mirrors the pre-commit hook.
- **Depsets:** `bash ./update_deps.sh --check` — verifies the dependency lockfile is current.
- **Skills sideload** (`anyscale/anyscale-debug-agent` clone at the end of `install.sh`) is required for the `/template` update flow; if it fails, `~/.claude/skills/` will be empty and `/fix` won't be available. Other dev workflows can proceed without it.
