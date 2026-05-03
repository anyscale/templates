# Agent guidance for anyscale/templates

Anyscale console templates. For any template-related work (bump Ray, format, publish, debug a test failure), use the `template` skill (`/template`) — canonical entry point for procedures and references.

For cloud agents, companion skills `/fix`, `/run`, and `/inspect` are loaded at user scope via `.cursor/environment.json` for debugging. Advice: `/fix` is a powerful skill that handles the entire debug loop, you might want to wrap it with a subagent to avoid cluttering your main context too much.

**CI invariant** — `.github/workflows/test-template.yaml` only runs when a PR comment matches `/test-template <template-id> [<template-id>...]` (up to 3, fanned out in parallel). After any push to a PR, comment to trigger or re-trigger validation.

**Known automation** — the `template-updater` Cursor automation owns Ray-version bumps end-to-end (open PR → CI → fix-loop) on every major/minor Ray release. Its PRs use label `ray-update` and branches `update/*/ray-*`.

## For Cursor Cloud Agent setup

Use `.cursor/Dockerfile` and `.cursor/install.sh` as the canonical environment setup. Run `bash .cursor/install.sh` from the repo root verbatim — don't infer or replicate its steps, the script is the source of truth.

Required Cursor secrets (already provisioned at team scope):
- `ANYSCALE_DEBUG_AGENT_GH_TOKEN`
- `ANYSCALE_CLI_TOKEN`
- `GCP_TEMPLATE_REGISTRY_SA_KEY`

If any issues, just read `.cursor/Dockerfile` and `.cursor/install.sh` and reproduce their steps yourself.

## Cursor Cloud specific instructions

- **No services to run.** This is a content repository of ~73 self-contained templates. There are no backend servers, databases, or long-running processes. The dev loop is: edit templates → run `pre-commit run --all-files` → push → CI validates.
- **Lint:** `pre-commit run --all-files` (the `pre-commit install` hook won't activate because Cursor sets `core.hooksPath`; run manually before committing).
- **Build:** `rayapp build all` — builds all templates (exits 0 on success; non-self-closing `<img>` warnings are benign).
- **Validate:** `python3 ci/validate_build_yaml.py --no-network` — schema + path check on `BUILD.yaml`.
- **Depsets:** `bash ./update_deps.sh --check` — verifies dependency lockfiles are current. Requires a `python` symlink (Ubuntu 22.04 only ships `python3`); create with `sudo ln -sf /usr/bin/python3 /usr/bin/python` if missing.
- **PATH:** Always ensure `$HOME/.local/bin` and `$HOME/google-cloud-sdk/bin` are on `PATH` (the install script appends to `~/.bashrc`, but subshells may not source it).
- **Private skills clone** at the end of `install.sh` (`anyscale/anyscale-debug-agent`) may fail if the GitHub token lacks access — this is non-fatal and doesn't affect core dev workflows.
