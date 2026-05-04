# Agent guidance for anyscale/templates

Anyscale console templates. For any template-related work (bump Ray, format, publish, debug a test failure), use the `template` skill (`/template`) — canonical entry point for procedures and references.

For cloud agents, companion skills `/fix`, `/run`, and `/inspect` are loaded at user scope via `.cursor/environment.json` for debugging. Advice: `/fix` is a powerful skill that handles the entire debug loop, you might want to wrap it with a subagent to avoid cluttering your main context too much.

**CI invariant** — `.github/workflows/test-template.yaml` only runs when a PR comment matches `/test-template <template-id>`. After any push to a PR, comment to trigger or re-trigger validation.

**Known automation** — the `template-updater` Cursor automation owns Ray-version bumps end-to-end (open PR → CI → fix-loop) on every major/minor Ray release. Its PRs use label `ray-update` and branches `update/*/ray-*`.
