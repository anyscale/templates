---
name: idle-workspace-sweep
description: >-
  Detect idle GPU waste across an Anyscale cloud (pinned GPU worker groups + GPU head nodes),
  and safely scale ONE named workspace's pinned GPU worker group to zero. Use when asked to
  find idle/expensive workspaces, "what GPUs are we paying for", pinned GPU workers that won't
  scale down, or to scale-to-zero a specific workspace. Detection is read-only and safe
  org-wide; the enforce is gated to the single workspace named in $TEST_WORKSPACE and requires
  a passing min_nodes-only validation before it mutates anything.
---

# idle-workspace-sweep

Two workflows. Detection is read-only and safe to run across a whole cloud. Enforce mutates
exactly one workspace and only after a validation gate passes.

Scripts live in the sibling `../../scripts/` dir (shared with the scheduling recipe):
`idle_sweep.py`, `validate_config_diff.py`, `enforce-scale-to-zero.txt`.

## Preconditions

- `anyscale >= 0.26` (older CLIs have no `workspace_v2 list`). `pip install -U anyscale`.
- Authenticated `anyscale` CLI with visibility into the target cloud.

## Workflow 1 — DETECT (read-only, org-wide)

```bash
python3 ../../scripts/idle_sweep.py <cloud-name>     # e.g. aws-public-us-west-2
```

For every RUNNING workspace it reads the inline compute config and flags:
- **GPU HEAD node** — a head can't scale to zero; recommend a CPU head (flag only, never
  auto-changed).
- **Pinned GPU worker group** — a GPU instance with `min_nodes > 0`: the real idle waste.

Nothing is modified. Verified 2026-07-10 against `aws-public-us-west-2`: scanned 10, 4 running,
correctly flagged `distill-dev`'s `g5.8xlarge` GPU head.

## Workflow 2 — ENFORCE (scale-to-zero, single workspace, gated)

Scoped to an **allowlist-of-one**: only the workspace whose name == `$TEST_WORKSPACE` is ever
touched. Follow `../../scripts/enforce-scale-to-zero.txt` verbatim — it is the procedure. In
short: resolve → build `autozero.json` (live config with `min_nodes: 0` on the GPU group only)
→ **validate** → terminate → update → start → verify.

```bash
export TEST_WORKSPACE='the-one-workspace-name'
# then run the enforce prompt (as a claude -p tick, or drive the steps directly)
python3 ../../scripts/validate_config_diff.py <workspace-id> autozero.json   # MUST print PASS
```

The exact, verified command sequence (and the flags that bit us) is in
[`references/verified-commands.md`](references/verified-commands.md).

## Safety rails (do not remove)

- **Allowlist-of-one.** Mutate only the workspace named in `$TEST_WORKSPACE`. With it unset,
  everything is detect-only.
- **Validation gate is required.** `validate_config_diff.py` must print `PASS` (only `min_nodes`
  differs, matched by worker-group name) before any terminate/update. Any other diff → STOP.
- **Change only `min_nodes` → 0.** Keep max_nodes, instance types, head, image, market, flags.
- **Never terminate/delete another workspace.** `terminate` here stops the target's own cluster
  (the workspace persists, `/mnt` survives) — it is one step of terminate → update → start.

## Not in this skill

Running the sweep on a cadence (the `setsid` loop + Slack delivery) is the scheduling recipe
[`../../recipes/slack-idle-workspace-sweep.md`](../../recipes/slack-idle-workspace-sweep.md),
not this skill. This skill is the core detect + scoped-enforce capability.

## To activate as an invokable skill

Authored here (versioned with the guide). To use it as `/idle-workspace-sweep` in your
sessions, copy or symlink this dir into `~/.claude/skills/`.
