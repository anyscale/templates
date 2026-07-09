# AGENTS_EXAMPLE.md — a starting template

The `AGENTS.md` shape that's worked for me across Anyscale ML projects, distilled to the parts
that generalize. It fits any "get a baseline running, then scale it out" process — fraud model,
recommender, fine-tune, batch pipeline, whatever. Copy it to your repo root as `AGENTS.md` and
fill in the `<placeholders>`.

> **Why the `_EXAMPLE` suffix:** Claude Code auto-loads any file literally named `AGENTS.md` (or
> `CLAUDE.md`) as standing instructions. This file is renamed so it *doesn't* get ingested — the
> real one goes at your repo root as `AGENTS.md`.

---

```markdown
# AGENTS.md — <project>

## Anyscale
- Cloud (where jobs + workspaces run): <name> `<cld_...>`
  ⚠️ If this is NOT your default cloud, `anyscale workspace_v2 ssh -n <name>` won't resolve this
     project's workspaces (name lookup is default-cloud-scoped). SSH by `--id`, or pass `--cloud`.
- Project: `<prj_...>`
- Warm workspace (for real-data GPU smokes): <name> `<expwrk_...>`. TensorBoard: tunnel from the
  laptop, not from inside the workspace.
- Compute configs: cpu=<name>, gpu-small=<name>, gpu-big=<name>.

## Storage — one persisted base, shared by the workspace AND every job
- BASE=/mnt/user_storage/<project>   (durable; survives cluster teardown, and the SAME path is
  mounted in the warm workspace and in every job — so a workspace smoke and a full job read/write
  the identical artifacts, no copying)
- Pinned eval artifact: $BASE/eval/...   — the frozen inputs every run is scored on. Version it;
  never regenerate casually (a moving eval set makes all your numbers incomparable).
- Per-run outputs: $BASE/<stage>/<run-name>/   (metrics.json, checkpoints, embeddings, TB events)
- Scratch only: /mnt/cluster_storage/   (per-cluster, dies with the cluster — never the source of truth)

## How we run things — one config per rung, each a one-liner to launch
Commit + push first (jobs pull code from git), then submit the rung you want:

    anyscale job submit -f jobs/smoke.yaml      # tiny/sampled data, CPU or 1 GPU, minutes —
                                                #   proves the code runs end-to-end. Says NOTHING about quality.
    anyscale job submit -f jobs/baseline.yaml   # the reference/number to beat — REPRODUCE IT FIRST.
    anyscale job submit -f jobs/full.yaml       # the real scaled-out run — the publishable numbers.

Same entrypoint in all three; only the config differs (data size, compute, epochs). The smoke IS
the run, just smaller — so nothing new can break at scale except scale itself.

## Progress + results — print to the logs, persist to disk
- Every stage prints its metrics to stdout in a greppable form, e.g.  `METRIC -> {"score": 0.87, ...}`
  so progress is tail-able:
      anyscale job logs --id <prodjob_...> | grep -E "METRIC ->|epoch|ERROR|Traceback"
- AND every stage writes those metrics + artifacts to $BASE as JSON/parquet — because log streaming
  truncates silently. The persisted file is the source of truth; the log line is just the heartbeat.
- Hands-free watch: attach a monitor that echoes on state change and, on any terminal state, greps
  the logs for the METRIC lines + failure signatures. Match EVERY terminal state — silence ≠ success.
- Cost: <fill in $/hr per GPU tier> — price a run before launching it.

## Rules (each one was paid for — don't relearn them)
- NEVER delete artifacts — move aside:  `mv $BASE/<dir> $BASE/<dir>_old_<stamp>`  (one stamp per
  rerun, across every artifact kind). A rename is free; a delete can cost a GPU re-run.
- Reproduce the baseline through YOUR pipeline before you believe any improvement over it.
- Bootstrap-CI every reported number; with few positive examples, a single point estimate is a rumor.
- Smoke the ENTRY POINT (the real script e2e on tiny data), not just the unit — failures live in the
  glue (writers, health checks, eval readers), not the tested core.
- Dump configs/outputs VERBATIM — whole config object, original key names, no filtering or renaming.
- One change = one off-by-default flag = one commit whose message names the experiment.

## Gotchas (Anyscale / Ray footguns)
- `RunConfig.name` reuse auto-resumes the latest checkpoint → a re-run can silently train 0 epochs.
  Use a unique run name per invocation (unless you deliberately want to resume).
- `iter_torch_batches(dtypes=...)` must cover every column or be None (a partial dict KeyErrors).
- GPU worker groups: advertise `resources: {CPU: 0}` so CPU-heavy stages can't scale them up; keep
  ≥1 CPU-capable group somewhere.
- Long workspace commands die with the SSH session — `setsid <cmd> > $BASE/x.log 2>&1 < /dev/null &`
  and poll the logfile, not the process.
```

---

*See [`README.md`](README.md#what-claude-needs-to-know-about-your-setup) for what each section is
for, and why a filled-in `AGENTS.md` is the highest-leverage thing you can hand Claude on a new
project.*
