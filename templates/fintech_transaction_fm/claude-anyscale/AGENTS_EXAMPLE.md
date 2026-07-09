# AGENTS_EXAMPLE.md — a worked example

A real, filled-in `AGENTS.md` from the transaction foundation-model project — the same campaign
these guides were distilled from. Copy the *shape*; replace the specifics with yours.

> **Why the `_EXAMPLE` suffix:** Claude Code auto-loads any file literally named `AGENTS.md` (or
> `CLAUDE.md`) as standing instructions. This file is renamed so it *doesn't* get ingested — in
> your repo, put the real thing at the root as `AGENTS.md`.

---

```markdown
# AGENTS.md — transaction foundation model (fintech_transaction_fm)

## Anyscale
- Cloud (where jobs + workspaces run): sa-demos `cld_g54aiirwj1s8t9ktgzikqur41k`
  ⚠️ My *default* cloud is `nk-demo` — so `anyscale workspace_v2 ssh -n <name>` will NOT resolve
     this project's workspaces (name lookup is default-cloud-scoped). SSH by `--id`, or pass
     `--cloud cld_g54aiirwj1s8t9ktgzikqur41k`.
- Project: `default_cld_g54aiirwj1s8t9ktgzikqur41k`
- Warm workspace (real-data GPU smokes): transaction-foundation-model
  `expwrk_wnk2e5xtrt3dt53x68ta1w7fwh` — TensorBoard on port 38399 (tunnel from the laptop).
- Compute: 4–8× A10G (24GB). scale→config: seq512=full, seq1024=xl (batch 64),
  seq2048=xxl (batch 16 — 32 OOM'd on 24GB).

## Storage
- BASE=/mnt/user_storage/transaction-fm-v2   (durable; survives cluster teardown)
- Pinned eval: $BASE/raw/full/benchmark.parquet — THE eval rows. Never regenerate casually.
- Per-stage artifacts: $BASE/{tokenized,model,embeddings,downstream}/<scale>
- v1 base /mnt/user_storage/transaction-fm/ kept for comparison (pre-benchmark era).

## How we run things
- Submit a job: commit + push first (jobs pull code from git), then:
    anyscale job submit -f job_baseline.yaml   # the gate  → reproduces 0.9875 / 0.1421
    anyscale job submit -f job_full.yaml        # pretrain + eval → prints the headline table
    anyscale job submit -f job_xl.yaml          # the 1024 / 2048 context act
- Smoke locally: python scripts/run_pretrain.py --scale smoke --base-dir <local>  (CPU, minutes)
- Smoke on GPUs: same entrypoints at --scale smoke on the warm workspace, on a real-data sample.
- After a smoke: verify it wrote model/ embeddings/ downstream/ + TB events under $BASE/<scale>
  before promoting to a full job. (SMOKE_OK is the sentinel in the chained entrypoint.)
- Watch a job: attach a monitor on the prodjob id; echo only on state change; on terminal state
  grep the logs for the metrics (AP / ROC / embed_xgb) and failure signatures. Match EVERY
  terminal state.
- Cost: <fill in A10G $/hr> — price before launching. A full 8×A10G run is ≈ 2h.

## Rules (each one was paid for — don't relearn them)
- NEVER delete artifacts — move aside:
    mv $BASE/$d/<scale> $BASE/$d/<scale>_old_$STAMP   # one $STAMP across all artifact kinds
  Deleting an embeddings dir once cost a $4 / 50-min GPU re-extraction; a rename is free.
- Dump configs/outputs VERBATIM — whole config object, original key names, zero filtering.
- Smoke the ENTRY POINT (the real script e2e on tiny data), not just the unit — failures live in
  the glue (writers, health checks, eval readers), not the tested core.
- Reproduce the reference number through OUR pipeline before believing any lift; bootstrap-CI
  every reported number (few positives → single point estimates are rumors).
- One experiment = one off-by-default flag = one commit whose message names the experiment.

## Gotchas
- `RunConfig.name` reuse auto-resumes the latest checkpoint → a re-run silently trains 0 epochs.
  Use a unique run name per invocation (unless you deliberately want a warm restart).
- macOS local runs segfault at the XGBoost stage (torch+xgboost libomp clash in the Ray driver)
  — run stage 5 in a separate process locally; the cluster is unaffected.
- `iter_torch_batches(dtypes=...)` must cover every column or be None (a partial dict KeyErrors
  mid-train-loop).
- GPU worker groups advertise `resources: {CPU: 0}` so CPU-heavy stages can't scale them up; keep
  ≥1 CPU-capable group (the head is CPU:0 too, and data-plane tasks need CPU:1 somewhere).
- Run the TensorBoard tunnel on the laptop, not inside the workspace.
```

---

*See [`README.md`](README.md#what-claude-needs-to-know-about-your-setup) for what each section is
for, and why a filled-in `AGENTS.md` is the highest-leverage thing you can give Claude on a new
project.*
