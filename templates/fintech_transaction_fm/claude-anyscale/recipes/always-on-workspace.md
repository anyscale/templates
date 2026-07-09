# Recipe: "Smoke it on real data + real GPUs first" — the always-on workspace ✅

**The ask:** *"Before I pay for the full job, prove the whole pipeline runs end-to-end on real
GPUs against a sample of the real data — and let me actually look at it."*

**The counterintuitive default:** develop locally, but **keep one workspace always running.** It's
essentially free (idle workspaces scale to zero), and it gives you a **real-hardware smoke rung**
between your laptop and an expensive job — the rung where the plumbing bugs that a CPU laptop smoke
can't reach actually surface.

**Why it's free:** idle = scaled to zero. Proof from my own org's workspace list — several have
been `RUNNING` for *months* and nobody kills them, because idle costs ~nothing:

```
NAME                       STATE     CREATED
ray-demo-v2                RUNNING   2026-06-26
zg-tfm                     RUNNING   2026-06-15
optimization-agent-dev     RUNNING   2026-03-11    ← running since March, and it's fine
```

---

## The three-rung dev loop

```
  LAPTOP  ──edit, git, CPU smoke, screenshots──►  git push
     │                                               │
     ▼                                               ▼
  WORKSPACE (warm)  ──scale up to a few real GPUs, run on a SAMPLE of real
     │               data off /mnt storage──►  the rung where plumbing breaks:
     │               CUDA/OOM, real dtypes, mounted-path reads, checkpoint
     │               writes, health-checks reading columns your new code emits
     │               → inspect in workspace metrics, drop a Ray breakpoint,
     │                 eyeball the TensorBoard it wrote
     ▼
  JOB (scale)  ──same job spec, bigger compute config──►  no surprises left
```

The seamless part: rungs 2 and 3 run the **same entry point and the same job spec**; only the
compute config changes. Your smoke *is* the run, just smaller.

## Run it

```bash
# 1. Is my warm workspace up? (scale-to-zero means "RUNNING" is cheap)
anyscale workspace_v2 status --id expwrk_XXXX        # -> RUNNING

# 2. Push your code, then run the real entry point on a real-data SAMPLE, on the workspace GPUs.
#    Long command -> detach with setsid and poll the logfile, not the process:
anyscale workspace_v2 ssh --id expwrk_XXXX -- '
  cd <repo> && git pull &&
  setsid python scripts/<entrypoint>.py --scale smoke --base-dir /mnt/user_storage/<proj> \
    > /mnt/user_storage/<proj>/smoke.log 2>&1 < /dev/null &
  echo "launched pid $!"'

# 3. Have Claude watch the logfile for a success sentinel or an error, then review TensorBoard.
```

You don't retype the verification every time — it's the same check on every smoke, so it lives in
your [`AGENTS.md`](../README.md#what-claude-needs-to-know-about-your-setup) as a convention:

```
## How we run things
- After a smoke: SMOKE_OK is the success sentinel. Before promoting to a full job, confirm it
  wrote embeddings, checkpoints, and TB events to their expected paths.
```

With that written down once, *"smoke it on the workspace"* is the whole ask — Claude tails the log
to the sentinel (or an error) and tells you whether the plumbing held, because it knows what "held"
means.

## What this rung catches that a laptop can't

Real failures that only appear with real hardware + real data + the real filesystem:

- CUDA OOM / device-placement bugs (no GPU locally)
- real column dtypes and nulls the synthetic smoke didn't have
- reads/writes against the actual mounted storage paths
- the trailing `embedding_health()` / eval-reader that hard-reads a column your new code stopped
  emitting — the classic "the tested core was fine; the glue killed the 40-minute job" failure
- TensorBoard actually landing where you'll look for it

## Turn it into an automation

- **Pre-flight gate:** make "green smoke on the warm workspace" the required step before Claude is
  allowed to `anyscale job submit` the scaled-up run.
- **Same-spec promotion:** *"the smoke passed — submit the identical job at the `gpu-big` compute
  config"* — Claude swaps the compute config and submits; the code is unchanged, so nothing new can
  break at scale except scale itself.
- **Breakpoints:** for a gnarly bug, run it interactively on the workspace and set a Ray breakpoint
  to inspect state live, instead of print-debugging through job logs.
