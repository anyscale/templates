# Using Claude on Anyscale — a field guide

*Geoff Counihan · how I've come to use Claude Code with Anyscale — the practices that have
worked best for me so far, mostly worked out while iterating on the transaction foundation-model
project. Everyday stuff, not research: what Claude can do, what it needs from you, and the
recipes I keep reaching for.*

> For the deeper "run a rigorous research campaign" version, see the companion
> [`CLAUDE_WITH_ANYSCALE.md`](CLAUDE_WITH_ANYSCALE.md) (my PI operating manual) and its
> [`AUTORESEARCH.md`](AUTORESEARCH.md). This guide is the on-ramp; those are the deep end.

---

## How I've been using it

Over the transaction-FM iteration a lot of my week went to Anyscale toil that isn't the
interesting part of the work: babysitting a training run, digging through logs to find why a job
died, checking whether I'd left a cluster running, hand-writing job YAML, smoke-testing plumbing
before a big run. Claude Code — running in my terminal, driving the `anyscale` CLI — turned out
to handle almost all of it, and (the part I underused at first) it can **keep watching things
while I do something else**. What follows is what's stuck: the mental model and the handful of
recipes I keep coming back to. It's what's working for me right now, not the last word.

**The habit that's paid off most for me:** tell Claude about your setup *once*, in a repo
[`AGENTS.md`](#what-claude-needs-to-know-about-your-setup), so you stop re-explaining your cluster
names, storage layout, and "how we submit jobs here" every session.

---

## Mental model — what Claude Code is here

Claude Code is a terminal-native agent. On Anyscale that means it can:

- **drive the `anyscale` CLI** — list/submit/inspect/terminate jobs, services, and workspaces;
- **read logs and artifacts** — job logs, Ray state, files on mounted storage;
- **edit configs and code** — job/compute/service YAML, training scripts, `requirements.txt`;
- **run git** — commit and push so a job picks up your code;
- **watch long-running things** — poll a job to completion and wake itself (and you) when the
  state changes or it fails. This is the part people underuse.

It works across all three Anyscale workload types:

| workload | what it is | Claude drives it with |
|---|---|---|
| **Workspace** | interactive cloud dev box with a cluster attached | `anyscale workspace_v2 ...` (+ SSH) |
| **Job** | batch / production run on an ephemeral cluster | `anyscale job submit -f <yaml>` |
| **Service** | online serving endpoint | `anyscale service ...` |

---

## The two surfaces: your laptop vs. the workspace

You work across two places, and it's worth being deliberate about which does what.

| | **Laptop** | **Workspace** (cloud) |
|---|---|---|
| best for | code, git, the fast edit loop, **screenshots** (see below), multi-repo work, cheap CPU smokes | real GPUs, real data on mounted storage (`/mnt/user_storage`, `/mnt/cluster_storage`, `/mnt/shared_storage`), live Ray debugging, TensorBoard |
| has | your full local env, the `anyscale` CLI | the cluster, the mounts, the hardware |
| doesn't | GPUs, cluster-local data | your local tooling / screenshot flow |

**How Claude bridges them:** either (a) edit locally → `git commit && git push` → the job pulls
your code from git, or (b) SSH into the workspace and run there. Long workspace commands die with
the SSH session, so for anything slow: `setsid <cmd> > /mnt/user_storage/x.log 2>&1 < /dev/null &`
and poll the **logfile**, not the process. (Run a TensorBoard tunnel *on your laptop*, not inside
the workspace.)

**Where should Claude live?** Run it on the laptop for the edit loop + screenshots; run it in the
workspace terminal when the work is data-heavy or GPU-adjacent and you want zero round-trips. Most
days: laptop, with a warm workspace standing by (next section).

---

## Keep a workspace warm — your real-hardware smoke rung

This is the counterintuitive one. **Develop locally, but keep one workspace always running.**

It costs almost nothing — idle workspaces scale down to zero, so a "running" workspace you're not
actively using is basically free. (Proof from our own org: when I asked Claude to list workspaces,
several had been `RUNNING` since *March* — nobody kills them because idle is free.)

What that warm workspace buys you is a **real-hardware smoke rung** between your laptop and an
expensive job:

- Scale it up to a few **real GPUs** and run your pipeline on a **sample of real data artifacts**
  straight off the mounted storage.
- That's where the plumbing bugs actually live — CUDA/OOM, real dtypes, the mounted-path reads,
  the checkpoint writes, the health-check that reads a column your new code doesn't emit. A laptop
  CPU smoke can't catch those.
- You inspect the smoke run right there in the **workspace metrics**, drop a **Ray breakpoint** if
  you want to poke at it live, and eyeball the **TensorBoard** it wrote.
- Then the seamless part: it's the **same job spec** — you submit it again pointed at a bigger
  compute config. Laptop edit loop → warm workspace e2e-on-real-hardware → full job at scale.
  One artifact, three rungs, no surprises left for the run you pay for.

Full recipe: [`recipes/always-on-workspace.md`](recipes/always-on-workspace.md).

---

## What Claude needs to know about your setup

Claude is only as good as the context you give it. The things that make it effective on Anyscale:

- **Which cloud + project.** (Gotcha: `anyscale workspace_v2 ssh -n <name>` resolves the name in
  your *default* cloud/project, not the whole org — if the workspace lives elsewhere it'll say
  "not found" even though `list` shows it. Fix: SSH by `--id`, or pass `--cloud`.)
- **Auth state** — `anyscale` logged in, staging vs. prod token, and cloud creds if you touch
  object storage.
- **Compute/cluster config names**, the base image, and that jobs need a `working_dir`.
- **Storage layout** — what's durable vs. ephemeral, per-user vs. shared, and where your artifacts
  live.
- **Cost model** — GPU types and rough $/hr, so Claude prices a run *before* launching it.

**The punchline: put all of it in an `AGENTS.md` (or `CLAUDE.md`) at your repo root, once.** Claude
reads it automatically every session, so you stop re-explaining your environment. A starter you can
fill in:

```markdown
# AGENTS.md — <your project>

## Anyscale
- Cloud: <name> (<cld_...>)   Project: <name> (<prj_...>)
- Compute configs: cpu=<name>, gpu-small=<name>, gpu-big=<name>
- Base image: <image>
- Warm workspace: <name> (<expwrk_...>) — scale up for real-data GPU smokes

## Storage
- Durable artifacts: /mnt/user_storage/<project>/  (survives cluster teardown)
- Scratch: /mnt/cluster_storage/  (per-cluster, ephemeral)
- Object storage: s3://<bucket>/... | gs://<bucket>/...

## How we run things
- Submit a job:   anyscale job submit -f jobs/<x>.yaml   (commit + push first — jobs pull from git)
- Smoke locally:  python scripts/<entrypoint>.py --scale smoke
- Smoke on GPUs:  run the same entrypoint on the warm workspace at --scale smoke
- After a smoke:  SMOKE_OK is the success sentinel. Before promoting to a full job, confirm it
  wrote embeddings, checkpoints, and TB events to their expected paths.
- Cost: A10G ≈ $X/hr, A100 ≈ $Y/hr — price runs before launching

## Gotchas
- <e.g. RunConfig.name reuse auto-resumes the last checkpoint — use unique names>
- <e.g. macOS local runs segfault at the XGBoost stage — run that stage in a separate process>
```

**Rule of thumb: if you catch yourself typing the same ask twice, it belongs in here.** A prompt
you retype is a convention you haven't written down yet — move it into `AGENTS.md` and "smoke it on
the workspace" starts meaning the whole check, not just the run.

---

## Where the information lives

Know what you get from where — and use screenshots to bridge the gap:

- **CLI** — `anyscale job status/logs`, `anyscale service status`, `anyscale workspace_v2 ...`;
  on the cluster, `ray status / ray list / ray logs / ray summary`.
- **Persist results as files, read the files.** Job-log *streaming* truncates silently — have
  each stage write its metrics JSON/parquet to durable storage and read that, don't scrape stdout.
- **Screenshots → Claude.** A lot of the useful signal only exists in a web UI. Paste it in:
  the **Ray dashboard**, a **Grafana GPU-utilization graph**, a **console stack trace / OOM
  modal**, a **TensorBoard curve**, an **autoscaling graph**. Claude reads the image and diagnoses.
  This is the fastest way to answer "why is my GPU idle?" or "what is this error?"

---

## The recipe box

These are the ones that have stuck for me so far — I add to them as I find new ones. Each is a
plain-English ask → what Claude does. ✅ = I've actually run it against a real Anyscale org.

| # | Ask | Recipe |
|---|---|---|
| 1 | "What's running, and what's it costing me?" | ✅ [`whats-running.md`](recipes/whats-running.md) |
| 2 | "Watch this job and ping me when it's done." | ✅ [`night-watch.md`](recipes/night-watch.md) |
| 3 | "Smoke it on real data + real GPUs first." | ✅ [`always-on-workspace.md`](recipes/always-on-workspace.md) |
| 4 | "This job failed — what happened, and fix it." | _coming — `/anyscale-platform-fix`_ |
| 5 | "Pre-flight this before I burn a cluster." | _coming — local smoke + YAML lint + cost estimate_ |
| 6 | "Why is my GPU stuck at 40%?" | _coming — screenshot → `ray-train-bottleneck-finder`_ |
| 7 | "Deploy this checkpoint as a service and hit it." | _coming — `/anyscale-platform-run`_ |
| 8 | "Every morning, tell me what happened overnight." | _coming — scheduled cloud agent → Slack_ |
| 9 | "Turn this into a job/compute config." | _coming — `/anyscale-platform-run`_ |
| 10 | "Bump Ray across all our repos." | _coming — git-worktree fan-out + `template` skill_ |

*Reserve ideas: tear down idle clusters · auto-resume preempted spot jobs · diff two runs' metrics ·
generate a runbook from a debug session · `/anyscale-platform-ask` for "how do I do X on Ray?" ·
`/anyscale-platform-inspect` an unfamiliar workspace.*

---

## Skills catalog — reach for X when Y

Installed via `anyscale skills install -p claude-code`:

| skill | when to reach for it |
|---|---|
| `/anyscale-platform-ask` | "how do I do X on Ray/Anyscale?"; architecture recommendations (read-only) |
| `/anyscale-platform-inspect` | static-validate a local dir; inspect a live workspace/job/service |
| `/anyscale-platform-run` | generate + execute a workspace/job/service config from a description |
| `/anyscale-platform-fix` | debug + fix a failing workload (the automated-debugging loop) |
| `anyscale-workload-ray-train` / `-ray-data` / `-ray-serve` / `-batch-embedding` / `-llm-serving` | design/optimize/productionize a specific workload type |
| `ray-train-bottleneck-finder` | GPU idle / dataloader tuning — emits a topology diagram |
| `/template` | maintain/format/publish console templates (this repo) |
| superpowers `executing-plans` / `systematic-debugging` / `using-git-worktrees` | run a written plan; debug methodically; isolate parallel work |

Tip: wrap a verbose skill (like `/anyscale-platform-fix`) **in a subagent** so its debug output
stays out of your main context — you get back just the diagnosis.

---

## Guardrails

The recipes above are still evolving — but these are the lines I've learned not to cross, and I
don't expect *these* to change much:

- **Cost** — always price a GPU run before launching; tear down idle clusters; know that a warm
  workspace is fine (scales to zero) but a running *service* or *job cluster* is live spend.
- **Irreversibility** — never delete artifacts, *move them aside* (`<name>_old_<stamp>`); confirm
  before terminating a job/service someone else depends on; keep **staging vs. prod tokens**
  straight.
- **Permissions** — allow-list read-only `anyscale`/`ray` *status* commands to kill prompt fatigue,
  and gate `submit`/`terminate`/`delete`. (`/fewer-permission-prompts` sets this up.)
- **Secrets** — never paste tokens into prompts; use env vars.

---

## Getting started in an afternoon

Roughly the setup that got me here — adjust to taste:

```
[ ] Install Claude Code.
[ ] anyscale skills install -p claude-code   (gets the /anyscale-platform-* skills)
[ ] anyscale login   (+ cloud creds if you touch object storage)
[ ] Drop the AGENTS.md template above into your repo root and fill it in.
[ ] Allow-list your common read-only commands (/fewer-permission-prompts).
[ ] Run recipe #1 ("what's running, and what's it costing me?") to confirm it's all wired up.
[ ] Optional: connect Slack via MCP so Claude can post job summaries / alerts.
```
