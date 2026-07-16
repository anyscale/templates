# harness/ — the autoresearch harness, made executable

The `REQUIREMENTS.md` spec, as running code. Everything here is **dependency-free stdlib
Python** and **tested** — the point is that the rigor rules are enforced by code, not trusted
to a human's memory. Each module turns one prose requirement (or one `critiques.md` fix) into
assertions that can't silently rot.

> **Scope, on purpose.** This is the "couple hundred lines first" that the build-order rule
> (`REQUIREMENTS.md` → *Guard against over-building from n=1*) says to write before anything
> speculative. It covers R1/R2/R4/R7/R8 — everything that needs **no live Anyscale infra**.
> R2's spec generation is here; the one infra-touching line (the actual `anyscale job submit`)
> is the deliberate boundary inside `launcher.submit()` — it returns the CLI it would run and
> needs PI sign-off before it's wired to really submit. R3 (canaries), R5 (proxy calibration),
> R9 (review agents) are intentionally absent until a second campaign justifies their shape.

## Modules

| File | Requirement | What it does |
|---|---|---|
| `registry.py` | **R1** results registry | Append-only JSONL of runs + typed promote/kill/hold decisions. Stores cost in raw **and** A10G-equivalent hours. Idempotent on `(run_id, status)`; detects cross-pin and conflicting-terminal-row bugs; `reconstruct()` rebuilds a campaign's full state from disk alone. |
| `budget.py` | **R4** submit-time caps | `estimate()` a run's cost (tier-weighted), `preflight()` refuses runs over the rung cap or remaining wave envelope and escalates `full` to the PI, `wall_clock_timeout_s()` gives the runtime hard-kill. The "prepaid card, not a pinky promise." |
| `evalpin.py` | **R7** eval pinning | Content-hash an eval spec / frozen eval file → the `eval_pin` every registry row carries. Order-independent; any content change → a new pin. |
| `artifacts.py` | **R8** artifact safety | Never delete — move each prior artifact to `<name>_old_<stamp>` under one shared stamp. Idempotent and guarded: a double-submit never clobbers a backup. (Paid for by a $4 deletion.) |
| `launcher.py` | **R2** experiment launcher (spec only) | `build_job_spec()` turns an experiment dict → an Anyscale job spec: runs the R4 budget preflight first, bakes in spot + scale-to-zero + CPU fence + wall-clock timeout, encodes knobs in the run name. `submit()` is the **PI boundary** — it returns the CLI it *would* run and refuses full runs without sign-off; it never calls Anyscale. |
| `erroranalysis.py` | **BEAT_IT** DIAGNOSE step | Turns a scored eval into direction: `slice_report()` (pointwise metrics) / `recall_by_subgroup()` (ranking metrics — scores subgroups against the *global* ranking, the fix for degenerate per-slice AP) expose where the ceiling is held down; `diagnose()` emits **ranked hypothesis cards** (symptom → cheapest test → direction). Excludes no-positive slices; compares worst-vs-best subgroup within one metric space. See `../BEAT_IT.md`. |
| `metrics.py` | **BEAT_IT** MEASURE/READ | Is the number real? `average_precision`/`accuracy`, `bootstrap_ci`, and `paired_bootstrap` (+`is_real_gain`) — a gain whose paired CI straddles 0 is inside the noise, don't bank it. `positives_warning` flags few-positive evals. Deterministic (seeded). |
| `recipe.py` | **BEAT_IT** DECIDE/BANK | The compounding stack: `add_trick` measures marginal lift **on top of the current best** (a dud never lowers the best), `waterfall`, `stale_tricks` (regression guard), `dev_holdout_gap` + `holdout_budget` (two-eval discipline). |
| `contract.py` | **BEAT_IT** any-job adapter | `validate_eval_output()` checks the standard per-example record so any job plugs into metrics/erroranalysis; `single_delta()` is the confound firewall (refuses a 2-flag or no-op experiment). |
| `climb.py` | **BEAT_IT** controller | `propose_next()` drives one loop turn end to end: diagnose → candidate experiments → budget-check → ranked runnable plan (stops at the submit boundary). `should_pivot()` / `climb_summary()` read the recipe. |
| `significance.py` | **rigor** — multiple comparisons | `benjamini_hochberg()` (FDR) / `bonferroni()` across a campaign's `p_value_gain`s — so running many experiments doesn't manufacture a fake win (a nasty sim found 5-ish false positives at per-test significance). |
| `holdout.py` | **rigor** — adaptive overfitting | `Thresholdout` (Dwork et al.): query the holdout through it; it returns dev unless dev/holdout diverge (the overfitting signature), then reveals the honest holdout and spends a budget query. Two-eval discipline with teeth. |
| `canary.py` | **R3** — leak / target-leak | `shuffled_label_control` (metric must collapse to the prevalence floor or it's a leak), `too_good_too_early`, `embedding_collapse`, `audit()` → BROKEN/SUSPICIOUS/CLEAN. |
| `hooks.py` | **R4 wired live** | Claude Code `PreToolUse`/`PostToolUse` entrypoints: the pre hook prices any `anyscale job submit` (worst-case fleet from the YAML + a mandatory `# autoresearch:` declaration) and refuses it via `budget.preflight()` **before it runs**; the post hook commits the estimate as the RUNNING registry row. Armed only when `AUTORESEARCH_BASE` is set. The "by physics, not discipline" close of budget.py's stubbed-wiring gap. |
| `reconcile.py` | **R6** terminal rows | The single writer of terminal rows: sweeps RUNNING heartbeats, asks `anyscale job status` what actually happened, writes the terminal row at wall-clock x committed fleet (labeled upper bound). Back-fillable weeks later from job state alone — a Monitor adds freshness, not correctness. |
| `feasibility.py` | **budget** — is $N even enough? | Combines compute-affordability (dollars → A10G-hr vs the plan), statistical power (MDE vs target gain), and a **learning-curve** power-law fit (`fit_power_law`) that extrapolates data/compute to a target — and flags when the target is *below the fitted floor* (no budget reaches it; change the method). Verdict: NOT_ENOUGH / PILOT_ONLY / GO. |

Dollar budgets: `budget.to_a10g_hours($)` / `to_usd()` convert a `$100` envelope into the A10G-equivalent hours the caps use; `preflight(..., envelope_ah=to_a10g_hours(100))` enforces it.

Also in `metrics.py`: `p_value_gain` (one-sided bootstrap p for FDR), `min_detectable_effect`/`detectable` (power up front). In `erroranalysis.py`: `discover_slices` (auto-find the weak subgroup, SliceLine-style).

## Run the tests

```bash
cd harness
for f in test_*.py; do python3 "$f"; done      # 100 tests, stdlib only, no pytest needed
```

## The critiques these close

- **#14/#20 tier-weighting** — `registry.py` stores `a10g_equiv_hours`; `budget.py` enforces
  against it. `test_budget.test_full_always_escalates` proves campaign 02's 120 H100-hr run is
  660 A10G-eq → refused under a Wave-2 envelope.
- **#2 "prepaid card, not a pinky promise"** — `budget.preflight()` (estimate + envelope) and
  `wall_clock_timeout_s()` (runtime kill) bound cost from both ends.
- **#3 typed decisions / #4 single-writer idempotency** — `registry.py`.
- **R8 / the "never delete" memory** — `artifacts.py`.

## Wiring the submit hooks (repo `.claude/settings.json`)

```json
{"hooks": {
  "PreToolUse":  [{"matcher": "Bash", "hooks": [{"type": "command",
    "command": "python3 templates/autoresearch/harness/hooks.py pre"}]}],
  "PostToolUse": [{"matcher": "Bash", "hooks": [{"type": "command",
    "command": "python3 templates/autoresearch/harness/hooks.py post"}]}]
}}
```

Arm with `export AUTORESEARCH_BASE=/mnt/user_storage/<campaign-base>` (unset = hooks are
inert). Every job YAML then needs one declaration line or the submit is refused:

```yaml
# autoresearch: campaign=fintech_fm wave=1 rung=proxy est_hours=2.0 eval_pin=sha256:...
```

Reconcile open rows any time (idempotent; a job-state Monitor just makes it prompt):

```bash
python3 templates/autoresearch/harness/reconcile.py $AUTORESEARCH_BASE
```

Known gaps, on purpose: the gate reads the literal Bash command, so SDK submits or
ssh-wrapped submits slip past it (guardrail, not security boundary — org-level instance
allowlists are the hard layer), and wall-clock x max fleet over-counts autoscaled runs
(the console stays the invoice of record).

## How a campaign uses it

```
# 1. generate the job spec (budget preflight runs inside; refuses/escalates here)
spec = launcher.build_job_spec(exp, base=BASE)
cli  = launcher.submit(spec, pi_approved=...)   # BOUNDARY: returns the CLI, never submits

# 2. inside the run: pin the eval, protect prior artifacts, write the ledger
pin = evalpin.eval_pin(eval_spec)
artifacts.move_aside([model, embeddings, tokenized], artifacts.make_stamp())  # never delete
registry.append_run(BASE, {... "eval_pin": pin, "status": "RUNNING" ...})     # heartbeat
# ... reconcile.py (R6) writes the single terminal row on any terminal state ...
```
