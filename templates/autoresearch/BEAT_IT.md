# BEAT_IT.md — the incremental-improvement engine (the crux)

Everything else in this repo is *defense*: how not to fool yourself (`AUTORESEARCH.md`'s iron
rules), how not to overspend (`BUDGET_POLICY.md`), how to hold a number you can trust
(`REQUIREMENTS.md`). This doc is the **offense** — the rigorous, structured machine for
**beating a number over and over and compounding the wins**. It is the point of the program.
The reproduction gate just hands you a trustworthy place to start climbing.

> **Reframe: baseline-first, not reproduce-first.** The gate is not "reproduce a paper." The
> gate is **a frozen baseline + an eval you trust.** Three ways in, same engine after:
> 1. reproduce a published number (you don't own a model yet),
> 2. **bring your own model** (you already have a scored internal baseline — most real users),
> 3. train an honest baseline from scratch (no reference exists).
> Once you have `(trusted_baseline_score, trusted_eval)`, the reproduction step is *done* and
> the climb is the whole game.

Light metaphor, because it's the right one: **you're climbing a mountain in the fog.** The
eval is your altimeter. Each experiment is a step. Error analysis is checking the map before
you step. The *recipe* is the rope of holds you've already banked. The holdout eval is base
camp — you do NOT trample it, or you lose your only honest altitude reading.

---

## The loop (memorize this — it's the engine)

```
        ┌─────────────────────────────────────────────────────────┐
        │  MEASURE → DIAGNOSE → HYPOTHESIZE → TEST → READ → DECIDE  │
        │     ↑                (cheapest test that could           │  → BANK a win
        │     └────────────────  change your mind)  ───────────────┘     (compound)
```

The naive way ("try stuff, keep what helps") loses to this every time, because the naive way
(a) chases noise, (b) confounds changes, (c) overfits the eval, and (d) thrashes on dead ends.
Each step below is a guard against one of those.

| Step | The question it answers | The failure it prevents |
|---|---|---|
| **MEASURE** | Where am I, honestly, with error bars? | reading tea leaves off one run |
| **DIAGNOSE** | *Why* is it wrong? Where's the ceiling? | guessing what to fix |
| **HYPOTHESIZE** | What's the highest expected-lift-per-GPU-hour idea? | random flailing |
| **TEST** | What's the *cheapest* experiment that could change my mind? | $200 answers to $2 questions |
| **READ** | Is the move real, and did it move for the reason I thought? | banking noise / right-number-wrong-reason |
| **DECIDE / BANK** | Keep it on the stack? Does the stack still hold? | silent regressions, non-compounding |

---

## DIAGNOSE — error analysis is the engine of direction

This is the part the program was missing and the part you asked for. **Do not guess what to
improve. Look at the errors and let them point.** Three moves, cheapest first:

### 1. Slice the eval and rank the failures
Compute the metric *per segment*, not just overall: by class, by difficulty, by sequence
length, by entity, by time bucket, by subgroup. Two lists fall out:
- **worst slices** (low metric), and
- **biggest slices** (large population).
Your target is the intersection — a segment that's both bad *and* common is where the metric
ceiling is being held down. (Ng's error analysis, and every Kaggle writeup ever.)

### 2. Ceiling / oracle analysis — measure ROI before building anything
For each pipeline component, ask: *if this were perfect, how much would the metric move?*
Cheaply fake perfection (inject ground truth into that stage) and read the delta. Now you know
the **maximum payoff** of fixing each part *before* you spend a GPU-hour building the fix. Fix
the part with the tallest ceiling, not the part that's most fun. (Ng's ceiling analysis.)

### 3. The diagnosis → direction table (the "what do I test next?" brain)

This is the heart of the engine. A symptom in the numbers maps to a cause, to the *cheapest*
test that could confirm it, to a direction:

| Symptom (what the numbers show) | Likely cause | Cheapest test to confirm | If confirmed → direction |
|---|---|---|---|
| train ≫ dev (big gap) | overfitting | +dropout/wd **or** 2× data on a proxy | regularize · augment · more data |
| train ≈ dev, both mediocre | underfit (capacity or optimization) | can it overfit 100 examples? | no → optimization/LR/bug; yes → bigger/longer/better features |
| strong overall, one slice awful | slice-specific failure | eval-only re-score that slice with one tweak | targeted feature / loss-weight / data for that slice |
| **upstream metric up, downstream probe flat** | upstream isn't learning the *useful* signal | swap the readout on frozen features (≈free) | change the pretraining objective, **not** the head |
| gain sits inside the noise band | it's not real | +N seeds, paired bootstrap | discard — do **not** bank; you need a bigger effect |
| tuned cheap baseline ≈ your model | model not earning its keep, or leakage, or easy task | run the honest floor to convergence | attack the gap, or suspect a leak |
| metric up but the *mechanism* metric flat | right number, wrong reason (confound) | ablate the suspected confound | don't trust it until the mechanism moves too |
| dev keeps rising, holdout flat | you're overfitting the dev eval itself | spend one holdout query | freeze dev, get fresh dev data |

Every row turns "I got a number, now what?" into a deterministic next move. **The engine's job
is to run the top rows automatically and hand you ranked hypothesis cards.**

---

## HYPOTHESIZE — the backlog is a portfolio, not a to-do list

Beating a metric is search under a budget. Treat ideas like a fund manager treats bets.

- **Every idea is a card:** `{symptom it addresses · mechanism (why it should move the metric)
  · cheapest validating test · rung · expected lift · cost (GPU-hr) · confound risk}`. Error
  analysis *emits* these cards; you don't brainstorm them from the ceiling.
- **Prioritize by expected-lift-per-GPU-hour**, and run it like **successive halving over
  ideas** (ASHA, but the "trials" are research hypotheses): give many ideas a cheap probe, kill
  the bottom half fast, pour the freed budget into survivors.
- **Probe before sweep** (already an iron rule): spend $2 to check an idea carries *any* signal
  before spending $200 tuning it. The campaign that nearly swept XGBoost params over
  signal-free embeddings is the cautionary tale.
- **Mechanism check as the entry filter** (already in `_TEMPLATE.md` §9): a ported idea must
  say *why the mechanism holds here*. A card that fails it gets one probe, not a sweep.

---

## TEST — the cheapest experiment that could change your mind, and the probe pattern

### One flag = one change (the confound firewall)
Every experiment is exactly one delta from its parent (Iron Rule #8). Make it *mechanical*:
the launcher should refuse a config diff of more than one key. If you change two things and the
number moves, you've learned nothing — you can't attribute it. This is non-negotiable for
compounding, because a win you can't attribute is a win you can't stack.

### The downstream-probe pattern (this is load-bearing — you called it out for a reason)
**You usually can't cheaply or directly measure the thing you actually care about.** "Is the
foundation model any good?" is abstract and expensive. So you attach a *cheap downstream task*
and measure **that** — the probe is your thermometer. In the fraud campaign you trained the FM,
then fine-tuned a fraud classifier on its frozen embeddings *specifically to prove the FM
learned something useful*. That extra step wasn't overhead — **it was the measurement**.

Generalize it into a required artifact for any campaign whose "thing" is upstream (a
representation, an FM, an embedding, a pretrained trunk):

- **Declare a probe.** A cheap, fast downstream task that stands in for "did the upstream
  improve?" (SSL's linear-probe convention: DINO/SimCLR are judged by linear-probe accuracy on
  frozen features, precisely because it's cheap and it isolates representation quality.)
- **Build a probe ladder**, cheapest → most faithful, and climb the cheap rungs:
  1. **linear probe** on frozen features (minutes, ~$0),
  2. **shallow MLP probe** (still cheap),
  3. **light fine-tune** of a head,
  4. **full downstream training** (the expensive, faithful confirmation).
  You hill-climb on rungs 1–2 and only *confirm* winners on 3–4.
- **Decouple the expensive from the cheap.** Extract embeddings **once** (expensive), then
  sweep 20 probes/readouts for nearly free. This cost structure is the whole reason
  representation hill-climbing is tractable — **and it's exactly where the fraud campaign's 7×
  readout swing lived.** The probe ladder isn't just validation; it's the *surface you climb*.
- **First validation after any upstream change:** does the cheapest probe clear the baseline?
  If the FM changed but the linear probe didn't move, the FM didn't learn anything useful — go
  fix the objective, don't waste a full downstream run.

### Rungs are the fidelity ladder of the test itself
smoke (does it run) → proxy (rank the idea) → full (publishable). Never HPO at full scale;
tune continuous knobs with ASHA at proxy, confirm the winner once at full.

---

## READ — interpret honestly (two ways the climb lies to you)

### 1. Is the move even real? (the noise floor)
Every claimed lift clears the noise band first: paired bootstrap over the eval, ≥3 seeds for
anything stochastic. A move inside the noise is **not a result** — most reported "improvements"
in the literature sit inside seed-to-seed variance. If you can't separate it from noise, you
either need a bigger effect or more positives (see power, below). Do **not** bank it.

### 2. Are you overfitting the eval by *looking* at it? (the two-eval discipline)
Here's the subtle killer of long climbs: if you make 200 decisions against the same eval set,
you've done gradient descent on it *with your own choices* — you've overfit it without ever
training on it. The fix is a discipline borrowed from competition ML and the "reusable holdout"
theory:
- **Dev eval** = the practice court. Iterate on it freely.
- **Holdout eval** = the championship. **Touch it rarely** — every query spends from a budget,
  and the more you peek the less it means. Report the **dev-vs-holdout gap**; when dev climbs
  and holdout stalls, you're overfitting dev — freeze it and get fresh dev data.
This makes "overfitting the eval" a *visible, spendable budget* instead of an invisible rot.

### 3. Did it move for the reason you thought? (mechanism confirmation)
The number going up is necessary, not sufficient. If your hypothesis was "longer context helps
because fraud clusters in bursts," then *also* check the burst-slice improved. A metric that
moves while its mechanism metric is flat is a **confound** wearing a win's clothing — the FM
campaign's whole reason-for-being was catching one of these (their "FM embedding" was secretly
single-transaction).

---

## DECIDE / BANK — how the wins compound (or silently rot)

Incremental improvement only *compounds* if wins stack and don't quietly undo each other.

- **The current-best recipe** is a first-class object: the ordered stack of every change that's
  ON, each tagged with its **measured marginal contribution** (the "bag of tricks" discipline
  from the ResNet-tricks paper and every Kaggle gold writeup).
- **Measure on top of the current best, not on top of the raw baseline.** An idea that helped
  the bare baseline may do nothing (or hurt) once three other tricks are on — interactions are
  real (the "batch-size bump halved the optimizer steps and degraded everything" gotcha is
  exactly this).
- **Regression guard:** periodically re-run the full stack and re-verify each trick still earns
  its slot (ablate-on-demand: turning off trick #3 should still cost you). Interactions shift as
  the stack grows; a trick can go stale.
- **Know when to pivot.** Track marginal lift per experiment. When it flattens on one lever
  (say, readout tweaks), you've hit a local ceiling — switch levers (data → architecture →
  objective → ensemble) or declare the local max and stop paying for noise.

---

## SEE IT — the eval numbers and TensorBoard, made into a climb you can watch

You think temporally and visually, so the instrumentation *is* part of the engine:

- **The climb chart** — decision-metric vs experiment-number, with the **current-best line**, a
  shaded **noise band**, and each experiment a promote/kill/hold dot. You literally watch the
  altitude rise, and a dot inside the band is visibly "not real."
- **The marginal-lift waterfall** — the recipe as a waterfall: `baseline → +trick₁ → +trick₂ →
  …`, each bar its measured contribution. The single most motivating view for incremental work,
  and it makes a stale/negative trick jump out.
- **The per-slice heatmap over time** — error-analysis slices on one axis, experiments on the
  other; watch which segments climb and which stay stuck red (your next target).
- **The climb tree** — experiments form a *tree* (each node a config, each edge a single
  change, each node its score); dead branches prune, the best path glows. It's tree search over
  config space, and it's the "which ways could we go" view you asked for. (UCB/MCTS can even
  pick which node to expand next.)
- **Wiring:** all of it is a read over the R1 registry (`harness/registry.py`) — runs carry the
  decision metric + CI + cost; decisions carry promote/kill + reason. The dashboard
  (`viz/mission-control.html`) already renders the registry; the climb chart / waterfall / tree
  are new views over the same rows. TensorBoard gets the per-experiment scalars for the
  in-the-weeds training curves; the dashboard gets the campaign-level climb.

---

## The anti-fooling-yourself rules, specific to climbing

The general iron rules still hold; these are the ones that bite *during a long climb*:
1. **Noise floor on every decision**, not just at publish.
2. **Two-eval discipline** — dev to climb, holdout as a rarely-spent budget; report the gap.
3. **Confound firewall** — one change per experiment, mechanically enforced.
4. **Regression re-check** — the stack is re-verified as it grows; tricks can go stale.
5. **Multiple-comparisons awareness** — across a whole campaign you run *many* tests; expect
   some fake wins by chance, and treat a lone surprising win with suspicion until it replicates.
6. **Power first** — before a campaign, given the positive count, ask what effect size you can
   even detect. If the answer is "nothing smaller than the gain you're chasing," fix the eval
   (more positives / more data) before spending GPU to confirm you can't tell.

---

## Worked example — the fraud FM campaign *as a climb*

The whole engine, on the real case:
1. **Baseline + eval:** reproduce NVIDIA's AP through the Ray pipeline (gate) → trusted start.
2. **Probe declared:** FM quality is abstract, so the thermometer = **fraud AP via a downstream
   classifier on frozen FM embeddings** (the extra step). Extract embeddings once; sweep readouts
   cheap.
3. **Diagnose:** the "FM embedding" barely beat raw features → *upstream metric up, downstream
   probe flat* row of the table → direction: it's the **readout**, not the representation.
4. **Hypothesize + test cheap:** sweep readouts on the frozen embedding (probe ladder rung 1–2):
   masked 0.077 → InfoNCE 0.184 → linear 0.397 → MLP 0.535 — a **7× swing, zero pretraining
   change.** Cheapest possible test, biggest lift in the campaign.
5. **Read honestly:** bootstrap CIs (only 112 test frauds → CI-mandatory); shuffled-label
   control collapses to prevalence; velocity-feature baseline (the honest floor) *doesn't*
   recover the lift → the win is real and it's the representation+readout, not feature eng.
6. **Bank + compound:** context-length 1024 (found on the curve, not guessed) stacks on top;
   frequency-prior blend stacks on the reco task. Each measured on top of the current best.

That climb — not the reproduction — is the deliverable. The reproduction just told us the
altimeter was trustworthy before we started up the mountain.

---

## What's built vs. what's next (this engine, concretely)

- **Built now:** `harness/erroranalysis.py` — the DIAGNOSE step as code: slice an eval, rank the
  worst/biggest slices, and emit **ranked hypothesis cards** from the diagnosis table above
  (symptom → cheapest test → direction). Tested.
- **Built now (this round):** `metrics.py` (bootstrap CI + paired significance — the noise
  floor), `contract.py` (the any-job eval contract + confound firewall), `recipe.py` (the
  compounding stack), `climb.py` (the loop controller). All tested; see `harness/README.md`.
- **Designed, next:** the climb-chart / waterfall / tree dashboard views over the registry; a
  `probe.py` helper standardizing the probe ladder (extract-once, sweep-many); R6 monitor
  (needs live job state). Build after a real second campaign shapes them.

---

## Literature — what's known, and where this engine stands on it

This isn't invented from scratch; each part rests on a line of work, and knowing it tells us
where to be careful. (arXiv IDs; the point is the map, not the citation count.)

**Autonomous ML-research agents — the space this lives in.** *The AI Scientist* (2408.06292)
and *-v2*'s agentic tree search, *Agent Laboratory* (2501.04227), *MLR-Copilot*, and
execution-grounded efforts (2601.14525) all attempt the full generate→run→write loop. The
*Automated LLM Speedrunning Benchmark* (2506.22419) is the closest cousin to us: it measures
whether an agent can *reproduce known incremental improvements* to nanoGPT — i.e. exactly the
"beat it, step by step" skill. **Where we differ:** these are mostly idea-generation-first and
weak on rigor/cost governance; our bet is the opposite — a hardened MEASURE/READ/BANK spine
(registry, budget, noise floor) with the agent proposing *within* it. The honest lesson from
this literature (and *"What Fits Doesn't Overfit"*, 2606.11045) is that research agents readily
**overfit the eval** — which is why the two-eval discipline below is load-bearing, not optional.

**READ — is the gain real.** *Accounting for Variance in ML Benchmarks* (2103.03098) shows a
single-seed "win" is usually inside variance, and that summing multiple variance sources gives
a far cheaper honest estimate — the empirical basis for our noise-floor-on-every-decision rule.
*"When +1% Is Not Enough: A Paired Bootstrap Protocol for Evaluating Small Improvements"*
(2511.19794) is essentially the method in `metrics.paired_bootstrap` — resample the *paired*
per-example scores, and only believe a small gain if the paired CI excludes zero. We implement
exactly this; it's the difference between banking a real hold and banking noise.

**READ — overfitting the eval you climb on.** Dwork et al., *Generalization in Adaptive Data
Analysis / Holdout Reuse* (1506.02629, Thresholdout), and Blum & Hardt's *Ladder* show that
adaptively querying a holdout thousands of times overfits it, and give the fix: a limited-
feedback holdout that only reports when you *significantly* beat your best. That's precisely
`recipe.holdout_budget` + the dev/holdout gap — a spendable query budget, not a free ruler.

**DIAGNOSE — error analysis / slice discovery.** Our slicing is the manual form of an active
research area: *SliceLine*, *DivExplorer*, DEIM/Edisa (2211.04476), and the human study
*"Where Does My Model Underperform?"* (2306.08167) automatically *discover* coherent high-error
slices instead of requiring you to name them. **Upgrade path:** today a campaign declares its
slice features by hand; plugging a slice-discovery algorithm into `erroranalysis` would let the
engine *find* the failing subgroup (e.g. "burst" frauds) without being told to look — a concrete
next step. Note their lesson too: naive per-metric slicing misbehaves on ranking tasks, which is
exactly the degenerate-AP bug simulation caught here (fixed via `recall_by_subgroup`).

**HYPOTHESIZE/TEST — allocation.** Successive Halving / *Hyperband* / *ASHA* are the theory
under "give many ideas a cheap probe, promote the top 1/η, pour budget into survivors." Our
probe-before-sweep and lift-per-GPU-hour ranking are SHA applied to *research hypotheses*, not
just hyperparameters; ASHA's asynchrony maps onto the monitor-driven chaining (R6) we haven't
built yet.

**The gaps the literature says we still have** (honest): no *program-level* multiple-comparisons
control (run enough campaigns and a fake win appears — Benjamini-Hochberg FDR is the tool); no
up-front *power analysis* (given the positive count, what effect is even detectable — `metrics.
positives_warning` is a stub, not a real MDE calc); and benchmark **contamination** for
foundation-model campaigns (did the FM pretrain on the eval?) has no systematic gate. These are
named in `critiques.md` and are the next rigor frontier once the engine is exercised on a real
second campaign.
