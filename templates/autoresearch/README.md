# Autoresearch — a rigorous, agent-driven loop for empirically improving a model

A harness and methodology for an **individual MLE** to run disciplined, agent-driven research:
take a concrete research question — *"what data mix / hyperparameter / reward shaping / readout
maximizes my model's downstream metric?"* — run the full experimental loop against it, and come
out with a **provenance-clean, defensible demonstration of whether the loop actually improved the
model.** The human owns the question and the taste; the agent owns the orchestration, the
bookkeeping, and the first-pass interpretation.

**The single outcome this exists to produce:** empirical evidence that the autoresearch loop can
improve a model's measured performance — or an honest finding that, for this question, it can't.
Nothing else. It is not a product pitch and carries no positioning; the technical design and its
rationale live in `research_director.md`.

**Ray / Anyscale is the compute substrate, and only that.** It's here because running an
experiment matrix affordably and observably needs distributed compute: Ray Data to stream and
reweight data mixes without recopying the dataset, Ray Train to run the matrix in parallel with
fault recovery, Ray Tune (ASHA) to kill losing branches early, Ray Serve for eval rollouts. That's
the whole role — the substrate that makes the loop tractable, not a vertical or a sales story.

Distilled from the fintech transaction-FM campaign (2026-07-06 → 07-11), which reproduced a
published fraud benchmark and then improved on it via the readout — the worked example in
`campaigns/00-transaction-fm-fraud.md`.

## Who this is for

An IC MLE who is either doing research inside their company, or experimenting with models on
Anyscale to understand and write up what works. You bring a model and a question; the harness runs
the rigorous loop and hands you a result you can trust.

## What counts as an outcome

Every outcome first requires a **trusted baseline + a trusted eval** (reproduce a reference's
number, bring your own scored model, or train an honest baseline). Then one of:

1. **A demonstrated improvement** — the loop moved the decision metric, and the move survives the
   noise floor (CI-separated, paired bootstrap) against the *strongest* cheap baseline.
2. **An efficiency win at held quality** — materially cheaper/faster with the metric held within
   CI, measured against a competent same-budget baseline (`REQUIREMENTS.md` non-functional #8).
3. **An honest rigor finding** — the improvement isn't real, or a published claim was overstated
   (a leak, a noise-level gain, a robustness gap). A truthful negative is a first-class result.

## The one non-negotiable

**Establish a trusted baseline before you claim anything.** Two gates, in order:
1. **Artifact gate** — run the reference's *shipped* eval on its *shipped* checkpoint, match the
   number. (Skip if you're improving your own already-scored model.)
2. **Pipeline gate** — reproduce that number through your own Ray pipeline, proving your data/eval
   stack is sound.

No trusted number, no claim. Improvement experiments don't spend budget until a baseline holds.

## The loop

The design is a **deterministic outer state machine that owns control flow, with focused agent
loops inside each phase and explicit gates between phases** — so an unreliable reasoner is boxed
in and its output stays trustworthy (`research_director.md` §3, §6):

```
FRAME ─G0─▶ DESIGN ─G1─▶ LAUNCH ─G2─▶ MONITOR ─G3─▶ EVALUATE ─G4─▶ INTERPRET ─G5─▶ REPORT
```

The sequencer (not the LLM) decides when a phase is done; each inner agent gets a scoped toolset;
every number in the report traces to a measurement actually taken. Rounds loop: INTERPRET's
recommended next matrix feeds a new DESIGN cycle until the stopping rule fires or you intervene.

## Operating model

- **You own** the question, the definition of "better" (the eval), the taste in experiment design,
  and the "is this real?" sign-off. No headline result is trusted without your read.
- **The agent owns** the grind: writing configs, launching and monitoring runs, killing losers,
  running evals, first-pass interpretation, and drafting the report — largely unattended.
- **The disk is the source of truth**, not any context window. Every result is a durable file;
  every decision is a registry row. Any session (human or agent) can resume from disk alone.

## Docs

| Doc | What it is |
|---|---|
| `research_director.md` | The technical design + rationale: the phased loop, the gates, the Ray substrate mapping. The north star. |
| `BEAT_IT.md` | The improvement engine: error-analysis → next test, the compounding recipe, anti-fooling-yourself rules. |
| `REQUIREMENTS.md` | What the harness must do (R1 registry … R9 review), made buildable. |
| `BUDGET_POLICY.md` | Cost discipline: GPU-hour caps, sample-size (parity) mode, submit-time enforcement. |
| `SEED_INDEX.md` | The menu of candidate campaigns, by cost and reproduction confidence. |
| `campaigns/*.md` | One pre-registered plan per candidate question; `_TEMPLATE.md` is the schema. |
| `harness/` | The engine as tested code (registry, budget, metrics, error-analysis, feasibility, …). |
| `AUTORESEARCH.md`, `CLAUDE_WITH_ANYSCALE.md` | The battle-tested methodology + operating notes from the worked example, in `../fintech_transaction_fm/claude-anyscale/`. |

## How to start a campaign

1. Copy `campaigns/_TEMPLATE.md` to `campaigns/NN-<slug>.md` and fill every field. The plan is a
   **pre-registration** — once the first paid run happens it's frozen; changes land as new commits
   marked `AMENDED`, and every registry row records the plan's commit SHA it ran under.
2. Size the budget with `harness/feasibility.py` and set the envelope (`BUDGET_POLICY.md`).
3. `git clone` the reference repo under `$AUTORESEARCH_REFS` (default `~/anyscale/`); durable
   artifacts live under `$BASE` on cluster storage — no laptop-specific paths.
4. Run the loop. Log every run to the registry (`REQUIREMENTS.md` R1) — no state in a context window.

## Design assumptions (stated so they're easy to challenge)

1. **GPU-hours are the budget currency**, converted to dollars through one table; a dollar
   envelope maps to A10G-equivalent hours (`harness/budget.py`).
2. **Reference numbers come from the repo's eval code, not its paper/blog** — a target to confirm
   from code, never trusted ground truth.
3. **Campaigns are pre-registered** — hypotheses, decision metric, eval pin, and budget written
   down *before* the first paid run.
4. **Cost estimates are pre-calibration guesses** (±50%); the proxy-calibration step tightens them
   before any full run.
