# Research Director Agent — Technical Design

_Author: Geoff Counihan · Draft v3 · 2026-07-13_
_Lineage: successor to "Bet B / Experiment Director Agent" from the May 2026 MLE-agent pitch stress-test (`mle-agent-pitch-stress-test.md`). This is a technical design, not a pitch — it assumes the strategic case is already made and focuses on how the thing is actually built._

---

## 1. What this is, in one paragraph

The Research Director is an agent that takes a **research question** — "what data mix maximizes downstream code-gen on our 7B base?", "what teleop:sim:internet ratio gives the best pick-and-place success on our VLA?", "what GRPO reward shaping closes the math gap without wrecking instruction-following?" — and runs the full experimental loop against it: it designs the experiment matrix, launches parallel training runs across Ray, kills underperforming branches with multi-fidelity early stopping, runs downstream evals, reasons about what the results mean, proposes the next round, and produces a defensible report with every number traceable to a measurement it actually took. It is **scaffolding for a researcher, not a replacement for one** — the human owns the question and the taste; the agent owns the orchestration, the bookkeeping, and the first-pass interpretation.

It is deliberately the *offensive* counterpart to the migration agent (Bet A). The migration agent saves money on a workload you already have. The Research Director expands what's tractable — it lets a team run the sweep they couldn't previously afford in researcher-attention.

## 2. Why the design is different now than in May

The May draft treated most of the underlying capability as something we'd have to build. Two things have changed the picture, and both simplify the design:

**The primitives now exist inside Anyscale.** Anyscale shipped an [LLM Post-Training skill](https://www.anyscale.com/blog/anyscale-llm-post-training-skill) that already chooses between SFT / DPO / KTO / ORPO / SimPO / PPO-RLHF / GRPO / DAPO and generates ready-to-run configs for LLaMA-Factory, SkyRL, or Ray Train against Anyscale Jobs. [SkyRL](https://www.anyscale.com/blog/vision-language-model-reinforcement-learning-skyrl) is a modular RL library that now supports vision-language RL. There's a published [VLA fine-tuning pipeline with LeRobot on Ray](https://www.anyscale.com/blog/vision-language-action-pipelines-vla-robotics-ray-anyscale) that pipelines CPU-bound video decode against GPU-bound transformer training. **The Research Director is therefore an orchestration layer on top of skills that already work — not a from-scratch training stack.** That is the single most important design change: it drops from "build the whole thing" to "build the experiment-design brain and the eval loop that drive the existing skills."

**The agent-scaffold layer has commoditized even faster than we feared.** On MLE-bench, the public numbers moved from ~17% (2024) to a crowded frontier in 2026: [EurekAgent reports 85.71% any-medal on the 7-task Lite subset](https://arxiv.org/pdf/2606.13662), [ML-ACE reports a 56.4% average medal rate](https://arxiv.org/pdf/2410.07095) on the fuller benchmark, ML-Master 2.0 leads across complexity tiers, and single-agent architectures like [Operand Quant](https://arxiv.org/pdf/2510.11694) are competitive. End-to-end research systems — [AI Scientist v2](https://arxiv.org/pdf/2504.08066), [AutoSOTA](https://arxiv.org/pdf/2604.05550), DeepEvolve, Kosmos — are now real. **The lesson is unchanged and sharper: do not compete on "we have a better research agent." The scaffold is a fast-moving research commodity.** The defensible layer is the distributed-compute substrate the agent drives (Ray Data streaming mixes, Ray Train multi-node, Ray Tune multi-fidelity, SkyRL) plus the eval/gate discipline that makes the agent's output trustworthy. Anyscale owns the substrate; almost nobody has productized the disciplined loop on top of it for serious training.

**The honest ceiling is documented.** [Why LLMs Aren't Scientists Yet (Jan 2026)](https://arxiv.org/abs/2601.03315) ran four autonomous ML-paper attempts and catalogued six recurring failure modes: training-data-default bias, implementation drift under execution pressure, memory/context degradation over long horizons, **overexcitement that declares success despite obvious failure**, insufficient domain intelligence, and weak scientific taste in experiment design. This is not a reason not to build — it's the *specification for the gates*. Every one of those failure modes maps to a gate below. The design's whole job is to box in an unreliable reasoner so its output is still trustworthy.

## 3. System shape: deterministic skeleton, agentic interiors

Same architecture that the migration agent converged on, for the same reasons. A pure sequential script is too brittle for the weird edges of real research setups; a single "one big agent, go" loop drifts, burns compute, and can't be validated mid-flight. The shape is a **deterministic outer state machine that owns control flow, with focused agent loops inside each phase and explicit gates between phases.**

```
                    ┌─────────────────────────────────────────────┐
   research         │            RESEARCH DIRECTOR                 │
   question    ───▶ │  (deterministic sequencer + budget owner)    │
   + constraints    └─────────────────────────────────────────────┘
                          │        │        │        │        │
          ┌───────────────┘        │        │        │        └──────────────┐
          ▼                        ▼        ▼        ▼                        ▼
      FRAME  ──G0──▶  DESIGN ──G1──▶ LAUNCH ──G2──▶ MONITOR ──G3──▶ EVALUATE ──G4──▶ INTERPRET ──G5──▶ REPORT
       │                │             │              │               │                │                │
    scoped           scoped        Ray Data /     multi-fidelity   downstream       LLM analysis    typed report,
    agent            agent         Train/SkyRL    early-kill       eval rollouts    (bounded)        no free numbers
                                    launch         controller
```

The outer sequencer, **not the LLM**, decides when a phase is done, whether to advance, retry, downgrade, or escalate to the human. This is what makes runs replayable from a checkpoint and what stops the agent from silently making policy calls. Each inner agent gets a **scoped toolset** — the DESIGN agent cannot launch jobs, the LAUNCH agent cannot write eval code, the INTERPRET agent cannot spawn new runs. Tool-scoping is a hard boundary, not a prompt request.

### The seven phases

1. **FRAME.** Turn the human's question into a formal *experiment spec*: the axes to vary (data-mix ratios, hyperparameters, reward configs, curricula), the fixed base (model, tokenizer, infra), the downstream eval suite that defines "better," the compute budget, and the stopping rule. This phase is mostly a structured interview with the researcher — it does not run anything. Output is a typed `ExperimentSpec`.
2. **DESIGN.** Turn the spec into a concrete *experiment matrix*: which points in the search space to run, in what order, at what fidelity, under what early-kill policy. This is where the LLM's judgment is used most (proposing a smart initial matrix) and where it is trusted least (every proposed point is checked against budget and validity gates).
3. **LAUNCH.** Materialize each matrix point as a real Anyscale job by delegating to the **existing skills** — the post-training skill for method/config selection, Ray Data for the streaming data mix, Ray Train / SkyRL for the run. The Research Director does not reimplement training; it fills in the experiment-specific knobs and submits.
4. **MONITOR.** Watch loss curves, GPU utilization, divergence signals, and throughput across all live runs. Feed a **multi-fidelity early-termination controller** (ASHA-style, already in Ray Tune) that kills clearly-losing branches to reclaim compute — with an LLM-in-the-loop only to *flag* anomalies for the human, never to silently kill a run outside the controller's rules.
5. **EVALUATE.** For surviving checkpoints, run the downstream eval suite as Ray Serve / batch rollouts. This is the phase that defines "better" and it is deliberately the most expensive and most guarded.
6. **INTERPRET.** The LLM analyzes the results table: which axis mattered, where the frontier is, what's surprising, what to run next. Bounded, and forbidden from inventing numbers — it reasons over a typed metrics blob, not free text.
7. **REPORT.** Emit the round report: the matrix, the results, the frontier, the recommended next round, and a full provenance trail. Every numeric claim is templated from the typed metrics blob, never free-form.

Rounds loop: INTERPRET's recommended next matrix feeds a new DESIGN→...→REPORT cycle until the stopping rule fires or the human intervenes.

## 4. The Ray primitive mapping

This is the part that is genuinely defensible, because it is where distributed compute does work that a Modal or a Lightning can't easily replicate.

| Experiment need | Ray / Anyscale primitive | Why it's load-bearing |
|---|---|---|
| Try many data-mix ratios without rebuilding the dataset each time | **Ray Data** streaming, heterogeneous-source mixing | The whole point of a data-mix sweep — reweight sources on the fly, stream into training, no N full dataset copies |
| CPU-bound preprocessing (multi-cam video decode, tokenization) overlapped with GPU-bound training | **Ray Data + Ray Train** pipelined stages, independently scaled | The published VLA pattern — GPU never idles waiting for decode |
| Many training runs in parallel across the matrix | **Ray Train** multi-node DDP/FSDP with fault recovery | The matrix is embarrassingly parallel; fault recovery matters at 50–200 runs |
| Kill losing branches early, reallocate to promising ones | **Ray Tune** ASHA / multi-fidelity scheduling | Turns a 200-run brute-force sweep into an attention-efficient search |
| RL post-training experiments (GRPO/DAPO/RLVR, VLA RL) | **SkyRL** + post-training skill | Method selection + config gen already solved; the agent drives, doesn't rebuild |
| Downstream eval rollouts at scale | **Ray Serve** / Ray Data batch inference | Spin up each surviving checkpoint, run the eval suite, tear down |
| Kill-decision signals (loss, GPU util, divergence) in real time | Anyscale cluster observability | The early-kill controller is only as good as the signals it reads |

The one-line moat statement: **the agent is the cognitive design layer; Ray is the substrate that makes running the design affordable and observable.** Databricks [Agent Bricks](https://www.databricks.com/blog/agent-bricks-dais-2026) — now a mature platform with 100k+ agents built — auto-optimizes *application-layer* agents over enterprise data. It does not run serious training experiments at frontier scale. That is the gap.

## 5. Search & optimization strategy

The matrix is not a grid search. The design has three layers, cheapest first:

1. **LLM-proposed structured search.** The DESIGN agent proposes the initial matrix using research priors ("start with these five mix ratios because the base is code-heavy already"). This is where LLM judgment adds value over blind search — it prunes the obviously-bad regions before spending a GPU-hour. It is also where the agent is least trustworthy, so every proposed point passes a validity gate (§6, G1).
2. **Multi-fidelity bandit allocation.** Runs start at low fidelity (fewer steps / smaller data slice / shorter rollouts). ASHA promotes survivors to higher fidelity and kills the rest. This is the compute-efficiency workhorse and it is deterministic — the controller's rules, not the LLM, decide promotion/kill.
3. **Round-over-round refinement.** After each round, INTERPRET proposes the next matrix conditioned on the frontier so far. This is Bayesian-ish but semantic: "the teleop-heavy corner won; next round explore around it and test whether adding sim helps or hurts." The human approves the next matrix before it launches (configurable: auto-advance within budget, or gate on approval).

**Sample-size discipline (carried over from the migration-agent harness).** Every experiment runs in two modes. *Parity/dev mode* is sample-sized — small data slice, few steps, short rollouts — runs in minutes for cents, used to validate the loop and shake out bugs before spending real money. *Full mode* scales to the real run and is only entered for matrix points that survived parity-mode sanity checks. This collapses the cost-management problem: you never launch a 1000-GPU-hour run to discover the config was malformed.

## 6. Validation gates

The gates are the agent's discipline. Each maps to a documented failure mode from the autonomous-research literature, has a deterministic check where possible, and an explicit on-fail action (`retry` / `downgrade` / `escalate` / `abort`).

| Gate | Checks | Guards against | On fail |
|---|---|---|---|
| **G0 Spec Well-Formed** | Every axis has a range; "better" is defined by a concrete eval suite; budget + stopping rule are set | Vague questions that produce vague sweeps | escalate to human |
| **G1 Matrix Valid & In-Budget** | Every proposed point is runnable (valid config, feasible resources) and the matrix sums under budget | LLM proposing invalid or ruinously expensive configs; training-data-default bias (e.g. deprecated libs/APIs) | retry design, then escalate |
| **G2 Launch Integrity** | Each job launched matches its matrix point; data mix actually applied; seeds/versions pinned | Silent drift between what was designed and what ran | abort the point, flag |
| **G3 Run Health** | Loss not NaN/diverging; GPU util sane; no silent stall | Wasting budget on dead runs; implementation drift | early-kill via controller |
| **G4 Eval Integrity** | Eval suite ran on the actual checkpoint; held-out data truly held out; graders deterministic where claimed | The "declared success despite failure" failure mode — the single most dangerous one | quarantine result, escalate |
| **G5 Provenance** | Every number in the report traces to a measurement taken this run; no interpolated/hallucinated values | Overexcitement; fabricated significance | block report emit |
| **G6 Significance** | Deltas have enough samples/seeds to be defensible; variance reported; re-run consistent | Calling noise a result | downgrade claim to "inconclusive" |
| **G7 Escalation Discipline** | Uncertain or out-of-scope decisions are surfaced, not silently made; forbidden questions (e.g. "should I just lower the bar?") are refused | Weak scientific taste dressed up as confidence | route to human |

Two gates carry most of the weight. **G4 (Eval Integrity)** and **G5 (Provenance)** together defeat the worst failure mode in the literature — the agent that reports success despite obvious failure and overstates significance. Structurally, the report layer accepts values *only* from a typed metrics blob produced by measured runs; the LLM never writes a number into the report directly.

## 7. Memory & knowledge accumulation

Two distinct memory systems, different access patterns:

- **Within-campaign state** (the sequencer's checkpoint): every round's matrix, results, and decisions, so any run is replayable and the next round is conditioned on real history. This directly counters the "memory degradation over long horizons" failure mode — coherence lives in the deterministic store, not the LLM's context window.
- **Cross-campaign corpus** (the production knowledge base): past campaigns as retrievable in-context examples, keyed by question archetype (data-mix sweep / RL reward search / curriculum design / scaling-law probe). When a new VLA data-mix question comes in, the agent pulls prior VLA data-mix campaigns as priors. This is the same Notion/Slack-style retrieval layer we discussed for the migration agent, adapted to research campaigns.

## 8. The agent's own eval suite

Separate from per-campaign gates, the agent itself needs a regression corpus run on every change — otherwise it's a one-shot demo, not an improvable product.

- **Replay campaigns**: (research question → known-good matrix → ground-truth frontier) tuples seeded from real internal sweeps. Score whether the agent rediscovers the known frontier within tolerance and budget.
- **Poisoned campaigns**: questions that look normal but contain a trap (a leaky eval, a mix ratio that silently NaNs, a "win" that's pure noise). Score whether the agent *catches and escalates* rather than declaring victory. Silent success on a poisoned case is the worst outcome; explicit escalation is a pass.
- **Failure-mode bank**: every real-run failure becomes a permanent corpus case. Regressions on past failures are blocked at PR time.

This is the loop that turns the racing MLE-bench curve from a threat into a non-issue: we don't compete on scaffold quality, we compete on a disciplined, regression-tested loop over a substrate nobody else has.

## 9. Interfaces & artifacts (schemas to pin early)

Learned from the migration harness: pin the contracts before writing orchestration code, because retrofitting them after the 4th campaign is painful.

- `ExperimentSpec` — axes, fixed base, eval suite ref, budget, stopping rule. Output of FRAME.
- `Matrix` — list of typed matrix points, each with fidelity schedule and resource estimate. Output of DESIGN.
- `RunManifest` — per-point: job handle, pinned seeds/versions, data-mix spec, config provenance. Written at LAUNCH.
- `MetricsBlob` — typed, append-only, the *only* source the report may cite. Written at EVALUATE.
- `Scorecard` / `RoundReport` — matrix + results + frontier + next-round recommendation + full provenance. Every numeric field references a `MetricsBlob` key.
- `AgentVersionManifest` — composite version (sequencer prompt + each delegated skill version + controller version), so when a campaign result shifts you know which component moved.

## 10. Buyer set & positioning (brief — the pitch doc has the full case)

The value is capability expansion, not cost reduction, so the buyers are the ~20–50 companies doing serious foundation-model / VLA / post-training work at a scale where researcher attention, not compute, is the bottleneck: robotics FM companies (Physical Intelligence, Figure, 1X, Skild), second-tier LLM labs (Cohere, AI21, Mistral, Reka, sovereign plays), bio-FM (Latent Labs, Cradle, EvolutionaryScale), and hyperscaler in-house teams below the top three. Most already run Ray. The pitch is **"Ray Tune on steroids with a researcher's co-pilot that proposes the next experiment and explains why"** — explicitly scaffolding for their researchers, not a replacement, because this buyer set is (rightly) allergic to fire-and-forget research agents.

## 11. Red team (grounded, not hand-wavy)

1. **The buyers may not let an agent near their experiments.** These teams' craft *is* experiment design. Mitigation: position as researcher scaffolding, keep the human on the next-matrix approval gate by default, make provenance auditable so they can trust every number.
2. **Autonomous research reliability is genuinely not there yet.** The [four-attempt study](https://arxiv.org/abs/2601.03315) had a 25% end-to-end success rate and documented overexcitement, drift, and weak taste. Mitigation: the design does not attempt end-to-end autonomy — it automates orchestration and bookkeeping (which are reliable) and gates the reasoning steps (which are not). The human owns taste; the agent owns the grind.
3. **Small TAM.** 20–50 companies is a prestige business, not a horizontal one. This has to be a deliberate strategic choice, not a default.
4. **You compete with the customer's internal platform team.** Mitigation: play well with their existing stack; the pitch is "3× your researchers on top of the infra you already built," never "replace your platform team."
5. **Scaffold commoditization.** Covered above — don't compete there; compete on substrate + discipline + regression-tested loop.

## 12. Build plan

Milestone-gated, cheapest-proof-first, mirroring the migration harness discipline.

- **M0 — One campaign, end to end, sample-sized.** Pick the cleanest real question (a small LLM data-mix sweep on a 7B base). Wire FRAME→REPORT delegating to the existing post-training + Ray Train skills. Parity mode only. Goal: prove the sequencer + gates work and produce a provenance-clean report. This is the "does the loop hold together" proof.
- **M1 — The early-kill controller + multi-fidelity.** Add ASHA-style promotion/kill over a real matrix. Goal: show the agent reclaims compute intelligently, deterministically.
- **M2 — The agent's eval suite.** Replay + one poisoned campaign + failure-mode bank. Goal: make the agent regression-testable, which is what makes it a product.
- **M3 — Second archetype.** Add a GRPO/RLVR reward-search campaign via SkyRL to prove the design generalizes beyond data mixes.
- **M4 — Cross-campaign memory + human approval UX.** The retrieval layer and the next-matrix approval loop.

Don't expand archetypes until the loop can score the current one. Three clean campaigns (data-mix, RL reward, VLA mix) validate the design before scaling.

## 13. Open questions

- **Where's the FRAME/human boundary?** How much of the question formalization is interview vs. inference? Getting this wrong makes it either annoying or reckless.
- **Auto-advance vs. approval-gate as default?** Budget-bounded auto-advance is more magical but scarier for this buyer set. Probably approval-gate default, auto-advance opt-in.
- **How much does the agent get to touch the eval suite?** Defining "better" is where research taste lives — likely human-authored eval suites only, agent runs them but doesn't design them (at least at first).
- **Composite versioning granularity** — lock all delegated skills together (simpler) or per-component manifest (more diagnostic)? Start composite.
- **Does this ship as an Anyscale Agent Skill** (like the post-training skill) or a heavier standalone product? The skill path is faster to a real user and reuses the delivery channel that already exists.

---

## Sources

- [Introducing the Anyscale Agent Skill for LLM Post-Training](https://www.anyscale.com/blog/anyscale-llm-post-training-skill)
- [Post-training for LLMs on Anyscale (docs)](https://docs.anyscale.com/llm/fine-tuning)
- [Optimizing VLA Fine-Tuning Performance with LeRobot on Ray/Anyscale](https://www.anyscale.com/blog/vision-language-action-pipelines-vla-robotics-ray-anyscale)
- [Introducing Vision-Language Reinforcement Learning in SkyRL](https://www.anyscale.com/blog/vision-language-model-reinforcement-learning-skyrl)
- [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [MLE-bench (OpenAI)](https://arxiv.org/pdf/2410.07095)
- [EurekAgent: Agent Environment Engineering for Autonomous Scientific Discovery](https://arxiv.org/pdf/2606.13662)
- [Operand Quant: Single-Agent Architecture for Autonomous ML Engineering](https://arxiv.org/pdf/2510.11694)
- [The AI Scientist v2 (progressive agentic tree search)](https://arxiv.org/pdf/2504.08066)
- [AutoSOTA: End-to-End Automated Research for SOTA Model Discovery](https://arxiv.org/pdf/2604.05550)
- [Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts](https://arxiv.org/abs/2601.03315)
- [Databricks Agent Bricks — DAIS 2026](https://www.databricks.com/blog/agent-bricks-dais-2026)
- [Ray Tune with Anyscale (ASHA / multi-fidelity)](https://www.anyscale.com/product/library/ray-tune)