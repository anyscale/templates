# Campaign NN — <title>

> Copy this file to `campaigns/NN-<slug>.md` and fill every field. This is the
> **pre-registration** of the campaign: hypotheses, decision metric, eval pin, and budget
> are committed *before* the first paid run. Unfilled fields block the budget approval.
> **After the first paid run the plan is frozen** — amendments are new commits carrying a
> visible `AMENDED (date, why)` note, never a rewrite, and registry rows reference the
> seed-plan commit SHA they ran under (`REQUIREMENTS.md` R1). Calling your shot after the
> ball drops isn't a prediction.

## 1. One-liner

<What published result do we reproduce, and what's the "beat it" thesis in one sentence?>

## 2. Why this is a good autoresearch testbed

- **What makes it clean:** <shipped code + eval + (ideally) checkpoints so a baseline is
  establishable; a real reproduction gate; a portable improvement thesis the loop can act on>
- **Ray substrate it exercises:** <which distributed-compute primitives the experiment loop
  needs here — Ray Data streaming/reweighting or batch-embed, Ray Train parallel matrix runs,
  Ray Tune ASHA multi-fidelity early-kill, Ray Serve eval rollouts. This is about what makes
  running the matrix affordable and observable, not a business angle.>

## 3. Reference (the thing we reproduce)

- **Repo:** <URL> — **license:** <...>
- **Paper:** <URL/citation>
- **Ships:** training code? eval code? pretrained checkpoints? <yes/no each>
- **Ray-native?** <Ray / torchrun / accelerate / slurm — drives how much rayification is real work>
- **Reproduction landscape:** <independently reproduced? known critiques / contested claims?>
- **`commercial_use`:** <`yes` / `no` / `needs-legal`> — can the reference's **code AND
  weights AND every dataset license** support this campaign's *intended use*? (Only matters if
  the use is beyond private research.) Check each separately — weights often carry a different
  license than code (e.g. Llama-2 community clause), and datasets are often the strictest
  (CC-BY-NC is non-commercial). A `no`/`needs-legal` is a **budget gate**
  (`REQUIREMENTS.md` non-functional #7): no spend until the license risk is signed off.

## 4. The gate (Iron Rule #1 — reproduce from CODE, before any model work)

- **Decision metric:** <AP / pass@1 / nDCG@10 / mIoU / success-rate / WER / Spearman ...>
- **Published number to match:** <value> — **defined in:** <eval script path in the repo>
  (target to *confirm from code*, not trusted from the paper).
- **Guard metrics:** <the everything-else that must not silently regress>
- **Artifact gate:** run their shipped eval on their shipped checkpoint → match their number.
- **Pipeline gate:** reproduce that number through our Ray harness (one protocol module).

## 5. Pinned eval artifact

- **What's frozen:** <eval rows / prompt set + decoding params / episode list / tile set>
- **Where:** `$BASE/<campaign>/eval/...` — content-hashed; the hash is the registry `eval_pin`.
- **Positives count** (drives CI plan): <N> — <point estimates allowed? or CI-mandatory?>
- **Env pin:** <library versions + device that the whole table is scored under>

## 6. Data

- **Dataset(s):** <name, size, license, how to obtain, access friction>
- **Where it lives:** `$BASE/<campaign>/raw/...` (durable, shared by workspace + jobs)
- **Input audit (Iron Rule #10):** <field/channel/token — ours vs theirs vs literature.
  What could we silently drop the way the FM campaign deleted per-event geography?>

## 7. Rayification — stage → Ray library

| Stage | What it does | Ray lib | Scale knob | Nearest opt-guide archetype |
|---|---|---|---|---|
| ingest | | Ray Data | | `workloads/...` |
| preprocess/tokenize | | Ray Data | | |
| train/pretrain | | Ray Train | | |
| embed/extract | | Ray Data | | |
| downstream/readout | | | | |
| eval | | | | |
| serve (optional) | | Ray Serve | | |

## 8. Fidelity ladder

| Rung | What it is here | May decide |
|---|---|---|
| smoke | <tiny/synthetic, CPU/1-GPU> | code runs — nothing about quality |
| **proxy** | <the reduced-fidelity axis — see below> | rank ideas; kill losers |
| full | <the pinned benchmark> | publishable claims |

- **Proxy axis (domain work):** <reduce the axis that scales cost while preserving the
  mechanism the hypotheses act on — subsample entities / smaller base model / sim-before-real
  / geographic subsample / shorter horizon>
- **Rare-signal trap to protect:** <the thing a naive subsample destroys — keep all
  positive-bearing entities / hold out whole sites / fix episode seeds / stratify by activity>
- **Proxy calibration plan:** <which known variants to replay to prove the proxy ranks right>

## 9. "Beat it" hypotheses (the backlog — each: mechanism + cheapest validating run + rung)

1. **<hypothesis>** — mechanism: <why it should move the metric>; validating run: <cheapest>;
   rung: <proxy/full>; flag: `<name>` (default OFF).
2. ...

<Carry the portable theses where they apply: the **readout thesis** (sweep readouts on the
frozen representation before judging it — a 7× swing is possible with zero pretraining
change); **context-length is a config knob, find the sweet spot on your data**;
**late→joint fusion on one substrate**.>

> **Mechanism check before porting a thesis (mandatory).** For each ported hypothesis, write
> one line: *why would this mechanism hold HERE?* The FM readout thesis won because those
> embeddings were pretrained generically with no prescribed readout — a free win waiting.
> A representation trained end-to-end *through* a specific readout (e.g. a contrastive
> retriever trained through mean-pooling) has that readout baked into its weights; swapping it
> on frozen weights should be expected to *hurt*. Any thesis that fails the mechanism check
> gets ONE cheap probe (`AUTORESEARCH.md` §3 "probe before sweep") before it earns a sweep.

> **If a hypothesis is a cost/efficiency beat, say so explicitly and name the baseline.**
> When the reference published no cost number (usual for throughput/$-per-unit claims), you
> are not "beating a published number" — you're beating a baseline you build. State it as a
> competent single-GPU implementation on the same hardware budget, reasonably tuned, with the
> quality metric held equal within CI (`REQUIREMENTS.md` non-functional #8). Never let it
> read as beating the paper.

## 10. Budget envelope

- **Wave:** <1 / 2 / 3> (per `BUDGET_POLICY.md`)
- **Estimated GPU-hours (raw):** smoke <..> · proxy sweep <..> · full runs <..> · controls/eval
  <..> · **total <..>** (±50%, pre-calibration)
- **A10G-equivalent total:** <raw × tier rel-cost> — **this is what sets the wave** (T4 ×0.5,
  L4 ×0.8, A10G ×1.0, L40S ×1.8, A100 ×3.5, H100 ×5.5). A campaign on A100/H100 hits its wave
  ceiling at far fewer raw hours than one on A10G.
- **GPU tier:** <T4/L4/A10G/L40S/A100/H100> — **spot?** <yes/no>
- **Rough $:** on-demand <..> / spot <..>
- **Approval needed:** <envelope; each full run; multi-node?>

## 11. Controls & rigor (before any "beat it" claim ships)

- **Shuffled-label / permutation control:** <collapses to what floor?>
- **Strongest cheap baseline** (not the weakest quoted): <what is it here?>
- **Faithful-replication branch?** <run one in parallel as the strongest ablation?>
- **Leak-audit trigger:** <what "too good" signature launches the leak-audit agent?>

## 12. Known risks / repro gotchas

<Dataset access friction, license limits, contested claims, substrate footguns from
`AUTORESEARCH.md` §5 / `CLAUDE_WITH_ANYSCALE.md` §6, host-RAM budgets, near-saturated
benchmarks where beating the number is easy but unpersuasive.>
