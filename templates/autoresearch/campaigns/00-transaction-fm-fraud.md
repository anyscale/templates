# Campaign 00 — Transaction foundation model vs NVIDIA TabFormer fraud (WORKED EXAMPLE, DONE)

> **Status: COMPLETE — this is the campaign the whole program is distilled from.** Included
> as the reference exemplar so every field of the seed schema has a real, finished answer.
> Full history: `../../fintech_transaction_fm/{TEARDOWN,EXPERIMENT_LOG,BLOG_NOTES}.md` and the
> RUN-tagged commit series on `geoff/fm_recs_and_fraud`; cookbook on `geoff/fm_clean_repro`.

## 1. One-liner

Reproduce NVIDIA's TabFormer transaction-FM fraud benchmark, then beat its published fusion
headline (AP 0.1755) with a sequential encoder read the right way — **won: embedding-alone
AP 0.23–0.27 at ROC ~0.997**.

## 2. Why this is a good autoresearch testbed

- **What makes it clean:** ships training + eval + a checkpoint, so a baseline is establishable
  both ways (artifact + pipeline gate); a real published number to reproduce; the readout thesis
  is a portable improvement lever the loop can act on. This is the worked example the whole loop
  is distilled from.
- **Ray substrate it exercises:** heterogeneous Ray Data batch-embedding (CPU-read + GPU-infer,
  scale-to-zero), Ray Train DDP for pretraining, Ray Serve for the downstream. The embedding pass
  is the recurring cost, so making it cheap and observable is what the substrate buys here.

## 3. Reference

- **Repo:** github.com/NVIDIA-AI-Blueprints/transaction-foundation-model + IBM/TabFormer —
  Apache-2.0.
- **Ships:** training code ✓, eval notebooks ✓, checkpoint ✓ (3,000 steps on 8×A100).
- **Ray-native?** No — RAPIDS/single-GPU. Rayification was the work.
- **Reproduction landscape:** the decisive finding came from the CODE — their "FM embedding"
  is single-transaction (MAX_LENGTH tokenized one row/txn, zero history), contradicting their
  own blog. Iron Rule #1 in action.

## 4. The gate

- **Decision metric:** Average Precision (AP / PR-AUC); AUC saturates at 0.12% fraud.
- **Published:** 0.9885 ROC / 0.1238 AP (blog), 0.1424 (notebook). **Reproduced 0.9875 /
  0.1421 five times byte-stable** via `src/nvidia_baseline.py` (the one protocol module).
- **Both gates green** before any model work: artifact gate on their checkpoint, pipeline
  gate through our harness.

## 5. Pinned eval

`benchmark.parquet` — seed 42, stable sort, written once, 1M balanced train / 100k stratified
val+test. **Only 112 test frauds → bootstrap CIs mandatory** (two false narratives were drafted
from point estimates before CIs existed). Env pin: xgboost==3.2.0 + CUDA (a version/device flip
moved a headline 0.05 AP). Upgraded to a 2.44M-row full-period eval (2,724 frauds) when 112
couldn't rank models — two tables, never compared across.

## 6. Data

TabFormer 24.4M synthetic card txns, ~0.12% fraud, Apache-2.0, via Git LFS or IBM Box mirror.
**Input audit found the core botch:** `_normalize` had deleted User/Card identity and all
geography — fraud clusters by user, so this cratered the baseline. No training could fix it.

## 7. Rayification

ingest→tokenize (Ray Data `map_groups`) → pretrain (Ray Train `TorchTrainer` DDP, ~29M params,
512d/8L/8H) → batch-embed (Ray Data `ActorPoolStrategy`, CPU-read + GPU-infer, scale-to-8-GPU) →
downstream XGBoost (single-node + a Ray Train `XGBoostTrainer` scale-out variant) → reco →
Ray Serve. One config file per rung, zero inheritance.

## 8. Fidelity ladder

smoke (`smoke.yaml`, CPU, ~2k cards) → proxy (`small.yaml`, 2×T4, all cards, 256d) → full
(`full`/`xl`/`xxl` = seq 512/1024/2048, 4×A10G train + 8×A10G embed). **Proxy axis:** subsample
cards, keep sequences whole; keep all fraud-bearing cards, subsample negative-only. Context is a
config knob — the curve **peaks at 1024** (paired bootstrap: 1024 > 512 > 2048), tracking fraud
burst structure.

## 9. Hypotheses that won

- **Readout thesis** — the frozen embedding read at the last position, no PCA, beat their PCA+XGB
  fusion embedding-alone. Same representation, readout swept: masked 0.077 → InfoNCE 0.184 →
  linear 0.397 → MLP 0.535 on the reco task (7× swing, zero pretraining change).
- **Context length** — 4–13× their ~315-txn readable window at identical capacity, one position
  per txn instead of ~12 tokens.

## 10. Budget (actuals)

Wave 1-ish. ~15 jobs; hero full+xl pair ~2h each on 4×A10G train / 8×A10G embed; **~$4–6/run
spot**, program total **~$50–150**. Eval-only reruns cost minutes with zero retraining.

## 11. Controls (all passed)

- Shuffled-label → AP collapses to 0.0016–0.0023 (prevalence floor). No harness leak.
- Velocity-feature baseline (strongest cheap classical) → AP 0.076, *below* raw — feature eng
  does NOT recover the FM lift.
- Faithful third-party replication (Zach's branch, `ZGARNER_INTEGRATION.md`) — their design
  trained by us landed 4× below ours: the strongest ablation and the env-forensics source.

## 12. Gotchas paid for

`RunConfig.name` reuse auto-resumes (0-epoch runs); `iter_torch_batches(dtypes)` must cover every
column; batch-size bumps are model changes (halved optimizer steps degraded everything); a
(card,minute) join collision dropped fraud-burst rows; host-RAM OOM on the no-PCA eval path;
macOS libomp segfault at the XGBoost stage. All in `AUTORESEARCH.md` §5 / §9.
