# CLEANUP — post-campaign handoff (written 2026-07-08 morning)

For the next session: restore the canonical namespace, prune the campaign
debris, and leave a minimal reproducible set. Follow IN ORDER — step 0 is
not optional. Rule of the house: prefer `mv` to archive dirs over deletion;
delete only what this doc marks safe.

## 0. RESTORE FIRST (the canonical namespace is currently EMPTY)

The failed G1 run moved the winning RUN-2 artifacts aside and died before
writing replacements. Before anything else, on any cluster with
/mnt/user_storage:

```bash
BASE=/mnt/user_storage/transaction-fm-v2
mv $BASE/model/full_old_1783493827      $BASE/model/full        # the R2 model (the winner)
mv $BASE/embeddings/full_old_1783493827 $BASE/embeddings/full   # its pooled embeddings (the 0.23-0.26 source)
mv $BASE/downstream/full_old_1783493827 $BASE/downstream/full   # its trio metrics
```
Verify: `model/full/model_config.json` should show periodic_amount/time true,
intra_tx_attention false; `embeddings/full` schema has embedding_last/mean/max.

## 1. Cluster storage ($BASE = /mnt/user_storage/transaction-fm-v2)

KEEP (the reproducible core):
- `raw/full/` — transactions.parquet + splits.json + **benchmark.parquet**
  (THE pinned eval rows; never casually regenerate)
- `tokenized/full/` — current 13-field schema incl. row_id
- `model/full`, `embeddings/full`, `downstream/full` — after step 0
- `embeddings/full_r2_target/` — R2 surprise/masked/single readouts
- `downstream/full_probe/` — probe_metrics_seed*.json (the seed CIs)
- `source/` — the TabFormer CSV cache (saves a download)
- `tensorboard/` — all runs (the blog needs the curves; small)

SAFE TO DELETE after confirming step 0 (superseded/flawed-era):
- `model/full_old_*`, `embeddings/full_old_*`, `downstream/full_old_*`,
  `tokenized/full_old_*` EXCEPT the 1783493827 stamps consumed by step 0
  (older stamps = navy/b128/CoLES-era models; navy is referenced in
  TEARDOWN history but its numbers are ledgered — keep `model/full_old_1783456646`
  (navy) ONLY if you want rerunnable Run-1 comparisons, else delete)
- `embeddings/{full_navy_last,full_navy_mean,full_navy_target,full_cl_target,subset_navy_target}`
  — Run-1-era readouts; all conclusions ledgered
- `downstream/{full_navy_last,full_navy_mean,compare}` — ledgered
- `raw/xl/`, `tokenized/xl*`, `model/xl*`, `embeddings/xl*`, `downstream/xl*`
  — flawed-era 1024 artifacts; the new 1024 act regenerates them
- `ray_results/*` older than the fm_full_seq512_b64x4_lr4e-4_20260707-210815
  run (per-epoch ckpts of dead models; keep the R2 run's for resume/probes)
- The ENTIRE v1 base `/mnt/user_storage/transaction-fm/` (pre-benchmark era)
  — screenshot/export any TB curves you want first

## 2. Repo files (branch geoff/fm_recs_and_fraud)

KEEP — canonical pipeline:
- `src/*` as-is (nvidia_baseline.py + benchmark_downstream.py are the
  protocol; tokenizer/model/pretrain/embed carry all RUN flags)
- `configs/`: smoke, smoke_learned, small, full, xl, xxl (+ full_g1 as the
  parked G1 variant), rtest/ttest (CI)
- jobs — the four canonical entrypoints:
  `job_baseline.yaml` (the gate), `job_full.yaml` (full retrain+eval),
  `job_xl.yaml` (1024 act), `job_probe.yaml` (readout probes, edit --set)
- scripts: all stage scripts + `probe_embeddings.py`, `baseline_repro.py`
- docs: TEARDOWN.md, BLOG_NOTES.md, EXPERIMENT_LOG.md, AUTORESEARCH.md, this file

DELETE (one-off diagnostics; every result they produced is ledgered):
- `job_eval.yaml`, `job_pooling.yaml`, `job_run1_readout.yaml`,
  `job_probe_r2.yaml`, `job_probe_r2_seeds.yaml`, `job_probe_controls.yaml`,
  `job_g1.yaml` (recreate from job_full + full_g1.yaml when G1 is chunk-fixed)
- STALE pre-campaign leftovers: `job_config.yaml`, `job_config_xl.yaml`,
  `job_config_xxl.yaml`, `job_pretrain_xl.yaml` (predate the benchmark era)
- `gemini_ideas.txt` (mined; verdicts in TEARDOWN/commits)
- local `demo_data/` (untracked smoke debris; regenerates in minutes)

## 2b. Clean-repro branch — strip dead flags ENTIRELY (don't just default-off)

Branch strategy: `geoff/fm_recs_and_fraud` stays frozen as the full research
record (flags, dead ends, matches TEARDOWN commit-for-commit). Cut
`geoff/fm_clean_repro` from it and REMOVE (not disable) everything the
campaign falsified or never validated — the clean branch is the minimal
faithful reproduction of the headline, and the 1024/2048 validation runs
from it.

REMOVE on the clean branch:
- Sequence contrastive: `seq_cl_weight` knob, `seq_contrastive_loss`,
  `_contrastive_views`, the seq_views/seq_cl_scale plumbing in
  model.forward + pretrain loop, and the seq_cl TB metrics
  (measured: no downstream lift, taxed the merchant head)
- G1: `intra_tx_attention`/`intra_tx_heads` + the attr-slot/MHA code in
  _embed + configs/full_g1.yaml (never successfully ran; lives on the
  research branch if ever chunk-fixed)
- Legacy whole-row masking support: the `isinstance(masked, dict)` tensor
  branch in field_loss (mask_batch always returns dict now)
- The one-off job yamls listed in §2

KEEP on the clean branch (small, load-bearing, or part of the story):
- periodic_amount / periodic_time / n_periodic (in the winning recipe)
- multi-column pooled extraction (last/mean/max — 5 lines; reviewers will
  ask for the pooling ablation) and TargetReadoutExtractor + surprise
  (Run-1 is part of the published narrative)
- heuristic eval sampling (synthetic/CI path) and src/downstream.py fallback
- pre-campaign template features not on trial (soft amount mode)

Acceptance for the clean branch: unit tests + synthetic smoke pass; then
the 1024 run validates the pipeline end-to-end from clean code.

## 3. Config normalization BEFORE the 1024/2048 act

`xl.yaml`/`xxl.yaml` predate RUN-2b. Propagate into both:
- model block: `periodic_amount: true`, `periodic_time: true`,
  `n_periodic: 16`, `intra_tx_attention: false`
- pretrain block: `seq_cl_weight: 0.0`
- (new tokenizer fields arrive automatically; batch sizes already
  seq-adjusted: xl 64, xxl 16 — do NOT bump, update-count rule)
Then the act is: `job_xl.yaml` as-is (moves aside, retrains, trio + target
extraction), then a probe with `--set xl_pooled_last=...:embedding_last
--raw-control --seed N`. Same for a job_xxl clone if 1024 pays.

## 4. Open items (from TEARDOWN, ordered) — WITH TIMING

1. Shuffled-label / shuffled-embedding sanity probe — kick IMMEDIATELY
   after step 0 (restore): ~$2, 15 min, needs only benchmark + restored
   embeddings; independent of the clean branch
2. Classical burst-aggregates baseline (XGB on 13 raw + card-velocity
   features) — kick IN PARALLEL with #1, BEFORE any blog drafting and
   before/alongside the 1024 run (it calibrates how impressed to be);
   CPU-only, needs only raw parquet + benchmark.parquet, no FM artifacts
   Suggested order overall: step 0 restore -> kick #1+#2 -> build clean
   branch (§2b) + normalize xl config (§3) -> launch 1024 from the clean
   branch while the controls finish
3. Cutoff unification: splits.json quantile -> nvidia_baseline date cutoff
   (removes the 1,394-row val impurity); then ONE clean publication run
4. Surprise ⊕ pooled-embedding fusion (untested, plausible small win)
5. G1 chunked-attention fix (parked); reco rework (parked)
6. Seed-replicate anything new per AUTORESEARCH rules (3+ seeds, CIs)
