# Campaign 07 — Chronos time-series FM: reproduce zero-shot Benchmark II, then beat it

> **Status: SEED (not started).** A foundation-model-in-a-new-domain campaign with a cheap,
> inference-bound gate and a naturally Ray-parallel eval (27 datasets). Wave-1 for the
> reproduction, Wave-2 for a fine-tune push.

## 1. One-liner

Reproduce Chronos-T5's published zero-shot forecasting accuracy (WQL/MASE) on its 27-dataset
benchmark, then beat *original Chronos-T5* via fine-tuning or a better readout — on the only
generation with a fully shipped, runnable gate.

## 2. Why this is a good autoresearch testbed

- **What makes it clean:** ships train + eval + committed results CSVs + checkpoints (full
  reproduction surface); fine-tune and decoding are concrete levers; classical baselines make a
  strong honest floor to clear.
- **Ray substrate it exercises:** eval is embarrassingly parallel across 27 datasets → Ray Data /
  Ray Tasks fan-out; batch inference over many series; Ray Train for the fine-tune push.

## 3. Reference

- **Repo:** `github.com/amazon-science/chronos-forecasting` — **Apache-2.0**.
- **Ships:** **original Chronos-T5 = train + eval + committed `results/*.csv` + checkpoints**
  (full reproduction surface). **Chronos-Bolt and Chronos-2 are inference-only — no training
  code** (a hard wall if "beat" means retraining them). Anchor the campaign on **Chronos-T5**.
- **Ray-native?** No — PyTorch + HF Transformers T5, `torchrun` DDP; eval is plain Python.
- **Reproduction landscape:** widely reproduced; gaps trace to single-vs-3-run checkpoints and
  GluonTS version drift. **Contested claim = leakage** (GIFT-Eval documents partial pretraining
  leakage for Chronos/TimesFM/Moirai). Original T5 is no longer SOTA (TiRex, TimesFM-2.5,
  Chronos-2 top leaderboards) — honest framing: reproduce-then-beat *original Chronos-T5*.

## 4. The gate

- **Decision metric:** zero-shot aggregated relative **WQL and MASE** on **Benchmark II (27
  unseen datasets)**, geometric-mean-normalized to Seasonal-Naive.
- **Published to match (the committed `results/*.csv`, NOT the paper figure):** Chronos-T5-Large
  ≈ **WQL 0.93 / MASE 0.94** (Fig. 5 is approximate; the CSVs are authoritative). Caveat: only 1
  checkpoint/size released vs the paper's 3-run averages → irreducible variance (discussion #120).
- **Guard metrics:** per-dataset WQL/MASE, inference throughput.
- **Artifact gate:** run `scripts/evaluation/evaluate.py` on the shipped checkpoint → match the
  committed CSV. **Pipeline gate:** reproduce it through the Ray-parallelized eval harness.

## 5. Pinned eval

The Benchmark II dataset list + the harness's dataset versions (auto-pulled from
`huggingface.co/datasets/autogluon/chronos_datasets`) + the context/prediction lengths +
sampling params + the WQL/MASE + Seasonal-Naive-normalization implementation. Pin the GluonTS
version (drift is a known repro gap). Report CIs — the single-checkpoint variance is real.

## 6. Data

- **Eval:** curated at `autogluon/chronos_datasets`, pulled by the harness → **low friction**;
  per-dataset licenses vary. **Train (for the fine-tune push):** TSMixup + KernelSynth (synthetic,
  scriptable). All effectively open.
- **Input audit:** context length, scaling/normalization per series, the tokenization (Chronos
  quantizes values into a vocab — the quantization bin edges are load-bearing), missing-value
  handling.

## 7. Rayification

| Stage | What | Ray lib |
|---|---|---|
| eval fan-out | forecast + score across 27 datasets | Ray Data / Ray Tasks (parallel) |
| batch forecast | inference over many series | Ray Data batch inference (scale-to-zero) |
| fine-tune (push) | continue-train T5 on a domain mix | Ray Train (`TorchTrainer`) |

## 8. Fidelity ladder

- **smoke:** one small dataset, chronos-t5-tiny, 1 GPU/CPU — the forecast+score loop runs.
- **proxy:** a subset of Benchmark II with tiny/small checkpoints; rank fine-tune / readout ideas.
- **full:** all 27 datasets at T5-Large (the pinned gate), then the fine-tune push.
- **Proxy axis:** model size (tiny→large) + dataset subset. **Rare-signal trap:** datasets differ
  wildly in seasonality/scale; a subset win may not hold — calibrate the subset ranks the full
  suite; report relative-to-Seasonal-Naive, never absolute errors.

## 9. "Beat it" hypotheses

1. **Domain fine-tune beats zero-shot** — continue-train Chronos-T5 on a target domain (retail
   or energy) and beat its own zero-shot number on that domain's datasets. Flag `finetune`.
2. **Readout/decoding** — the readout thesis for forecasting: does a different sampling/decoding
   or quantization scheme improve WQL at fixed weights? Flag `decode`.
3. **Leakage-honest subset** — following the GIFT-Eval critique, report on a subset with
   documented no-leakage; is the zero-shot advantage stable there? (a control that doubles as a
   claim). Flag `clean_subset`.

## 10. Budget

- **Wave 1** for the zero-shot reproduction (inference-only, cheap); **Wave 2** for the
  fine-tune push.
- **GPU-hours:** smoke <0.5 · reproduction (all 27 datasets, T5-Large, inference) ~3–8 · proxy
  (fine-tune/readout ideas at small scale) ~10 · full fine-tune push ~15–30 · **total ~30–50**.
- **GPU tier:** A10G/L4 for eval, A100 for the 8-GPU fine-tune; spot ON. **~$40–120 on-demand.**
- **Approval:** envelope (Wave 1); + each fine-tune full run (Wave 2).

## 11. Controls

- **Seasonal-Naive and a classical baseline (AutoETS/AutoARIMA via StatsForecast)** — the honest
  cheap floors the FM must clear (classical baselines routinely beat neural forecasters under
  fair eval — the same lesson as the FM reco frequency baseline).
- **Leakage guard:** report the leakage-honest subset explicitly.
- **Leak-audit trigger:** zero-shot WQL implausibly below the committed CSV → dataset-version or
  normalization mismatch.

## 12. Risks

Bolt/Chronos-2 have no training code — anchoring on T5 is deliberate (say so). Single-checkpoint
variance vs 3-run paper averages (report CIs, don't chase the point). GluonTS/version drift.
Original T5 is legacy SOTA — frame honestly as reproduce-then-beat-Chronos-T5, with the modern
leaderboard (GIFT-Eval) as context, not a claim to top it.
