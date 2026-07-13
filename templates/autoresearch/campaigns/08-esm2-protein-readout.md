# Campaign 08 — ESM-2 protein embeddings + FLIP readout ladder

> **Status: SEED (not started).** The **purest test of the readout thesis** in a new domain,
> and the cheapest gate in the whole program — the recommended first campaign to shake out the
> harness. Directly mirrors the transaction-FM frozen-embedding → probe-ladder pattern.

## 1. One-liner

Reproduce a published FLIP protein-property Spearman using frozen ESM-2 embeddings + a probe,
then run the readout ladder (linear → MLP → attention-pooled) to show — as in the FM campaign —
that the readout, not the representation, decides the number.

## 2. Why this is a good autoresearch testbed

- **What makes it clean:** frozen ESM-2 + a probe is the purest test of the readout thesis and the
  cheapest gate in the set; the FLIP literature already shows a large readout swing to reproduce
  and interrogate.
- **Ray substrate it exercises:** the textbook Ray Data batch-embedding pattern — a stateless map
  over a frozen encoder, an actor pool across GPUs, write Parquet once, then sweep trivial CPU
  probes (extract-once / sweep-many).

## 3. Reference

- **Repo:** ESM-2 `github.com/facebookresearch/esm` — **MIT** (but **archived**; use HF
  `transformers` `EsmModel`). FLIP `github.com/J-SNACKKB/FLIP` — AFL-3.0.
  **⚠️ ESM-3/ESM-C are non-commercial (Cambrian license) — do this on ESM-2 (MIT), not ESM-3.**
- **Ships:** ESM-2 checkpoints all 6 sizes (`facebook/esm2_t33_650M_UR50D` etc.) + extractor;
  FLIP ships baseline train + eval + regenerated `*_results.csv`.
- **Ray-native?** No — plain PyTorch scripts. The batch-embed rayification is pure upside.
- **Reproduction landscape:** reproduced many times (Kermut, Schmirler et al.). **Readout/pooling
  swings the number hugely** (GB1 0.79 attention1d → 0.54 mean-pool) — which is exactly the
  thesis. FLIP2 (2026) exists — cite it so we're reproducing FLIP-v1 on purpose.

## 4. The gate

- **Decision metric:** Spearman ρ on a named FLIP split — recommended **GB1 `three_vs_rest`**.
- **Published to match (FLIP paper bioRxiv 2021.11.09.467890 + regenerated CSVs; no README
  table):** ESM baseline ρ ≈ **0.79** (ESM-1b, attention1d readout); mean-pool ≈ 0.54. **Fix the
  readout to compare like-for-like** — the readout is the confound and the subject.
- **Guard metrics:** ρ on other FLIP splits (AAV, Meltome), calibration.
- **Artifact gate:** reproduce the FLIP baseline CSV with the shipped code. **Pipeline gate:**
  reproduce ρ through the Ray-embed + probe harness (embed with ESM-2, probe separately).

## 5. Pinned eval

The exact FLIP split file (GB1 `three_vs_rest`), the Spearman implementation, fixed seed, and —
**most important — the pinned readout** (attention1d vs mean vs linear), since it dominates.
Extrapolation splits are high-variance → report ≥3 seeds with CIs (`AUTORESEARCH.md` §9.1).

## 6. Data

- **FLIP splits** — CSV `splits.zip` in-repo, FASTA at data.bioembeddings.com — **low friction**,
  AFL-3.0. GB1 ~150K seqs, AAV ~250K, Meltome ~48K (long seqs dominate embedding cost).
- **Input audit:** sequence length truncation (ESM-2 has a max context — long proteins get cut,
  a silent signal loss), the exact ESM-2 size used (8M→15B change ρ), which hidden layer's
  representation is read (mid vs last).

## 7. Rayification

| Stage | What | Ray lib |
|---|---|---|
| ingest | FASTA → sequences | Ray Data |
| **embed** | frozen ESM-2 forward, cache per-residue + pooled reps | **Ray Data `map_batches` + ActorPoolStrategy** (0.5–1 GPU/replica by size) |
| readout/probe | ridge / MLP / attention1d on frozen reps | driver / Ray tasks (trivial) |
| eval | Spearman per split | driver |

## 8. Fidelity ladder

- **smoke:** a few dozen sequences, esm2-8M, CPU/1-GPU — embed→probe→score runs.
- **proxy:** GB1 with a small ESM-2 (150M); rank readout variants (this is where the ladder is
  cheap to sweep).
- **full:** GB1 `three_vs_rest` with esm2-650M (the gate), + AAV/Meltome for breadth.
- **Proxy axis:** ESM-2 size + split. **Rare-signal trap:** extrapolation splits (few training
  mutations → held-out) are high variance — fix seeds, report CIs; don't rank readouts on a
  single draw.

## 9. "Beat it" hypotheses

1. **Readout ladder (primary, the thesis)** — mean → linear → MLP → attention-pooled → per-residue
   attention1d on the *same* frozen ESM-2 embedding. Show the ρ swing (expected 0.54→0.79+) with
   zero change to the representation — the FM campaign's 7× reco swing, replicated in a new
   domain. Flag `readout`.
2. **Layer selection** — which ESM-2 layer's representation probes best (mid layers often beat
   the last for property prediction)? Flag `layer`.
3. **Frozen vs light fine-tune honesty** — note fine-tuning beats frozen broadly, so frame the
   frozen ladder as a *deliberately constrained* claim; optionally measure the frozen-vs-FT gap.

## 10. Budget

- **Wave 1** — the cheapest campaign in the program.
- **GPU-hours:** smoke <0.2 · embed GB1 with 650M (cached once) ~1–3 · proxy readout sweep
  (probes are ~free) ~2 · full (650M embed + AAV/Meltome breadth) ~5–10 · **total ~10–15**.
- **GPU tier:** A10G/L4; spot ON. **~$15 on-demand / ~$6 spot.**
- **Approval:** envelope only.

## 11. Controls

- **Shuffled-label control** → ρ collapses to ~0.
- **Strongest cheap baseline:** one-hot / k-mer features + ridge (the honest non-FM floor — ESM
  must clear it convincingly; for some FLIP splits simple baselines are surprisingly strong).
- **Leak-audit trigger:** ρ above the published attention1d number from mean-pool → a split
  leakage or a readout mislabel.

## 12. Risks

Readout is a confound *and* the subject — pin it explicitly per comparison. Extrapolation-split
variance (CIs mandatory). ESM-2 sequence-length truncation on long proteins. Use ESM-2 (MIT),
not the non-commercial ESM-3. This campaign's value is as much *methodological* (proving the
readout thesis generalizes) as it is a life-sciences result.
