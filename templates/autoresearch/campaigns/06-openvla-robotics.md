# Campaign 06 — OpenVLA robotics: reproduce LIBERO, then attack the robustness gap

> **Status: SEED (not started).** The VLA/robotics campaign (`AUTORESEARCH.md`'s "VLA on a
> manipulation suite" case). The heaviest and softest-gated seed — its thesis is deliberately
> **not** "beat 76.5%" (commoditized) but "the headline overstates robustness."

## 1. One-liner

Reproduce OpenVLA's published LIBERO success rate from its shipped checkpoints, then quantify
and attack the memorization/robustness gap the 2025 literature exposed — the defensible "beat"
now that raw success rate is saturated.

## 2. Why this is a good autoresearch testbed

- **What makes it clean:** shipped checkpoints → eval without fine-tuning (cheap gate); the
  robustness-gap finding is a well-supported, defensible thesis.
- **Ray substrate it exercises:** Ray Train (FSDP/LoRA) for the 7B fine-tune, and — the harder
  need — a decoupled CPU-sim-fleet → batched-GPU-policy-actor design for parallel rollout eval
  (see §7), where sim rollout otherwise burns the wall-clock.

## 3. Reference

- **Repo:** `github.com/openvla/openvla` — code **MIT**. **⚠️ Weights carry the Llama-2
  Community License** (>700M-MAU clause), not MIT — flag for any commercial use.
- **Ships:** train, LoRA fine-tune, and the LIBERO eval harness
  (`experiments/robot/libero/run_libero_eval.py`); `openvla/openvla-7b` + four LIBERO fine-tunes.
  **Shipped checkpoints → eval without any fine-tune (the cheap gate).**
- **Ray-native?** No — PyTorch FSDP via `torchrun`; LoRA single-GPU; LIBERO sim (robosuite +
  MuJoCo) is the main env risk.
- **Reproduction landscape:** broadly reproduced but **the gate is soft** — repo issue #282
  reports 68% vs published 88.4% on LIBERO-Object (unresolved); LIBERO-PRO (arXiv:2510.03827)
  and LIBERO-Plus (arXiv:2510.13626) show >90% scores collapse toward 0 under position/phrasing
  shifts (memorization critique); and real SOTA moved to ~97% (OpenVLA-OFT). **Beating the 2024
  autoregressive 76.5% is already commoditized — do not make that the thesis.**

## 4. The gate

- **Decision metric:** LIBERO success rate (fraction of successful rollouts).
- **Published to match (paper App. E + README, confirm by running the harness):** Spatial 84.7 /
  Object 88.4 / Goal 79.2 / Long 53.7 / **avg 76.5%**. Treat **Object as fragile** (the #282 gap).
- **Guard metrics:** per-suite breakdown (never just the mean), rollout count, seed variance.
- **Artifact gate:** run `run_libero_eval.py` on the shipped fine-tuned checkpoints → match the
  per-suite table. This is the primary gate (no training required).
- **Pipeline gate:** reproduce a LoRA fine-tune from `openvla-7b` → recover a suite's number.

## 5. Pinned eval

The LIBERO task/episode list, the initial-state seeds, the rollout budget (they use 500
rollouts/suite), the success predicate, and the MuJoCo/robosuite versions — **all frozen**.
Success rates on few episodes have huge CIs → fix episode seeds, report per-task breakdown and a
CI over rollouts (`AUTORESEARCH.md` §3 VLA row).

## 6. Data

- **Fine-tune:** `openvla/modified_libero_rlds` on HF (~10 GB, public) — must use *their*
  "modified" RLDS build to match numbers. **Eval:** the LIBERO simulator (installed separately).
- **Pretraining data** (Open X-Embodiment, multi-TB, heterogeneous licenses) is **NOT needed**
  for the gate — skip it entirely.
- **Input audit:** observation space (camera views, proprioception), action space + normalization
  (the VLA analog of the field audit — a wrong action-unnormalization silently zeros success),
  the instruction-phrasing distribution (central to the robustness critique).

## 7. Rayification

| Stage | What | Ray lib |
|---|---|---|
| fine-tune (optional) | LoRA on 7B VLA | Ray Train (FSDP/LoRA) |
| **rollout eval** | N episodes × M suites in parallel MuJoCo workers | **CPU sim fleet → batched GPU policy actors (decoupled — see below)** |
| robustness eval | LIBERO-PRO/Plus perturbations, parallel | same decoupled fleet |
| aggregate | success-rate table + CIs | driver |

> **The rollout is not "embarrassingly parallel" if you put a 7B model in every worker.** A
> LIBERO rollout is a loop: sim step (CPU) → policy inference (7B, GPU) → sim step → … If each
> of 500 parallel workers owns a GPU-resident 7B model, the GPUs sit ~idle waiting on CPU sim
> steps and the bill explodes. The correct architecture — and the actually novel Anyscale
> story — is **decoupling: a large CPU fleet of MuJoCo sim workers streaming observations to a
> small pool of *batched* GPU inference actors** (continuous batching / fractional-GPU packing,
> the same instincts as the embed campaigns pointed at robotics). The ~10–20 GPU-hr eval-gate
> budget below is only achievable under this design; a GPU-per-worker layout would be
> multiples more.

## 8. Fidelity ladder

- **smoke:** a few episodes of one suite on the shipped checkpoint, 1 GPU — the sim+eval loop
  runs end-to-end (this is where MuJoCo setup bugs surface).
- **proxy:** one suite (Spatial), full rollout budget; validate we reproduce that suite's number
  and rank any robustness intervention.
- **full:** all four suites + the LIBERO-PRO/Plus robustness sweep.
- **Proxy axis:** suite subset + rollout count + (eval-only vs fine-tune). **Rare-signal trap:**
  low rollout counts have huge CIs — don't rank interventions on 20 episodes; and per-suite
  behavior differs, so a Spatial win may not transfer to Long.

## 9. "Beat it" hypotheses (thesis = robustness, not raw success)

1. **Quantify the memorization gap** — reproduce 76.5% on stock LIBERO, then report the drop
   under LIBERO-PRO/Plus perturbations. The finding IS the contribution (echoes the FM campaign's
   "the benchmark can't answer the question — upgrade the benchmark"). Flag `robustness_eval`.
2. **A cheap intervention that narrows the gap** — e.g. augmentation of initial states / phrasing
   during LoRA fine-tune; does it recover robustness at fixed stock success? Flag `aug_finetune`.
3. **Readout thesis for actions** — action-head / decoding variant on the frozen VLM trunk;
   does it help stock or robust success? Flag `action_head`.

## 10. Budget

- **Wave 3** (heaviest seed; multi-GPU fine-tune + large sim fleet).
- **GPU-hours:** smoke ~1 · eval-only gate (4 suites × 500 rollouts) ~10–20 · proxy interventions
  ~20–40 · full (LoRA fine-tune ~10–15h/suite × suites + robustness sweep) ~80–150 · **total
  ~120–220**.
- **GPU tier:** A100 (7B FSDP/LoRA + sim); spot ON with checkpointing. **~$400–800 on-demand;
  multi-node possible → PI approval.**
- **Approval:** envelope + each full run + multi-node.

## 11. Controls

- **Random-action / no-op baseline** → success ≈ 0 (sanity that the harness scores correctly).
- **Shipped-checkpoint reproduction** is itself the control for our fine-tune pipeline.
- **Leak-audit trigger:** success materially above the published per-suite number → an eval
  harness bug (wrong success predicate, replayed init states).

## 12. Risks

MuJoCo/robosuite install + version pinning is the #1 time sink (pin the whole sim env). The
#282 Object gap means budget forensic time to reconcile. Llama-2 weight license limits
commercial framing. Because raw success is commoditized (OFT ~97%), a naive "we beat 76.5%"
reads as stale — the robustness thesis is what makes this publishable. Highest-cost, softest-gate
seed: do it only after a couple of Wave-1 wins prove the harness.
