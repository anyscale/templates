# Campaign 02 — Zero-RL reasoning (SimpleRL-Zoo / verl) reproduction, then extend

> **Status: SEED (not started).** The LLM post-training vertical — Anyscale's heaviest
> investment (4 dedicated workloads). verl is Ray-native, so this is the most substrate-natural
> campaign in the program.

## 1. One-liner

Reproduce a published zero-RL result — GRPO from a base model lifting GSM8K/MATH accuracy —
using the Ray-native verl stack, then test whether a cheaper/faster recipe holds the gain.

## 2. Vertical & why Anyscale

- **Vertical:** AI-natives / agentic reasoning (LLM post-training).
- **Why Anyscale:** the `workloads/post-training` archetype is *already* a "clone a public
  GRPO repo and rayify" template; verl uses **Ray natively** (PPO Ray Trainer, vLLM rollouts
  inside training). Echoes Attentive (5× faster training, 99% cost cut). The story: one Ray
  cluster co-locating rollout generation (vLLM) + reward scoring + policy training with
  per-component autoscaling — the thing torchrun-in-a-notebook cannot do.

## 3. Reference

- **Primary (training gate):** github.com/hkust-nlp/simpleRL-reason — **MIT**. Built on verl
  (GRPO + Ray + vLLM). **Ships released checkpoints for all 10 models** → a real artifact gate.
- **Framework:** github.com/volcengine/verl — Apache-2.0, Ray-native; its docs baseline page
  links per-run logs (the pinned protocol).
- **Eval protocol:** borrow github.com/huggingface/open-r1's lighteval settings (AIME24 ×64,
  MATH-500 ×4, GPQA-D ×8) regardless of what we train — the tightest eval pin available.
- **Micro-proxy:** github.com/Jiayi-Pan/TinyZero (Countdown, <$30, 2×A100) — Ray/verl-native,
  but **no formal numeric gate** and its "aha moment" claim is contested (oat-zero,
  arxiv 2503.20783: self-reflection present at epoch 0, length-increase is a reward artifact).
  Use only as a calibrated micro-scale proxy, never as a headline.

## 4. The gate

- **Decision metric:** pass@1 (greedy or averaged over N samples with pinned decoding).
- **Published to match (confirm from SimpleRL's `eval_math_nodes.sh` + released checkpoints):**
  Qwen2.5-7B **GSM8K 88.2→91.7, MATH-500 64.6→78.2**; Mistral-7B GSM8K 21.2→75.0.
- **Guard metrics:** response length, KL to reference, entropy (the reward-hacking canaries).
- **Artifact gate:** eval their released checkpoint → match their post-RL number.
- **Pipeline gate:** run their GRPO script through our verl-on-Anyscale harness → reproduce
  the lift from the base model.

## 5. Pinned eval

The lighteval/vLLM harness with **fixed sample counts and decoding params** (temperature,
top-p, max-tokens, N) — sampled LLM eval is noisy, so **N eval seeds are part of the pin**
(`AUTORESEARCH.md` §2). Determinism receipt is within a pre-declared CI, not byte-exact.
Freeze the benchmark prompt sets (GSM8K test, MATH-500, AIME24) as versioned files.

## 6. Data

GSM8K / MATH / the 8k GSM8K+MATH prompt mix (parquet on HF) — all open. Small. **Input audit:**
prompt template + system prompt + reward spec (the RL analog of the field audit — a wrong
reward spec is this domain's "deleted geography").

## 7. Rayification

| Stage | What | Ray lib |
|---|---|---|
| rollout generation | vLLM sampling from the current policy | Ray (verl) + vLLM |
| reward scoring | verifier / rule-based reward | Ray tasks |
| policy update | GRPO/PPO, FSDP | Ray Train (verl PPO Ray Trainer) |
| eval | lighteval + vLLM, pinned decoding | Ray Data / batch |

Mostly a matter of running verl *on Anyscale* well (co-location, autoscaling rollout vs
learner) rather than re-plumbing — verl already speaks Ray.

## 8. Fidelity ladder

- **smoke:** TinyZero Countdown on a 0.5–1.5B model, 1 GPU, few steps — the code + reward loop
  runs.
- **proxy:** Qwen2.5-1.5B/3B zero-RL on the 8k mix, single 8-GPU node; rank recipe ideas.
- **full:** Qwen2.5-7B (SimpleRL's headline), 2 nodes ×8 GPU (~15h in their quote).
- **Proxy axis:** smaller base model + fewer RL steps + an **eval subset**. **Rare-signal
  trap:** rank transfer across model scale is the weakest link in this domain — **calibrate it
  explicitly** (replay a known 1.5B→7B ordering) before trusting any small-scale ranking.

## 9. "Beat it" hypotheses

1. **Cheaper recipe, same gain** — does GRPO-LoRA (verl reports 94.6–96.0 on GSM8K with LoRA)
   hold the full-FT lift at a fraction of the GPU-hours? Flag `lora`.
2. **Reward-shaping ablation** — the contested-claims literature says length gains can be
   reward artifacts; test a length-penalized reward and measure whether accuracy survives.
   Flag `reward_variant`.
3. **Rollout-count vs steps trade** — the throughput-is-a-model-change lesson: does more
   rollouts/step at fewer steps beat the reverse at equal GPU-hours? Flag `rollout_n`.

## 10. Budget

- **Wave 1 for the proxy/smoke (Qwen 1.5–3B), Wave 2 for the 7B full run.**
- **GPU-hours:** smoke ~2 · proxy (3B, several recipes) ~30–50 · full (7B, 2 nodes ×8, ~15h) —
  a single full run is ~120 GPU-hr → this is the budget-dominant campaign.
- **GPU tier:** H100/A100 (7B GRPO with vLLM rollouts wants the memory + throughput); spot ON
  with checkpointing (RL runs are long — checkpointing bounds preemption cost). **Full run
  ~$400–700 on spot; multi-node → PI approval required.**
- **Approval:** envelope + each full run + the multi-node request (Wave 2/3 gates).

## 11. Controls

- **Reward-hack canary (R3):** KL/entropy collapse aborts the run.
- **Base-model floor:** always report the pre-RL number on the same pinned eval (the honest
  baseline — several "RL gains" in the literature shrink against a properly-decoded base model).
- **Contested-claim guard:** don't headline "aha moment" / length-increase; anchor to pass@1
  deltas only.

## 12. Risks

vLLM + verl + Ray version coupling is brittle — pin the whole image (this domain's env-parity
gate). Multi-node interconnect (NCCL/InfiniBand verify) matters at 7B+. Sampled-eval variance
is large — never blog a single-seed pass@1. The 7B full run is the program's most expensive
single run; gate it hard behind a calibrated proxy.
