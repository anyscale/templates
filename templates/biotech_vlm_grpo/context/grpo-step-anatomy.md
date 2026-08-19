# Anatomy of one GRPO step — dummy-but-real, every tensor shown

One prompt, G=4 completions, real numbers throughout. This is the whole loop for
the PathAI shape (single-turn, reward model, VLM).

---

## Stage 0 — The dataset row (what the parquet holds)

```
row = {
  "prompt": [
    {"role": "system", "content": "You are a pathology assistant. Examine the
                                   tissue patch and reason step by step."},
    {"role": "user",   "content": "<image> What tissue type is shown? Choose one:
                                   [adipose, background, debris, lymphocytes, mucus,
                                   smooth muscle, normal colon mucosa,
                                   cancer-associated stroma, adenocarcinoma epithelium].
                                   Think step by step, then answer as
                                   <answer>class_name</answer>."}
  ],
  "images":      [<224×224 RGB H&E patch — dense small dark-purple round cells>],
  "env_class":   "pathology_single_turn",
  "reward_spec": {"ground_truth": "lymphocytes"}
}
```

The dummy image (imagine it):

```
┌────────────────┐
│ ●●● ● ●● ●●● ● │   dense field of small, round,
│ ● ●●●● ● ●● ●● │   dark-purple nuclei, scant
│ ●● ● ●●●●● ● ● │   cytoplasm, evenly packed
│ ● ●●● ● ●● ●●● │   → textbook lymphocytes
└────────────────┘
```

## Stage 1 — Tokenization (what the generator actually sees)

The processor expands `<image>` into vision tokens. For Qwen2.5-VL at 224×224:
14px patches → 16×16 = 256 patches → 2×2 merge → **64 vision tokens**.

```
input_ids (prompt) = [sys tokens ×~30] +
                     [<|vision_start|>] + [<|image_pad|> ×64] + [<|vision_end|>] +
                     [user text tokens ×~55]
                   ≈ 152 prompt tokens          ← images ride along as pixel_values
pixel_values: float tensor [256, 1176]           (patches × patch_dim), fed to the
                                                  vision tower, fused at the pad slots
```

## Stage 2 — Rollout: vLLM samples G=4 completions (temp 0.8)

Same prompt, sampled 4 times. Abbreviated text, real structure:

```
c1 (58 tok): "The patch shows densely packed small round cells with dark nuclei
              and minimal cytoplasm, uniformly distributed. This morphology is
              classic for immune cells. <answer>lymphocytes</answer>"

c2 (41 tok): "Small dark round cells, high nuclear-to-cytoplasm ratio.
              lymphocytes. <answer>lymphocytes</answer>"

c3 (63 tok): "The fragmented dark material with irregular shapes and no clear
              cellular structure suggests necrotic material.
              <answer>debris</answer>"

c4 (37 tok): "These look like small immune cells, likely lymphocytes, though
              I am not fully certain."          ← no <answer> tags
```

**What vLLM returns per completion — this answers the logprob question:**

```
c1: token_ids  = [791, 11140, 5039, ...]                (58 ints)
    logprobs   = [-0.41, -1.87, -0.03, ...]             (58 floats)
```

**One scalar per SAMPLED token — not the logits.** At each decode step the model
produces a full logit vector over the vocab (~152k floats), softmaxes it, samples
ONE token, and keeps only `log p(that token)`. The 152k-float vector is discarded
immediately. So a 58-token completion contributes 58 floats, not 58×152k.
Full logits only ever exist transiently inside a forward pass; nothing in the
pipeline stores them.

## Stage 3 — Reward (env parse + reward model)

Rule reward: `1.0 · label_match + 0.2 · well_formed_tags`

```
           parsed answer      match   format   r_i
c1         "lymphocytes"      ✓       ✓        1.2
c2         "lymphocytes"      ✓       ✓        1.0   (weak CoT — say RM shaved it; or pure rule: 1.2)
c3         "debris"           ✗       ✓        0.2
c4         (no tags)          ✗       ✗        0.0
```

In Phase 3 the reward model replaces/blends the rule: same shape, the number just
comes from a forward pass of your scorer over (prompt, completion).

## Stage 4 — Group advantages (the "GR" in GRPO)

Normalize each reward against ITS OWN GROUP — no critic/value model anywhere:

```
r      = [1.2, 1.0, 0.2, 0.0]
mean   = 0.6
std    = 0.51
A_i    = (r_i − mean) / std  =  [+1.18, +0.78, −0.78, −1.18]
```

The advantage is a **per-completion scalar, broadcast to every token in that
completion**. All 58 tokens of c1 get +1.18; all 37 tokens of c4 get −1.18.
(This is why identical completions kill learning: r all equal → std 0 → A all 0.)

## Stage 5 — What ships to the trainer (the full batch dict)

Per sequence (padded to max length within the batch):

```
input_ids       [4, 152+64max]   prompt + completion token ids
attention_mask  [4, 216]
response_mask   [4, 216]         1 only on completion tokens (loss lands here)
pixel_values    [4, 256, 1176]   the SAME image, re-processed trainer-side
old_logprobs    [4, 64max]       Stage-2 floats from vLLM  (per sampled token)
advantages      [4, 64max]       Stage-4 scalars, broadcast (+1.18, +1.18, ...)
```

Note what is NOT in here: no logits, no reward model state, no env state.

## Stage 6 — Trainer forward: recompute, ratio, clip, KL

FSDP policy teacher-forces the same sequences and reads off ITS logprob for each
completion token → `new_logprobs [4, 64max]`. One real token from c1:

```
token = "lymphocytes"  (inside the answer tags)
old_logprob (vLLM, at sample time) = −0.90
new_logprob (policy, after k grad accum microbatches drifted it) = −0.70

ratio   = exp(new − old) = exp(0.20) = 1.22
clipped = min(ratio, 1+ε) = 1.20            (ε = 0.2)
loss_tok = −min(ratio·A, clipped·A) = −(1.20 × 1.18) = −1.42   → push it UP
```

Same math on a c4 token with A = −1.18 pushes its probability DOWN.

**KL penalty — the third model copy:** the frozen reference (the SFT checkpoint,
never updated) also teacher-forces the batch → `ref_logprobs [4, 64max]`.
Per-token `kl ≈ new_logprob − ref_logprob`, scaled by β (~1e-3), added to the
loss so the policy can't wander far from its SFT behavior.

⚠️ **Logprob-drift lives right here:** Stage 6 recomputes what Stage 2 measured.
If the trainer's tokenizer/chat-template/image-preprocessing disagrees with
vLLM's in ANY way, old vs new logprobs differ for the wrong reason and the ratio
is garbage. Same bug class as the SFT→RL stack handoff — this is why one stack
owning both sides of the recompute matters.

## Stage 7 — Update + weight sync (the arc in your sketch)

```
loss.backward() → optimizer.step() on FSDP shards
      → trainer NCCL-broadcasts fresh weights into the vLLM engines
      → engines now generate with the updated policy
      → next prompt batch (back to Stage 2)
```

Spike gate: generations for the same prompt at step N+1 differ from step N —
proof the broadcast actually landed. With LoRA the question sharpens: does the
sync merge adapters before pushing, or push adapters to vLLM's LoRA runtime?

---

## GPU tenants during all of this (one node, the colocation story)

```
vLLM engine(s)     — generation, Stage 2          ┐
FSDP policy        — recompute + update, Stage 6  │ four workloads,
frozen reference   — KL logprobs, Stage 6         │ one shared pool
reward model(s)    — scoring, Stage 3             ┘
```

That scheduling problem is the thing SkyRL owns and raw Ray Train doesn't touch.
```
