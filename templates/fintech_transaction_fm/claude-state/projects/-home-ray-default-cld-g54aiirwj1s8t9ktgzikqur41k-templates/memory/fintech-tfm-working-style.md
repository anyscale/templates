---
name: fintech-tfm-working-style
description: How Zach wants the fintech_transaction_fm template series built — working style and craft bar
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 27bfccd3-8165-4406-85f6-09e28afd43a0
---

Building the `fintech_transaction_fm` notebook series (templates/templates/fintech_transaction_fm), Zach's guidance:

**THE WRITING/REVIEW METHOD LIVES IN ZACH'S OWN REPO — github.com/zachgarner/zgarner-ai-skills,
checked out at `zgarner-ai-skills/` inside the template dir (embedded copies deleted
2026-07-28). Read it FRESH each session — he edits it himself (Lexicon, skim test,
earned punctuation: linter flags every semicolon and doubled mark) and his text is the
baseline.** Load BOTH skills: `zgarner-prose` (general method, voice, tells, linter) +
`zgarner-fieldeng-template` (notebook layer, collaboration protocol), and run
`zgarner-ai-skills/zgarner-prose/scripts/prose_lint.py` (both modes) before every
hand-back. That repo supersedes the voice notes that used to be duplicated here. What stays in THIS memory is fintech-project-specific:

- **RESULT HIERARCHY (NON-NEGOTIABLE, he was furious):** headline = (1) our foundation
  model beats NVIDIA's (embedding 0.04–0.06 vs 0.0123, 3–5×), (2) our fusion beats their
  fusion (peak 0.284 vs 0.1755); fine-tuned ALONE is the pipeline-replacement story
  (parity with raw, bootstrap 52% of draws — never headline fine-tuned+raw over it);
  raw 0.1238 exact match is the CONTROL, never the headline. Applies to every table,
  takeaway, and deck slide.
- **NVIDIA framing (his words, 2026-07-28):** "NVIDIA is part of the story here, but its
  really how Anyscale Ray lets you build Financial Transaction models. We are talking
  about NVIDIA only because they have the first public blueprint and we build on top of
  it because people are familiar with it." Lead every notebook with what we build;
  NVIDIA-comparison is the measurement protocol, never the lead framing ("What we're
  comparing" read as a grudge → became "Three fraud detectors"). Results sections keep
  the beat-NVIDIA headline per the result hierarchy above.
- **Real scale:** show lessons at full scale, refer heavy runs out as Anyscale Jobs;
  mini exists to prove plumbing, and its numbers mean nothing (say so).
- **Every number from a real run** — his hardest rule; ranges where reruns move a value
  (embedding AP is quoted as the range 0.04–0.06 by his call).
- **Bias to action; questions in prose, never the AskUserQuestion widget** (rejected twice).
- **The node is ephemeral:** repo files + prompt push are the only durable store;
  `./setup_claude.sh backup` snapshots memory/settings into tracked claude-state/.

**Why:** he reviews top-to-bottom and is exacting; the codified method is what keeps
his review burden down. **How to apply:** load the skill, run its protocol, ship the
written audit with every hand-back. See [[fintech-tfm-series-state]],
[[fintech-tfm-fidelity-principle]], [[fintech-tfm-no-concurrent-edits]].
