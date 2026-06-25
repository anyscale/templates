---
name: fintech-tfm-working-style
description: How Zach wants the fintech_transaction_fm template series built — working style and craft bar
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 27bfccd3-8165-4406-85f6-09e28afd43a0
---

Building the `fintech_transaction_fm` notebook series (templates/templates/fintech_transaction_fm), Zach's guidance:

- **Bias to action over asking.** He rejected a multiple-choice AskUserQuestion and said "letting you roll with it, work hard." When a call is mine to make and I can reason it out, decide and proceed (and flag it) rather than asking. Reserve questions for genuine forks I can't resolve.
- **Honesty over sales-pitch.** No faked/illustrative numbers — every number must come from a real run. Don't claim Ray "shines" on a stage where it's undifferentiated (he caught me calling embarrassingly-parallel work a Ray win, and caught skew/stragglers as not-Ray-specific). Show real bottlenecks and the real fix.
- **Show the lesson at real scale.** He explicitly did *not* want the scale-up story stuck at `mini`/CPU; he wanted the full dataset and a synthesized bigger one, with heavy runs referred out to the reader as Anyscale Jobs.
- **Prose bar is high.** Purpose before mechanism in every heading and lead; no AI-cringe (editorializing titles, `**Label**:` lists, raising a concept only to dismiss it, naming a term then waving at it). These are codified in the `zgarner-fieldeng-template` skill's `references/notebook-authoring.md`.

**Why:** he reviews top-to-bottom and is exacting; getting altitude/voice/honesty right up front saves heavy iteration.
**How to apply:** load the `zgarner-fieldeng-template` skill before authoring; ground every claim/number in a real run; decide-and-flag rather than over-ask. See [[fintech-tfm-series-state]].
