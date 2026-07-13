---
name: fintech-tfm-working-style
description: How Zach wants the fintech_transaction_fm template series built — working style and craft bar
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 27bfccd3-8165-4406-85f6-09e28afd43a0
---

Building the `fintech_transaction_fm` notebook series (templates/templates/fintech_transaction_fm), Zach's guidance:

- **Bias to action; ask in PROSE, never the multiple-choice widget.** He's fine with questions ("you can ask questions") but the abrupt AskUserQuestion (1)(2)(3) selection UI interrupts his flow — he rejected it twice. So: ask clarifying questions inline as normal prose, and default to decide-and-flag when a call is mine. When several threads are open, pick the highest-value one and go rather than presenting a menu.
- **Honesty over sales-pitch.** No faked/illustrative numbers — every number must come from a real run. Don't claim Ray "shines" on a stage where it's undifferentiated (he caught me calling embarrassingly-parallel work a Ray win, and caught skew/stragglers as not-Ray-specific). Show real bottlenecks and the real fix.
- **Show the lesson at real scale.** He explicitly did *not* want the scale-up story stuck at `mini`/CPU; he wanted the full dataset and a synthesized bigger one, with heavy runs referred out to the reader as Anyscale Jobs.
- **The node is EPHEMERAL — `~/.claude` (incl. this memory) is regularly lost.** The ONLY durable store is the git repo in the working dir (pushed to origin). So: write durable notes as repo files (`CHANGES.md`, `PERFORMANCE.md`), **commit AND push promptly** (unpushed work dies with the node too), and run `./setup_claude.sh backup` to snapshot `~/.claude` settings+memory into the tracked `claude-state/` and push. Don't rely on memory files surviving.
- **Prose bar is high.** Purpose before mechanism in every heading and lead; no AI-cringe (editorializing titles, `**Label**:` lists, raising a concept only to dismiss it, naming a term then waving at it). These are codified in the `zgarner-fieldeng-template` skill's `references/notebook-authoring.md`. **No cute idioms / colloquial sayings** — he reacted sharply to "without crying wolf". Say it literally ("without false positives"), not with a folksy metaphor. This applies to chat replies too, not just committed prose.

- **VOICE: write in ZACH'S voice, not Claude's — he called the existing template/presentation prose "real cringe" (2026-07-09).** His verbatim sample of a good opening: *"Transaction foundation models are the latest generation of transformer models - like LLM's, but instead of language, they are focused on financial transactions. This lets transaction foundation models recognize distinct patterns like fraud, that traditional ml techniques can't detect. Today I'm gonna show you how to build your own transaction foundation model and achieve performance and scalability that surpasses comparable approaches by Nvidia."* What it does: defines by analogy in one breath (no bolded-term formal definitions, no company name-drop lists); each sentence advances what-it-is → why-you-care → what-you-get-today; spoken register, first person, direct confident claim, zero throat-clearing. Measure all presentation AND notebook prose against this sample.

- **RESULT HIERARCHY (2026-07-09, Zach was FURIOUS about this):** the HEADLINE is always
  (1) our foundation model beats NVIDIA's foundation model (fm 0.04–0.06 vs 0.0123, 3–5×) and
  (2) our fusion beats their fusion (peak 0.284 vs 0.1755, clears it in ~1/6 draws). The raw
  0.1238 exact match is a TERTIARY result — it is the CONTROL, pinned to match by construction
  (their features, their recipe); leading with it "rewards the thing we don't care about."
  Mention raw-match only as the calibration that makes the two wins trustworthy. This ordering
  applies to every table, takeaway, summary, and deck slide.
- **THE AI-PROSE TELL IS STAGING, NOT LENGTH (2026-07-13, rejected twice same day):** first
  draft failed as em-dash essay sentences with cute idioms; the "fixed" version failed again
  because the short sentences were THEATRICAL — dramatic negation-hook openers ("The
  foundation model never sees a fraud label."), beat-drop rhythm sentences ("Similar
  transactions get similar embeddings."), staged reveal arcs. Zach: "this style of writing
  reeks of AI. You really cant tell?" His register is informational: subject first, facts in
  speaking order, plain connectives ("So its...", "Here, we..."), nothing composed for
  effect. Test every sentence: does it carry information or does it try to LAND? Codified in
  the skill's voice section.

- **No persona-metaphor definitions ("measures what a fraud team lives with") — Zach: "AI skeeze" (2026-07-13).** Define terms as literal fractions/quantities: "AP is the fraction of flagged transactions that are actually fraud, averaged across thresholds." Same family as the cute-idiom and staging bans; the tell is defining a metric via a vivid relatable persona instead of stating what it computes.

- **No 'fm' abbreviation in prose (2026-07-13):** "It slows me down constantly trying to
  remember what 'fm' is. Its a foundation model. Abbreviation here is harmful." Feature-set
  is called **embedding** in prose/tables (raw / embedding / fusion); spell out "foundation
  model" elsewhere. nb02–06 prose + nvscore's printed labels still say fm — sweep pending.
  Related nb01 lessons same session: walk the parts IN ORDER from where the reader sits
  (starting at "Parts 3&4" with no 1&2 is disorienting); introduce **raw** where it comes
  from (the data, Part 2), not first mentioned in a bullet list; section title was his
  parallel "What we're building" → "How we're building it".

- **Lingo rules (2026-07-09):** no "smoke run"/"smoke test" phrasing — say what it literally is ("stops after 30 steps as a demonstration", "proves the plumbing"). And any number offered as impressively small/large MUST ship with its comparator ("30 steps" means nothing until you say ours is ~16,000). "smoke" still lingers in nb02/03/04/06 + configs/mini.yaml — purge on the step-through.

**Why:** he reviews top-to-bottom and is exacting; getting altitude/voice/honesty right up front saves heavy iteration.
**How to apply:** load the `zgarner-fieldeng-template` skill before authoring; ground every claim/number in a real run; decide-and-flag rather than over-ask. See [[fintech-tfm-series-state]].
