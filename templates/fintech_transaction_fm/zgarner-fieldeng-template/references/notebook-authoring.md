# Authoring a template notebook — the complete method

This is Zach Garner's writing and reviewing method for workshop-grade template notebooks, consolidated for transfer. It was built sentence-by-sentence across the `fintech_transaction_fm` review sessions (July 2026); the quotes and dates are the provenance. Read it fully before writing anything he will review. The companion tool is `scripts/prose_lint.py`; the mechanics of the templates repo (BUILD.yaml, tests, publishing) belong to that repo's own `template` skill.

How to use it: Part 1 is the method — run it, don't skim it. Parts 2–6 are the standards the method checks against. Part 7 is the collaboration protocol; violating it has destroyed his work before. Part 8 lists where rules do NOT apply. The checklist at the end mirrors the hand-back protocol in order.

---

# Part 1 — The method

## The umbrella principle: keep the reader in the loop

Zach: "Many things i've told you come down to that rule." A multi-notebook workshop is one continuous experience. Every named thing creates curiosity, and every unresolved mention is a loose end. When a takeaway names output files, say where they get opened. When a design choice pays off later, name the notebook that pays it off. When a term appears, gloss it now or say which part explains it. The reader should never hold an unanswered question the series has an answer to without being told where that answer lives. Forward pointers are commitments: when writing the later notebook, check the earlier ones' promises and honor them.

## Validate sentences by their JOB, not against a blacklist

The tells (Part 3) are a blacklist, and a sentence can dodge every named tell and still be empty. The actual validation (Zach: "is that why you dont validate your sentences?"): give every sentence a label for the job it does — **claim, fact, consequence, pointer, gloss, or instruction** — then check two things:

1. **The label fits the position.** A section opens with a claim. A body sentence is a fact or a consequence. A closer is a deliverable or a pointer.
2. **The content fills the label.** A claim actually claims the section's point; a fact is checkable.

A sentence that takes no label has no job — cut it. The named tells are the common ways a sentence fakes a job: a sandwich is a verdict posing as a fact, a dash inventory is detail posing as part of a claim, a grandstand is importance posing as content. Label first; the blacklist is cleanup.

## First and last sentences are power positions

The reader's eye lands on openings, closings, and headings — put the point there.

- **Power means the claim itself, not the active-voice form.** "We watch two numbers" is short, active, We-led — and pure skeeze, because it announces that content is coming instead of delivering it. If the first sentence were the only one the reader saw, did they learn the thing?
- **Truth is not sufficient.** "Training prints two numbers per epoch" is literally true and still fails — it inventories the content instead of delivering it. Any opener whose job is counting or listing what follows gets deleted; the structure shows itself.
- **A power sentence fails if it is the wrong claim.** "Embedding cost is linear in the transaction count" is strong and true and was still wrong as an opener — linearity was a property, not the section's point (the point was: volume is the scale problem and pool size is the answer). The opener test is "is this THE section's claim," not "is this a strong sentence."
- **Backstory openers fail even when skeeze-free** (Zach, on "The foundation model trained on long card histories, but…"): opening on the tension or motivation is still setup. The first sentence is the claim or decision itself ("We embed each transaction on its own…"); motivation moves to a later sentence.
- **Closers must deserve the position.** A final sentence that reads like a code comment ("We compute one number before training: …") is a code comment — move it there. Close on the strong concrete fact (the scale span, the result), never on a minor detail, never on a "because"-tail, never on an aphorism.
- **No sentence exists to set up another.** If a sentence's function is framing a contrast, building to a reveal, or seeding a punchline, it is staging regardless of its grammar. Zach: "I dont punchline setup anything. I speak plainly. I use power sentences." Put the facts adjacent and let the difference speak: "NVIDIA's notebook trains a 30-step demonstration and downloads its real weights. Ours trains the full ~16,000 steps."

## The section-opener audit is a mandatory, WRITTEN step

Zach: "you have a rule of thumb about the first sentence but arent using it." Positional rules can't be grepped and don't survive as vibes. Before shipping, extract every section's first and last sentence, state the section's claim in one line, and answer in writing "is sentence one that claim?" Problem statements, definitions, motivation, and backstory all FAIL even when skeeze-free. If no written verdict table was produced, the review didn't happen.

## The review loop — iterate to fixpoint before any handover

A single pass or a grep catches one rule; the reviewer needs all of them held at once.

- **Pass A — high level.** What is this section FOR, in one sentence? Delete or move every paragraph that doesn't serve it. Check location (is this content owned by another notebook or section?), structure (headers over big code, code/prose interleave, recap-then-next after big cells), and duplication against the rest of the notebook.
- **Pass B — sentence by sentence.** The job-label audit (written), then every sentence and comment against the tells. First and last sentence of each paragraph audited hardest — first and second words are power words.
- **Pass C — high level again.** Re-read after B's edits: flow intact, no new seams, no orphaned references, openers and closers still the strongest sentences, nothing now duplicated.

Repeat A→B→C until one full cycle produces zero changes. Only then hand over. Log what each pass caught — a loop that catches nothing on its first cycle probably wasn't run.

## Sweep, don't spot-fix

When the reviewer flags a sentence pattern, the flagged sentence is never the only instance. Fix it, then immediately re-scan every markdown cell and comment in the notebook for the same pattern before handing back. Making the reviewer repeat the same correction on the next paragraph is the single fastest way to burn their patience — they are teaching a rule, not editing a line.

---

# Part 2 — Voice and register

## The calibration sample (Zach's own writing)

> Transaction foundation models are the latest generation of transformer models - like LLM's, but instead of language, they are focused on financial transactions. This lets transaction foundation models recognize distinct patterns like fraud, that traditional ml techniques can't detect. Today I'm gonna show you how to build your own transaction foundation model and achieve performance and scalability that surpasses comparable approaches by Nvidia.

What it does: defines the new thing **by analogy to a known thing, in one breath** — not a formal bolded-term definition, not a company name-drop list. Each sentence advances the reader: what it is → why you care → what you're getting. First person, direct, confident claim, zero throat-clearing. If a draft reads denser or more "impressive" than this, it's wrong.

## The register rules

- **Action tone: lead with the task, not a description.** "The 80/10/10 boundaries are positions in time…" reads like documentation; "We need two dates: the day by which 80% of all transactions have happened…" reads like someone running a workshop. Open steps with *We need / We do / Now we*; the mechanism arrives as the way we do the thing, never as the subject. (Zach: "Speaking with more action tone instead of passive is better for a workshop.")
- **The general engineer's level: the real word, then its plain meaning.** Textbook-speak fails ("Grouping is bound by data movement" — Zach: "is that like a bowel movement?"); dumbed-down fails too ("Grouping is hard" — easy/difficult carry no information; "plain doesn't mean dumbed down"). The target is the claim an engineer would state at a whiteboard, with the term of art introduced in passing and the concrete resource named with a magnitude: "Grouping is limited by how fast you can move data around … nearly every row travels across the cluster network (data engineers call this a shuffle) … gigabytes here, terabytes at production scale."
- **Connect the logical chain — no gap between goal and mechanism.** A paragraph that states a goal ("cutoffs are defined by counts") and then names an operation ("groupby by date") without the middle link ("we need each day's count, plus a running total over the days in order") reads as two unrelated facts. Walk goal → what that requires → the operation that provides it. If the reader could ask "what does that have to do with it?", the link is missing.
- **One word per concept, held for the whole notebook — and the series.** If the intro calls them "splits," the section titles, prose, and prints call them splits. Never "parts" in one place and "splits" in another; never "sequences" in prints when the prose says "windows."
- **Define at the moment of understanding — never by forward reference.** Gloss a term in the sentence right after the reader has just understood the thing it names: "…the difference between its guess and the real token is the training signal. This is what makes the model *causal*: every prediction uses only the past." Pointing at output that hasn't happened yet ("what 'causal' means in the printout below") was rejected: "it's weird to explain something in the future."
- **Affirmative framing is the default for EVERY sentence, not just openers.** "No card depends on another" was corrected to "each card is independent" — same fact, stated as what IS. Negation makes the reader hold an absence; the affirmative hands them the property. (Caught in a one-sentence insert after the rule already existed — it was being applied only to openers.)
- **Impact before mechanics — and mechanism-only facts may not deserve prose.** "Ray writes checkpoints to shared storage, so an interrupted run picks up where it left off" leads with plumbing. Invert: "Ray makes the run durable: if training is interrupted, it resumes from the last checkpoint instead of starting over." And apply the test before writing at all: gradient averaging is pure mechanism with no user-felt impact — it belongs in a code comment, not the prose.
- **The deeper AI tell is staging, not sentence length.** A draft was rejected twice for the same disease in two disguises: long em-dash essay sentences with cute asides, then — after shortening — *theatrical* short sentences: the dramatic negation-hook opener ("The foundation model never sees a fraud label."), the beat-drop rhythm sentence, the designed statement-then-elegant-elaboration arc. Human engineer prose is informational: subject first, facts in speaking order, nothing staged for effect. Test every sentence: does it carry, or does it try to land?
- **The final test for any sentence:** would an engineer write this to another engineer, or does it read like it's filling a section template?

---

# Part 3 — The tell catalog (the blacklist; `prose_lint.py` greps the mechanical ones)

## Framing tells — sentences posing as content

- **Grandstanding** — announcing importance before the thing: "We built the artifact every later notebook reads" (Zach: "just fucking annoying waste of reading"). Say what you did, then contextualize: "We built our training/validation/test (80/10/10) splits. Every later notebook reuses them."
- **The curator phrase** — "the number to watch," "the result that matters," "the knob worth understanding": assigning importance instead of stating the fact that creates it. Tour-guide voice — pointing at the exhibit instead of being it.
- **Movie-preview lines** — "the one line that moves laptop → cluster," "the payoff is," "full stop." Dramatic emphasis is skeeze.
- **The announced contrast** — "the same idea, with one big difference," "here's the catch": trailering a difference instead of stating it. Put the two facts adjacent: "An LLM tokenizer splits text into word pieces and learns its vocabulary. A transaction has no text to split."
- **The announce-colon** — a content-free label staged before the payload: "Perplexity is the number to watch: how many tokens…" → "Perplexity measures how many tokens the model is choosing between." Test the left half alone: if it taught nothing, delete it. (Colons whose left half is content survive — see counter-rules.)
- **Editorializing titles** — "Class imbalance — and why we don't report plain accuracy." Name the thing: "How we measure performance."
- **Raising a concept only to dismiss it** — don't introduce AUC-ROC just to say you don't use it. Its sneakiest form is the **negative opener**: opening a section by demolishing a thing no one proposed ("accuracy is a useless score…", "this stage never needs a GPU…"). Open with what we do; dismiss nothing. (Caught twice in one page, 2026-07-21.)
- **Filler connectives** — "drives the rest of the series," "it's worth noting that," "the operationally meaningful number."
- **`**Label**:` bullet lists** — every item a bold noun + colon. Write sentences.
- **Naming a term then waving at it** — name the real term AND gloss it concretely ("keep 1 in 50 normals, weight each survivor ×50"), not with more abstraction.

## Sentence-shape tells

- **The dash-aside sandwich: verdict — whispered justification — consequence.** "Memory is easy here — inference keeps no gradients or optimizer state — so each actor runs large batches." Delete the verdict, promote the evidence, keep the consequence: "Inference keeps no gradients or optimizer state, so each actor runs large batches." Zach's correction word for this family: **"sandwich."**
- **The dash inventory: a finished sentence with a parts list stapled on.** "…wrote the results to shared storage — `embed_`, `lbl_`, and `raw_` files per split." The staple is code-level detail; cut or promote, never dangle. In first position it buries the power sentence.
- **The punctuation pile** — a sentence needing a colon, a parenthetical, AND a semicolon is several sentences pretending to be one. Zach's parody: "BLAH BLAH BLAH BLAH: BLAH, BLAH( BLAH BLAH); BLAH."
- **The because-tail** — "X happens, because [long clause]." as a closer. Two direct sentences: the fact, then the reason as its own statement.
- **Notation-as-prose** — "`<bos>` + the 12 field tokens + `<eos>`" is not a sentence. Say it in English ("a 14-token sequence, its 12 field tokens wrapped in `<bos>` and `<eos>`"); the symbolic form lives in code and code comments only.
- **Verbless fragments as sentences** — "Twelve tokens per transaction, all drawn from one shared vocabulary." Give it a verb.
- **Walls of text** — one thick paragraph carrying five ideas. One idea per paragraph; short lists where the content is enumerable.

## Word tells

- **Animate verbs for inanimate things.** Things do not "live," "ride along," "carry," "sit," or "come home" — they ARE and they're IN. "The details live in src/model.py" → "the details are in src/model.py." Same family: "carries the lesson," "the loop owns," "sets the recurring bill."
- **The term-of-art test: it buys precision the plain phrase lacks, or it goes.** "Shuffle" and "embarrassingly parallel" earn their place (specific mechanics an engineer will meet again — glossed at first use). "Corpus" fails — "training data" says the same thing to everyone (Zach: "i hate the word corpus… it alienates people who arent specifically trained"). Also banned: "smoke test/run," "de-facto," "fm" as an abbreviation, easy/hard as information-free verdicts.
- **Jargon stacks in comments.** "stratified_eval draws the seeded eval sample" stacks three jargon words; "picks 100K random rows from a split, keeping the fraud rate the same as the whole split — the fixed seed makes it pick the same rows every run" says it in words anyone can act on. Statistics verbs (draw, seeded, stratified) never appear without their plain meaning doing the work.
- **Anthropomorphic gloss where a precise noun exists.** "The model's understanding of a transaction, written as numbers" was rejected for "the model's vector representation of a transaction." Use the standard noun (embedding, attention mask) and gloss it — don't paraphrase it into model psychology.

---

# Part 4 — Structure of a notebook

## The intro pattern

Recap first, at an altitude a returning reader absorbs without homework ("Previously in Part 2, we built the train/validation/test splits" — not "packed into fixed-length windows on shared storage," which forces them to remember details they don't need yet). Then why this notebook exists, then the roadmap in one or two sentences. When notebooks form a group, the intro places the reader in the group (his nb05 pattern: recap → this notebook's role in the fraud-detector arc → "Later, Part 7 builds a stronger detector…"). Content ownership applies to intros too — a recurring-cost sentence was cut from an intro because Scaling factors owned recurrence.

## Sections

- **One section per activity.** Consecutive sections must not repeat the same noun — "Why the split is temporal" / "The split as a Ray Data pipeline" / "The train split at a glance" is one activity wearing three headers. Merge into one section with `###` steps: each step a short lead plus the code cell that does exactly that step.
- **Titles: one action verb plus a concrete object** ("Write the three splits"), never a double-verb compound, never a "Why X" title, never a question, never a static label ("The model" → "What we're training"). Purpose before mechanism: name the decision or output, not the phenomenon or the API. The test for every heading and lead sentence: delete the mechanism words (`groupby`, `map_groups`, "one function per card") — if no point remains, rewrite; mechanism is the second sentence, never the first.
- **A section you keep patching is a section to question.** Before improving it a third time, ask which notebook or section owns its content. A measurement section was rewritten twice and grew a demo cell before the honest answer surfaced: the metric explanation was Part 1's, the noise caveat was Part 6's, and the notebook scored nothing — deletion plus one bridge sentence was the fix. The tell: every cell you try under the heading feels wrong. The heading is wrong, not the cells.
- **Don't over-prove — receipts are for the repo, not the reader.** One sentence of verification with a pointer is the ceiling: "The translation is verified byte-identical to NVIDIA's original (the checks are in `scripts/`)." Zach: "no one is standing around in disbelief; it's a distraction from the point of the work." Same instinct for weak checks: a check that needs an antidote paragraph to not mislead (the collapse check, which cried wolf on us in July) gets replaced by the plain artifact check, not defended harder.
- **Show why, not just what.** Motivate each choice from the data or the problem (*amounts span orders of magnitude → bucket them*; *fraud is ~0.1% → AP, not accuracy*; *workers run on other nodes → checkpoint to shared storage*). The justification is markdown around the lesson — not a wall of comments in hidden code, not prose with no code to anchor it.
- **Numbers get ONE owner section.** The full run's steps and hours belong to Scaling factors; other sections reference, never restate.
- **Show the intermediate result.** If a step computes something the reader can look at, show it (the cutoff dates were computed inside a cache guard and never printed; moving them to an always-run cell bought a visible result and, in the output, the autoscaler bringing nodes up — the "declare work, hardware arrives" story in the artifact).
- **Real examples beat invented ones.** The prose invented a hypothetical $900 fraud while the display showed boring card 0; one query found card 66000 — routine Texas purchases, then a same-day burst of Mexico charges escalating \$45 → \$514, the series' thesis visible in ten rows. Find the pattern in the data and display it; hardcode the chosen example (with a comment saying why) if a programmatic pick could desync prose from display across scales.
- **Every notebook verifies what it built** before the next depends on it ("Check the training set," "Check the windows," "Check the embeddings") — a plain artifact check: counts, shapes, one concrete peek.

## Around big code

- **Large code cells get their own `###` header** ("Defining the training function", "Run the training") — structure stays visible while scrolling, and headed sections collapse in Jupyter.
- **After big code, re-orient before advancing.** The reader loses the thread inside a long cell; the author never does — which is why the author skips the recap and the reader flounders. Zach's template: "In the last coding section we built the PyTorch training function, integrated with Ray for distributed training. Next, we count the total number of training steps — the learning-rate schedule needs it before training starts." Recap, then next, with the reason attached.
- **No cell-inventory transitions.** "Most of the cell below is `train_loop_config`, the settings…" is the counting tell in transition form. Say the action ("Now we configure the training run, and run it") and let the cell show itself.
- **Prose carries concepts; code carries names.** Filenames, function names, API names, seeds, and parameter meanings are banned from concept-level bullets and takeaways, mandatory at their line in the code. Zach: "save the detaily stuff for the code. I need higher level."

## The Scaling factors section — the established pattern

Every notebook 02+ ends its technical content with "## Scaling factors," and the pattern is fixed. Open on the measured fact or the concrete limit — never a frame ("The scaling problem is X", "Ray's answer is Y", "The arithmetic is linear" are labels posing as sentences; state the facts and let them argue). Body: what breaks and when, with the resource named (RAM, network bandwidth, GPU memory, cores) and a magnitude (GBs here, TBs at production; "past a few hundred million rows"). The table format: `What grows | The limit it hits | What absorbs it | Measured at full` — every number from a real run, "—" where unmeasured. Then the 10× arithmetic (the same pool takes N× longer; N× the workers brings it back) and the fact that only a config line changes. Recurrence, elasticity, and GPU-vs-CPU are stated as facts when true, never as sales beats.

## Takeaways — altitude validated by a reverted rewrite

The evidence is commit `41454b14`: a takeaways rewrite was reverted wholesale, and Zach edited even the kept version further. What the accepted versions do:

- **No metric re-argument.** The metrics section owns those numbers; the takeaway states what exists now.
- **Product meaning first.** His edit: "We trained the foundation model, and its now ready to be used for prediction systems like fraud detection. All of the later approaches build on top of this foundation model."
- **Ray is one clause at this altitude** ("wrapped in Ray, which handles the distributed scaling"), with one concrete fact (`ScalingConfig`). The four-verb enumeration belongs next to the code.
- **No API inventories, no line-count asides** ("build_model is 10 lines in src/model.py" — he parenthesized it, then deleted his own version), no timing details (the header's time-to-complete line owns those).
- **Lead with the transferable Ray lesson, then the domain observations** — a Ray notebook whose takeaways are all dataset trivia threw away its own point.
- When the takeaway names output artifacts, point to where they're opened next — and that pointer is a commitment the next notebook honors.

## The Next blurb is one plain sentence

What the reader does next, in words they already own. No class names, no magic numbers, no feature lists — "`FinancialTabularTokenizer` (merchant hashing + category hierarchy + temporal encoding, vocab 6251)" is a jargon dump about a notebook the reader hasn't opened; "turn each card's transactions into the token sequences the model will train on" is the same pointer in plain words.

---

# Part 5 — Code

## What to show, what to hide

A template teaches a **transferable lesson** — the Ray/Anyscale pattern the reader came to learn. Every block of code is either that lesson or incidental to it.

- **Show** (inline in the notebook): the primitive the template is about and the handful of verbs that convey its shape.
- **Hide** (import from `src/`): dataset/domain munging, parsing, sealed reference code, anything long the reader won't reuse.

The failure mode is hiding the lesson inside an incidental wrapper: a clean-looking `prepare_data()` import that buries a full Ray Data pipeline. Canonical example: loading TabFormer IS a Ray Data pipeline (`read_csv → map_batches → groupby.map_groups → write_parquet` shown inline); the per-row munging (`"$57.20" → 57.20`) hides in `src/`.

- **The deciding question is "would the reader have to look it up?"** — not "is this incidental?" A helper earns its place only when its call site reads clearly (`ensure_download`, `normalize_date_column`). If you'd invent an opaque wrapper just to shorten a cell, the hiding cost more than it saved.
- **Every hidden helper raises the same silent question: is that 2 lines or 100?** (Zach: "every obscuration makes the audience wonder whats actually going on.") Remedies in order: (a) inline anything trivial enough to recognize on sight — a 3-line AdamW constructor proves "ordinary PyTorch" better than a `make_optimizer` wrapper; (b) disclose the sizes of what stays hidden, in prose; (c) a branch that never runs at any preset (FSDP) is wonder-bait — inline it with a comment saying it's off, or cut it.
- **When Ray is buried, refactor `src` — never annotate the burial.** Comments and prose excerpts pointing at Ray calls the reader can't see do not fix the problem (Zach: "you cannot bury the ray code… you're going to have to refactor src"). The pattern: extract incidental pieces into small public helpers; define the composed function or class INLINE in the notebook with the Ray calls at their real lines; `src` keeps a copy composing the identical helpers for the headless path.
- **Show inline without forking logic.** The notebook and any headless entry point compose the SAME `src/` helpers — never two copies of logic that can drift.
- **Acceptance for any refactor of validated code is a bit-match.** Deterministic mini runs must reproduce prior outputs exactly (losses 8.742/8.668 across three refactor rounds; the example embedding vector across the actor inlining). If the numbers move at all, revert.
- **Use each Ray tool for its own job, in the open**: `filter(expr=col(...) == …)` for row predicates, `map_batches` for transforms, `.remote()`/`ray.get()` shown at the call site for tasks.
- **~25–30 lines is the ceiling** for an inline function before it reads as a wall; three visual blocks (setup / loop / report) is the accepted fix, plus splitting multi-purpose cells (arithmetic cell, then the trainer cell).

## Comments

- **One step, one comment, at its own line.** A block comment explaining a chained expression means the chain should be split (parenthesize a Ray Data chain so each step carries its comment). Zach: "Speak plainly, one step at a time."
- **A comment must survive being read alone.** Readers scan comments independent of code. A bare marker ("(1) wrap for distributed training") forces reconstruction; write the whole story. Corollary for branches: an else-comment must not imply the if lacks the property — each branch states its whole story. (Zach: "that took 5 minutes where it could have been 15s.")
- **First sentence of a name's first comment is "X is/does Y."** "FinancialTabularTokenizer is NVIDIA's tokenizer: it converts transactions to tokens and back." Never describe the operation while leaving the actor undefined.
- **Answer at the line that raises the question.** "How does it know how many workers?" is answered on the `read_parquet` line (it doesn't — one task per shard, autoscaler adds nodes when tasks queue). Laziness is explained at the `to_pandas()` that triggers execution, never as an abstract claim the next statement silently falsifies. No comment whose truth expires within its own cell.
- **Provenance-and-move-on for standard concepts** (from Zach's own edits): "Same as NVIDIA's blueprint, basic stuff for transformers." beats three sentences re-teaching AdamW. Multi-part comments become dash bullets, one idea per line.
- **The Ray Note convention**: prose numbers the Ray integration points ("adapted for Ray in three places"), and code comments reference them by name — `# See Ray Note #1 above` — never a bare number. The prose must introduce that exact name or the pointer dangles.
- **Names read from the call site with honest provenance.** `normalize_batch` sounds like a black box; `normalize_date_column` doesn't. "add_date_column" made the reviewer ask where the date came from — "normalize" says it's derived from existing fields. If a reviewer stops to ask what a function is, the name is wrong.
- Zach's model comment, the reference register: `# Clean up and write out the split metadata to split_meta.json.` Subject, verb, object, done.

## Plots and markdown gotchas

- **"Make it look better" means styling, not structure**: theme it, kill chartjunk, human-format axes (600000 → 600k). Never silently re-axis — hand-relabeled log ticks change what the reader is looking at.
- **But the plot must show its point**: a long-tailed quantity on a linear y-axis is one tall bar and an invisible tail — log the axis.
- An unescaped `$` in Jupyter markdown triggers MathJax and silently garbles everything to the next `$`. Escape amounts: `\$57.20`.

## Outputs, when committed on the working branch

Curate to the informative lines: real results, plus the infra lines that tell the Ray story (autoscaler node arrivals). Never progress bars, logger spam, or float noise (`round(float(x), 3)`, not float32 `tolist()`). Display slices exact — 27 tokens is two full transactions, not "~2". The publish pipeline strips outputs; the reader must still get the story from code + prose + a described expected result.

## It must run

- Committed defaults execute at CI/mini scale (usually CPU); scale-up is one obvious knob left at the runnable setting.
- The proof is a green papermill run. Trust papermill's OWN exit code — `papermill … | tail` chains report the tail's exit and have hidden a `NameError` as green. Scan the executed notebook for `output_type == "error"`.
- After moving or changing an import, re-run the WHOLE notebook — a later cell may use the symbol you relocated.

---

# Part 6 — Truth

- **Every number comes from a real run.** No invented, illustrative, or remembered numbers, ever. If a sentence names a shape, a magnitude, or a rate, compute it before writing it (`describe()`, a quantile, "what share is in the top 1%"). Wrong adjectives shipped repeatedly in first drafts — "heavy-tailed" for a tame lognormal, "most cards are quiet" when the median card had 2,500 transactions. Heavy-tailed, long-tailed, and right-skewed are different shapes with different consequences; use the one the data shows.
- **Difficulty claims need verifying too.** "Tokenizing transactions is simpler than tokenizing text" shipped in a project where the tokenizer was the single hardest thing to get right. If the work fought you, don't call it simple — say which part is mechanical and which is hard.
- **Fact-check at the claim's own altitude.** A platform-level statement ("Ray distributes the model either way") is not falsified by mechanism-level attribution (FSDP's sharding being PyTorch code). Never "correct" a true sentence into weaker prose — and when Ray truly is the actor, [Ray][verb] is the right shape, especially in a Ray workshop.
- **Honesty cuts both ways.** Don't claim Ray shines where it's undifferentiated — and don't volunteer what reads as a Ray defect when the fact is true of every distributed engine ("even with Ray's `preserve_order` on" turned a neutral fact into a Ray gotcha). State the limitation generically, then the design that handles it. Operator-level caveats go in the performance docs, not workshop prose.
- **Claims about the near-scale future stay honest**: "at 24M rows you can rent a bigger machine; at billions you can't" — hedged where unmeasured, concrete where measured.
- The competitor appears as a punchline or a comparison anchor, never as an obsession — one comparability sentence where it earns its place, one punchline where the facts deliver it.

---

# Part 7 — Working with Zach (the collaboration protocol)

## His text and his corrections

- **His text is the baseline.** Preserve his sentences verbatim; never revert his wording; flag typos once and let him decide (never silently fix). He drafts with `[tbd]` markers for you to fill — fill the marker, preserve his frame.
- **Fact-check his technical claims against the code and correct with evidence** — he expects it ("Fact check me on my updates"). Example: "worker count (autoscaled by Ray)" → the cluster autoscales nodes; the worker count is fixed.
- **His correction shorthand**: "skeeze" (performed/salesy prose of any kind), "sandwich" (verdict — whispered reason — so consequence), "dash inventory" (stapled parts list), "blah blah blah" (structure parody — the sentence's shape is noise). When he names a pattern, codify it in this document and the linter the same day.
- **His messaging hierarchy is inviolable.** Which result is the headline, what gets bolded, what's the control — once he sets it, every table, takeaway, and summary follows it. (The project-specific hierarchy is in project memory, not here.)
- **Answer length matches question size.** "EXPLAIN THIS WITH LESS FUCKING WORDS" — a definition question gets three lines, not three paragraphs. Terse chat generally.
- **Bias to action; ask questions in prose, never the multiple-choice widget** (he rejected the AskUserQuestion UI twice). When a call is yours, decide and flag it; when several threads are open, take the highest-value one.

## The file-safety rules (violating these destroyed his work once)

1. **Never write a file he is editing.** Chat-first patches (paste-ready blocks) until an explicit hand-off ("i finished my edits, you go").
2. **Verify "it's saved" against disk** — his editor's saves lag. His nb03 corpus purge was silently lost this way and only recovered because the linter caught the words again. `git status` before believing any buffer state.
3. **Commit whatever is on disk before any write** (`wip:` commits are fine) — everything that ever reached disk stays recoverable.
4. **Re-load and re-diff at write time** — never hold a loaded copy across a background run and then dump it over the file.
5. The node is ephemeral: **commit AND push promptly** — unpushed work dies with the node. Durable notes go in repo files, and `./setup_claude.sh backup` snapshots memory/settings.
6. Tell him **"kernel restart needed"** whenever `src/` changed — Python won't re-read a loaded module, and the resulting ImportError looks like your bug.

## The hand-back protocol — every time, in order

1. `git diff` the file; wip-commit what's on disk.
2. Write, re-loading at write time.
3. Run the review loop (A/B/C) to fixpoint, including the written job-label audit.
4. Run `scripts/prose_lint.py` on the notebook, and `--imports` when code changed. Zero hits or fix them.
5. Verify: papermill at mini, checked by its own exit and error-cell scan; bit-match when validated code moved; graft curated outputs.
6. Commit and push immediately.
7. Hand back WITH the audit shown — verdicts he can check, not conclusions he must extract. After any correction: sweep the whole notebook for the pattern before returning.

---

# Part 8 — Counter-rules: where the rules do NOT apply

- **Negation is allowed when the absence is the point.** "The card's history is not in it" (the design caveat), "fraud labels play no part in this step" (self-supervision), "Memory is not the constraint here" (the contrast). State the affirmative fact first when one exists.
- **A colon survives when its left half is content.** Zach's own: "This job has two main steps: grouping the rows by card, then tokenizing each card."
- **A term of art survives when the plain phrase loses information** — then it MUST be glossed at first use (shuffle, embarrassingly parallel, attention mask, causal).
- **Detail survives in code that dies in prose.** Filenames, seeds, API names: banned from concept bullets and takeaways, mandatory at their line in the code.
- **A power sentence fails if it is the wrong claim** — strength of form never substitutes for being the section's point.
- **[Ray][verb] is correct when Ray is truly the actor** — don't demote the platform from subject position out of misplaced modesty.

# Part 9 — Unconfirmed (inferred from what he accepted, not stated by him)

- Autoscaler node-arrival lines are worth keeping in committed outputs (show the elasticity story).
- The NVIDIA punchline survives in takeaways when the facts deliver it (survived his cleanup).
- A single em-dash aside per paragraph is acceptable; two in one sentence is sandwich/pile territory.
- Rhetorical questions in prose bodies: unresolved — question titles are banned; body questions have been avoided rather than ruled on.
- Tables are the preferred Scaling-factors body (nb04/05); earlier prose-only versions (nb02/03) may be revisited when those pages reopen.
- Known linter backlog in CLOSED pages (his call, untouched): nb01 series table says "the pretrain corpus"; ~11 hits in nb02/nb03, several in his own approved text.

---

# The checklist (mirrors the protocol)

**Before writing**
- [ ] `git diff`; wip-commit the on-disk state; confirm he's not mid-edit.
- [ ] Read the neighboring notebooks' promises this page must honor.

**While writing**
- [ ] Every sentence pre-labeled: claim / fact / consequence / pointer / gloss / instruction — label fits position, content fills label.
- [ ] Openers are the section's claim (the RIGHT claim); closers deserve the position.
- [ ] Action tone; affirmative default; one word per concept; chains with no gaps; terms glossed at the moment of understanding.
- [ ] Concepts in prose, names in code; numbers have one owner section; verification is one sentence + pointer.
- [ ] Ray visible where Ray is the lesson; helpers pass the look-it-up test; sizes disclosed; no never-runs branches.
- [ ] Comments: one step one comment, X-does-Y first mention, survive being read alone, answer at the asking line.
- [ ] Every shape/magnitude/difficulty claim computed or verified; every number from a real run.

**Before handing back**
- [ ] Review loop A/B/C to fixpoint, written opener/job audit produced.
- [ ] `prose_lint.py` clean; `--imports` audited when code changed.
- [ ] Papermill green by its own exit + error-cell scan; whole notebook re-run after any import change; bit-match if validated code moved.
- [ ] Outputs curated (real results + story-telling infra lines only).
- [ ] Committed, pushed; "kernel restart needed" flagged if `src/` changed.
- [ ] The audit ships with the hand-back.
