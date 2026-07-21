# Authoring a template notebook

The craft of *what a good Anyscale demo-template notebook shows, hides, and proves.* Read it while authoring or reviewing notebook content — including the first draft `anyscale-template-agent` produces; these are the judgments to apply to it. (Mechanics — BUILD.yaml, compute configs, tests, publishing — live in the anyscale/templates repo's own `template` skill; this is only the craft.)

## The one question: what to show, what to hide

A template teaches a **transferable lesson** — the Ray/Anyscale pattern the reader came to learn and will reuse. Every block of code is either *that lesson* or *incidental* to it. Show the lesson; hide the incidental.

Test each block: **"Is this teaching the pattern, or is it plumbing specific to this dataset/domain?"**

- **Show** (inline in the notebook) — the primitive the template is about (`ray.data` pipeline, `TorchTrainer` + `ScalingConfig`, `@serve.deployment`) and the handful of verbs that convey its shape.
- **Hide** (import from `src/`) — dataset/domain munging, parsing, boilerplate; anything long that the reader won't reuse.

**The failure mode is drawing the boundary in the wrong place — hiding the lesson *inside* an incidental wrapper.** A single `prepare_data()` import looks clean, but if loading the data is *itself* a distributed Ray Data job (read → transform → aggregate → write), that import buries the most on-thesis moment in the notebook. The reader sees `prepare_data()` and learns nothing transferable.

### Canonical example — `fintech_transaction_fm`, "Load & explore the data"

Loading the TabFormer dataset *is* a Ray Data pipeline. The first draft hid all of it behind one opaque `from src.tabformer import prepare_tabformer` call — in a notebook whose entire point is data loading at scale. The fix redraws the line:

- **Shown inline:** `read_csv → map_batches(normalize_batch) → groupby("card_id").map_groups(card_statics) → write_parquet` — the four verbs, the distributed shape, and the "driver never materializes 24M rows" point.
- **Hidden in `src/tabformer.py`:** the per-row munging (`"$57.20" → 57.20`, MCC → category, modal home-state) — dataset-specific, not a Ray lesson.

## Show inline without forking logic

When the notebook shows an inline version and a script/job needs a headless version, **both compose the same `src/` helpers** — never two copies of the logic that can drift.

Pattern: put the *callbacks* and *steps* in `src/` as public functions; the notebook composes them in the open; the headless entry point (`prepare_*`, a `scripts/` stage, the job) composes the **identical** sequence. In the example, `prepare_tabformer()` was refactored to call the same `normalize_batch` / `sample_cards` / `card_statics` / `attach_statics` the notebook shows — so the walkthrough and the job can't produce different data.

`src/` is the **hide bucket**; the notebook is the **show surface**. Reach for an import when code is (a) long, (b) incidental, or (c) must run identically headless. Inline it when it carries the lesson.

**But hiding isn't free — a name the reader must look up is worse than the code itself.** A helper earns its place only when its *call site* reads clearly: name + args tell the reader what happens without opening `src/` (`ensure_download`, `normalize_batch`, `sample_cards`). If you'd invent a wrapper like `write_temporal_splits()` just to shorten a cell — and the reader would have to go read it to know what it does — the hiding cost more than it saved; show the few lines inline. The deciding question is never "is this incidental?" but **"would the reader have to look it up?"** Hide what's incidental *and* self-evident by name; inline what's incidental but short or would need an opaque new name.

## Show *why*, not just *what*

A shown cell should answer **"what am I seeing and why does it matter,"** not narrate the API. Motivate each engineering/modeling choice from the data or the problem (*amounts span orders of magnitude → log-bucket them*; *fraud is ~0.1% → report PR-AUC, not accuracy*; *workers run on other nodes → checkpoint to shared storage*). That justification belongs in the markdown around the lesson — not as a wall of comments inside hidden code, and not as prose with no code to anchor it.

## Verify the claim against the data — never assert the shape from memory

"Show *why*" only works if the *why* is true. The fastest way to lose a sharp reader is a confident, wrong characterization of the data: calling a distribution "heavy-tailed" when it's a tame lognormal (top 1% held only ~10% of the mass, max ~200× the median); "most cards are quiet" when the median card had 2,500 transactions; "then nothing for weeks" when 99.5% of gaps were under a week. Every one of those was a one-line check away, and every one shipped in a first draft.

**Rule: if a sentence names a shape, a magnitude, or a rate, compute it before you write it.** `df.describe()`, a quantile, a `value_counts()`, "what share of the mass is in the top 1%" — all cheap. A shipped wrong adjective is not: it's the exact thing a reader catches, and it discredits the real lesson sitting next to it. This applies *doubly* to plot captions and section prose an agent generated from a template — those are guesses until the numbers confirm them.

And the shape words are not interchangeable color: **heavy-tailed** (fat tail carrying real mass), **long-tailed** (thin tail stretching far along the axis), and **right-skewed / lognormal** are different shapes with different modeling consequences. Use the one the data shows, not the one that sounds impressive.

## The closing cell banks the *transferable* lesson, not dataset trivia

The takeaways/summary cell is where the reader files away "what do I reuse." A Ray Data notebook whose takeaways are three bullets about transactions and zero about Ray has thrown away its own point. **Lead the summary with the primitive** — the `read_csv → map_batches → groupby → write_parquet` streaming shape, "the driver never materializes the full table," "same code from mini to full, only the config changes" — then the domain/modeling observations second. If you organized the notebook around a Ray lesson, the closer has to bank it.

## Purpose before mechanism — for every heading *and* lead sentence

The single most common failure: leading with *how the code works* instead of *what it's for and produces*. It shows up two ways, and both must be fixed.

**Headings** name the decision/output the reader takes away, not the phenomenon or the API. "Class imbalance" describes the data; **"How we measure performance"** is the lesson it drives. "One card at a time, across the cluster" is flavor; **"Turn each card's history into token sequences"** says what the cell produces. Name the section for the temporal split, not for "transaction volume rose over time."

**Lead sentences** state purpose/output before any mechanism. *"The cell below groups the transactions by `card_id` and runs one function per card with `map_groups`"* leads with plumbing — the reader doesn't yet know what's being built or why they'd care. *"This cell is the tokenizer: it turns each card's raw transactions into the integer sequences the model trains on"* leads with the point; the `map_groups`/grouping/stateless details come **after**.

The test to run on every heading and first sentence: **"Does this say what the reader is getting before how it's built? If I deleted the mechanism words (`groupby`, `map_groups`, 'one function per card'), is there still a point left?"** If not, rewrite. Mechanism is the second sentence, never the first.

Tells that you've inverted it: a heading that's a chapter-epigraph phrase ("One card at a time…"), or a sentence starting "The cell below…", "This section covers…", "Here we group/call/run…".

Two more heading rules from review. **Consecutive sections must not repeat the same noun** — "Why the split is temporal" / "The split as a Ray Data pipeline" / "The train split at a glance" is one activity wearing three headers; merge into one section with `###` steps, each step = short lead + the code cell that does exactly that step. **A title is one action verb plus a concrete object** ("Write the three splits") — never a double-verb compound ("Write the parts and draw the evaluation samples"); the second activity is explained inside.

## Voice: write like an engineer to a peer, not a content model

Calibrate against this sample Zach wrote himself (an opening, presentation register — but the qualities transfer to notebook prose):

> Transaction foundation models are the latest generation of transformer models - like LLM's, but instead of language, they are focused on financial transactions. This lets transaction foundation models recognize distinct patterns like fraud, that traditional ml techniques can't detect. Today I'm gonna show you how to build your own transaction foundation model and achieve performance and scalability that surpasses comparable approaches by Nvidia.

What it does: defines the new thing **by analogy to a known thing, in one breath** — not a formal bolded-term definition, not a company name-drop list. Each sentence advances the reader: what it is → why you care → what you're getting. First person, direct, confident claim, zero throat-clearing. If a draft opening reads denser or more "impressive" than this, it's wrong.

**Workshop register is action tone: lead with the task, not a description.** "The 80/10/10 boundaries are positions in time…" reads like documentation; **"We need two dates: the day by which 80% of all transactions have happened…"** reads like someone running a workshop. Open steps with *We need / We do / Now we*, keep the verbs on us, and let the mechanism arrive as the way we do the thing — never as the subject of the sentence. (Zach: "Speaking with more action tone instead of passive is better for a workshop.")

**Connect the logical chain — no gap between goal and mechanism.** A paragraph that states a goal ("cutoffs are defined by counts") and then names an operation ("groupby by date") without the middle link ("we need each day's count, plus a running total over the days in order") reads as two unrelated facts. Walk goal → what that requires → the operation that provides it. If the reader could ask "what does that have to do with it?", the link is missing.

**One word per concept, held for the whole notebook.** If the intro calls them "splits," the section titles, prose, and prints call them splits — never "parts" in one place and "splits" in another.

**The deeper AI tell is staging, not sentence length.** A draft was rejected twice for the same disease in two disguises: first as long em-dash-chained essay sentences with cute asides ("deserve a word of honesty," "deliberately boring"), then — after shortening — as *theatrical* short sentences: the dramatic negation-hook opener ("The foundation model never sees a fraud label."), the beat-drop mini-sentence for rhythm ("Similar transactions get similar embeddings."), the designed statement-then-elegant-elaboration arc. Every sentence was performing. Human engineer prose is informational: the subject comes first, facts arrive in the order you'd say them out loud, and nothing is staged for effect. Test: does the sentence exist to carry information, or to land? If it lands, rewrite it to carry.

Generated demo prose has telltale filler patterns that make a sharp reader trust the content less. Cut them:

- **Editorializing titles** — "Class imbalance — and why we don't report plain accuracy." Name the thing: "How we measure performance."
- **The `**Label**:` bullet list** where every item is a bold noun + colon ("**Metric**: …", "**Sampling**: …"). Write sentences.
- **Filler connectives** — "drives the rest of the series," "it's worth noting that," "the operationally meaningful number."
- **Raising a concept only to dismiss it** — don't introduce AUC-ROC just to say you don't use it. If it isn't load-bearing, cut it.
- **Naming a term then waving at it** — name the real term (`importance weighting`) *and* gloss it concretely ("keep 1 in 50 normals, weight each survivor ×50"), not with more abstraction ("counts for the many it represents").

The test for any sentence: would an engineer write this to another engineer, or does it read like it's filling a section template?

## Code comments: one step, one comment — and answer at the line that asks

**A block comment explaining several chained lines is the tell that the chain should be split.** Three "rambling" comment lines above a three-step chain means: break the chain into three statements and give each a short comment of its own. One idea per comment, sitting on the line it describes. (Zach: "Speak plainly, one step at a time. You have three rambling lines of text that is supposed to explain a couple lines of code.")

**Put the answer at the line that raises the question.** The comment on `read_parquet` is where "how does it know how many workers?" gets answered (it doesn't — one task per shard, scheduled on whatever CPUs exist, autoscaler adds more when tasks queue). The laziness of a Ray dataset is explained at the `to_pandas()` that triggers execution — not as an abstract claim three lines earlier that the very next statement silently falsifies. Never leave a comment whose truth expires within the same cell.

**Names must read from the call site — and say the right provenance.** A generic name (`normalize_batch`) sounds like a black box and forces the reader to ask; a concrete one (`normalize_date_column`, `add_analysis_columns`) doesn't. Verbs carry provenance: "add_date_column" made the reviewer ask *where the date came from* — "normalize" says it's derived from fields already there. If a reviewer stops to ask what a function is, the name is wrong, whatever the docstring says.

## Show the intermediate result, and use real examples

**If a step computes something the reader can look at, show it.** The cutoff dates were computed inside a cache guard and never printed — moving the aggregation into its own always-run cell costs one worker-side scan and buys a visible result (`train < 2017-04-17 <= val < 2018-09-29 <= test`) plus, in the committed output, the autoscaler bringing CPU nodes up: the "declare work, hardware arrives" story in the artifact itself.

**A real example from the data beats an invented one.** The one-card sequence showed card 0 (nothing to see) while the prose invented a hypothetical $900 purchase. One query found card 66000: routine Texas purchases, then a same-day burst of Mexico department-store charges escalating \$45 → \$514 — the series' sequence-context thesis, visible in ten rows. When the prose describes a pattern, find the pattern in the data and display it; hardcode the chosen example (with a comment saying why it was chosen) if a programmatic pick could select a different instance at another scale and desync the prose from the display.

**Ray must be visible where Ray is the lesson.** `src/` helpers for taught stages hold pandas per-batch functions and sealed reference code only — never the `ray.data` calls themselves. A wrapper like `load_normalized()` that hides read + transform + filter obfuscates exactly what the notebook exists to teach ("we cannot be obfuscating ray usage because part of the task here is showing ray"). Corollary: use each Ray tool for its own job in the open — `filter(expr=col(...) == …)` for row predicates, `map_batches` for transforms — and if committed outputs are kept in the working branch, curate them to the informative lines (real results, plus infra lines that tell the Ray story, like autoscaler scale-up; never progress bars and logger spam).

## Plots: restyle, don't restructure — and make the point visible

Two different jobs; don't confuse them:

- **"Make it look better" means change the *styling*, not the *structure*.** Theme it (seaborn `set_theme(style="white")` + `despine`), kill chartjunk (gridlines), human-format the axes (`600000 → 600k`). Do **not** silently re-axis it — relabeling a log-x with hand-written ticks (`$0.10, $1, $10`) changes *what the reader is looking at*, and they'll (rightly) call it weird. Keep the axes they expect.
- **But the plot must actually show its point.** A long-tailed quantity on a linear y-axis is one tall bar and an invisible tail — put the y-axis on a log scale so the tail is visible. Revealing-the-point is fair; restructuring for its own sake is not.

Gotcha: an unescaped `$` in a Jupyter **markdown** cell triggers MathJax and silently garbles everything between two dollar signs. Escape amounts as `\$57.20`. (Code cells and backtick spans are safe.)

## It must run — outputs don't ship

The anyscale/templates repo **strips notebook outputs** before commit (a `clear-notebook-outputs` pre-commit hook) and the test re-runs the notebook end-to-end with **papermill** to prove it executes. Consequences for how you author:

- **Committed defaults must execute in the test environment** — the template's compute at CI/mini scale, usually **CPU**. Never commit the author's `use_gpu=True` or large-scale config as the default; expose scale-up through one obvious knob (`SCALE`, `num_workers`, `ScalingConfig`) and leave it at the runnable setting.
- **Don't rely on committed outputs to tell the story.** The reader sees code + prose first and runs it themselves. Render plots/numbers in-cell so they appear on run, *and* state the expected result in prose so a reader knows what "working" looks like before they execute.
- **The proof of correctness is a green papermill run, not a screenshot.** If a stage can't run at mini scale in CI time, shrink it (fewer cards/epochs/rows) rather than committing a version only a GPU cluster can execute.
- **Trust papermill's *own* exit code, not a chained command's.** `papermill … > log 2>&1; echo done` reports the `echo`'s exit (0) and hides a failed run — a notebook that raised `NameError` looked "green." Read papermill's exit directly, or scan the executed notebook for cells with `output_type == "error"`. A false green is worse than no check.
- **After moving or changing an import, re-run the *whole* notebook.** A later cell may still use the symbol you relocated. Rewriting one cell's imports silently broke a downstream cell that used `STATIC_FIELDS`; only a full top-to-bottom run caught it. Editing any cell means re-verifying all of them, not just the one you touched.

## Checklist

- [ ] Every inline cell carries a transferable Ray/Anyscale lesson; domain munging is imported from `src/`.
- [ ] No lesson is hidden inside an incidental wrapper.
- [ ] Inline-shown logic and any headless entry point compose the **same** `src/` helpers.
- [ ] Each shown step says *why it matters*, motivated from the data/problem.
- [ ] Every claim about the data's shape/magnitude/rate was **computed, not assumed** (and the shape word is the right one).
- [ ] The closing/takeaways cell leads with the transferable Ray/Anyscale lesson, not dataset trivia.
- [ ] Every heading and lead sentence leads with purpose/output, not mechanism — delete the API words and a point still remains.
- [ ] Prose reads like an engineer wrote it — no editorializing titles, `**Label**:` lists, or concepts raised only to dismiss them.
- [ ] Plots are styled not restructured, show their point (log scale for tails, human-formatted axes), and `$` is escaped in markdown.
- [ ] Committed defaults run top-to-bottom under papermill at CI/mini scale (CPU); scale-up is one knob.
- [ ] Verified by papermill's own exit code / zero `error` output cells — not a chained command's exit — and the *whole* notebook re-ran after any import change.
- [ ] Expected results are described in prose (outputs will be stripped on commit).
- [ ] Prose is in action tone (We need / We do), the goal→mechanism chain has no gaps, and one word per concept holds notebook-wide.
- [ ] Chained code with a block comment is split: one statement, one short comment, answers at the line that raises the question.
- [ ] Function names read from the call site with honest provenance; no Ray calls hidden in `src/` for taught stages.
- [ ] Intermediate results are shown; examples come from the data (chosen instance hardcoded + justified), not invented.
