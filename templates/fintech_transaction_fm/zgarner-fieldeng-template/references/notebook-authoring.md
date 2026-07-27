# Authoring a template notebook — the field-engineering layer

This is the template-specific half of a two-skill pair. **Prerequisite: the `zgarner-prose` skill** — its `writing-method.md` owns sentence validation, voice, the tell catalog, truth rules, and comment craft; nothing here works without running that method. This document owns what only exists in a template notebook: the show/hide boundary, Ray visibility, notebook structure, outputs, and the collaboration protocol. Built with Zach across the `fintech_transaction_fm` reviews (July 2026). Repo mechanics (BUILD.yaml, tests, publishing) belong to the templates repo's own `template` skill.

---

# Part 1 — What to show, what to hide

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
- **~25–30 lines is the ceiling** for an inline function before it reads as a wall; three visual blocks (setup / loop / report) is the accepted fix, plus splitting multi-purpose cells.
- **The `--imports` audit is mechanical**: `zgarner-prose/scripts/prose_lint.py --imports <nb.ipynb>` lists every `src/` import with line count and Ray content — the facts for the show-or-hide call, run every time code changes.
- **The Ray Note convention**: prose numbers the Ray integration points ("adapted for Ray in three places"), and code comments reference them by name — `# See Ray Note #1 above` — never a bare number. The prose must introduce that exact name.

# Part 2 — Structure of a notebook

## The intro pattern

Recap first, at an altitude a returning reader absorbs without homework ("Previously in Part 2, we built the train/validation/test splits" — not detail they must remember). Then why this notebook exists, then the roadmap in one or two sentences. When notebooks form a group, the intro places the reader in the group (recap → this notebook's role in the arc → the later pointer). Content ownership applies to intros too — a recurring-cost sentence was cut from an intro because Scaling factors owned recurrence.

## Sections

- **One section per activity.** Consecutive sections must not repeat the same noun — three headers about "the split" is one activity wearing three hats; merge into one section with `###` steps, each step a short lead plus the code cell that does exactly that step.
- **Titles: one action verb plus a concrete object** ("Write the three splits"), never a double-verb compound, a "Why X" title, a question, or a static label ("The model" → "What we're training"). The test for every heading and lead: delete the mechanism words — if no point remains, rewrite.
- **A section you keep patching is a section to question.** Ask which notebook or section owns its content; if another does, delete it and leave a bridge sentence. The tell: every cell you try under the heading feels wrong — the heading is wrong, not the cells.
- **Show why, not just what.** Motivate each choice from the data or the problem (*amounts span orders of magnitude → bucket them*; *fraud is ~0.1% → AP, not accuracy*). The justification is markdown around the lesson.
- **Numbers get ONE owner section.** The full run's steps and hours belong to Scaling factors; other sections reference, never restate.
- **Don't over-prove — receipts are for the repo, not the reader.** One sentence of verification with a pointer is the ceiling ("verified byte-identical to NVIDIA's original; the checks are in `scripts/`"). Zach: "no one is standing around in disbelief." A weak check that needs an antidote paragraph to not mislead gets replaced by a plain artifact check, not defended harder.
- **Show the intermediate result.** If a step computes something the reader can look at, show it — moving the cutoff computation out of a cache guard bought a visible result and the autoscaler's node-arrival lines in the committed output.
- **Real examples beat invented ones.** Find the pattern in the data and display it (card 66000's Texas→Mexico fraud burst replaced a hypothetical $900 purchase); hardcode the chosen example with a comment saying why, so prose and display can't desync across scales.
- **Every notebook verifies what it built** before the next depends on it ("Check the training set," "Check the windows," "Check the embeddings") — counts, shapes, one concrete peek.

## Around big code

- **Large code cells get their own `###` header** — structure stays visible while scrolling, and headed sections collapse in Jupyter. (Zach's rule.)
- **After big code, re-orient before advancing.** The reader loses the thread inside a long cell; the author never does. Zach's template: "In the last coding section we built the PyTorch training function, integrated with Ray for distributed training. Next, we count the total number of training steps — the learning-rate schedule needs it before training starts." Recap, then next, with the reason attached.
- **No cell-inventory transitions** ("Most of the cell below is `train_loop_config`…"). Say the action and let the cell show itself.
- **Prose carries concepts; code carries names.** Filenames, function names, API names, seeds: banned from concept-level bullets and takeaways, mandatory at their line in the code. Zach: "save the detaily stuff for the code. I need higher level."

## The Scaling factors section — the established pattern

Every notebook 02+ ends its technical content with "## Scaling factors." Open on the measured fact or the concrete limit — never a frame ("The scaling problem is X", "Ray's answer is Y" are labels posing as sentences; state the facts and let them argue). Body: what breaks and when, resource named (RAM, network bandwidth, GPU memory, cores), magnitude attached. The table format: `What grows | The limit it hits | What absorbs it | Measured at full` — every number from a real run, "—" where unmeasured. Then the 10× arithmetic (the same pool takes N× longer; N× the workers brings it back) and the fact that only a config line changes. Recurrence, elasticity, and GPU-vs-CPU are facts when true, never sales beats.

## Takeaways — altitude validated by a reverted rewrite

Evidence: commit `41454b14`, a takeaways rewrite reverted wholesale. What the accepted versions do:

- **No metric re-argument** — the metrics section owns those numbers; the takeaway states what exists now.
- **Product meaning first.** His edit: "We trained the foundation model, and its now ready to be used for prediction systems like fraud detection. All of the later approaches build on top of this foundation model."
- **Ray is one clause at this altitude** ("wrapped in Ray, which handles the distributed scaling") with one concrete fact (`ScalingConfig`); the enumeration belongs next to the code.
- **No API inventories, no line-count asides, no timing details** (the header's time-to-complete line owns those).
- **Lead with the transferable Ray lesson, then the domain observations.**
- When the takeaway names output artifacts, point to where they're opened next — and honor that pointer in the next notebook.

## The Next blurb is one plain sentence

What the reader does next, in words they already own. No class names, no magic numbers, no feature lists — those belong in the next notebook, where they get explained.

# Part 3 — Outputs, plots, and it-must-run

- **Committed outputs (working branch) are curated**: real results plus the infra lines that tell the Ray story (autoscaler node arrivals); never progress bars, logger spam, or float noise (`round(float(x), 3)`). Display slices exact — 27 tokens is two full transactions, not "~2". The publish pipeline strips outputs, so code + prose + a described expected result must carry the story alone.
- **Plots: "make it look better" means styling, not structure** — theme, kill chartjunk, human-format axes (600000 → 600k); never silently re-axis. But the plot must show its point: log the axis when a tail is invisible.
- An unescaped `$` in Jupyter markdown triggers MathJax and garbles everything to the next `$`. Escape amounts: `\$57.20`.
- **Committed defaults execute at CI/mini scale** (usually CPU); scale-up is one obvious knob left at the runnable setting.
- **The proof is a green papermill run, checked by papermill's OWN exit code** — piped chains have hidden a `NameError` as green. Scan the executed notebook for `output_type == "error"`.
- **After moving or changing an import, re-run the WHOLE notebook** — a later cell may use the symbol you relocated.

# Part 4 — Working with Zach (the collaboration protocol)

## His text and his corrections

- **His text is the baseline.** Preserve his sentences verbatim; never revert his wording; flag typos once, never silently fix. He drafts with `[tbd]` markers for you to fill — fill the marker, preserve his frame.
- **Fact-check his technical claims against the code and correct with evidence** — he asks for it ("Fact check me"). Example: "worker count (autoscaled by Ray)" → the cluster autoscales nodes; the worker count is fixed.
- **His messaging hierarchy is inviolable** — which result is the headline, what's the control: once he sets it, every table, takeaway, and summary follows it (the project-specific hierarchy is in project memory).
- **Bias to action; ask questions in prose, never the multiple-choice widget** (rejected twice). When a call is yours, decide and flag it.
- His correction shorthand is in the `zgarner-prose` SKILL.md; when he names a new pattern, codify it in that skill and its linter the same day.

## The file-safety rules (violating these destroyed his work)

1. **Never write a file he is editing.** Chat-first patches (paste-ready blocks) until an explicit hand-off ("i finished my edits, you go"). Every "reload the notebook" you ask of him is a chance for his unsaved buffer to die — his nb02 work was destroyed exactly this way.
2. **Verify "it's saved" against disk** (`git status`) — his editor's saves lag; his nb03 corpus purge was silently lost this way and only recovered because the linter re-caught the banned word.
3. **Commit whatever is on disk before any write** (`wip:` commits are fine).
4. **Re-load and re-diff at write time** — never hold a loaded copy across a background run and then dump it over the file.
5. The node is ephemeral: **commit AND push promptly**; durable notes go in repo files; `./setup_claude.sh backup` snapshots memory/settings.
6. If he reports lost work, search in order: papermill/scratchpad snapshots, `.ipynb_checkpoints/`, `~/.vscode-server/data/User/History/*/entries.json`, git blobs.
7. Tell him **"kernel restart needed"** whenever `src/` changed — Python won't re-read a loaded module, and the resulting ImportError looks like your bug.

## The hand-back protocol — every time, in order

1. `git diff` the file; wip-commit what's on disk.
2. Write, re-loading at write time.
3. Run the `zgarner-prose` review loop to fixpoint, including the written job-label audit.
4. Run `prose_lint.py` on the notebook, and `--imports` when code changed. Zero hits or fix them.
5. Verify: papermill at mini by its own exit + error-cell scan; bit-match when validated code moved; graft curated outputs.
6. Commit and push immediately.
7. Hand back WITH the audit shown — verdicts he can check, not conclusions he must extract. After any correction: sweep the whole notebook for the pattern before returning.

# Part 5 — Template-specific counter-rules and unconfirmed notes

- **Detail survives in code that dies in prose** — banned from concept bullets and takeaways, mandatory at its line in the code.
- **[Ray][verb] is correct when Ray is truly the actor** — don't demote the platform from subject position in a Ray workshop.
- *(Unconfirmed, inferred from acceptance)*: autoscaler node-arrival lines are worth keeping in committed outputs; the NVIDIA punchline survives in takeaways when the facts deliver it; tables are the preferred Scaling-factors body; known linter backlog in closed pages (nb01 "the pretrain corpus"; ~11 hits in nb02/nb03, several in his own approved text) awaits his call.

# The checklist (template layer — run the zgarner-prose checklist too)

- [ ] Every inline cell carries the transferable Ray/Anyscale lesson; hidden helpers pass the look-it-up test; sizes disclosed; no never-runs branches.
- [ ] Ray visible where Ray is the lesson; `--imports` audited when code changed; notebook and headless path compose the same helpers.
- [ ] Intro recaps and places the notebook in the arc; sections are one activity each; numbers have one owner; verification is one sentence + pointer.
- [ ] Big code has `###` headers; re-orientation follows it; concept bullets carry no filenames or API names.
- [ ] Scaling factors follows the pattern; takeaways at product altitude; Next is one plain sentence honoring no jargon.
- [ ] Papermill green by its own exit; whole-notebook re-run after import changes; bit-match if validated code moved; outputs curated.
- [ ] The hand-back protocol ran in order; the audit ships with the hand-back.
