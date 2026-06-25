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

A shown cell should answer **"what am I seeing and why does it matter,"** not narrate the API. Motivate each engineering/modeling choice from the data or the problem (*amounts are heavy-tailed → log-bucket them*; *fraud is 0.1% → report PR-AUC, not accuracy*; *workers run on other nodes → checkpoint to shared storage*). That justification belongs in the markdown around the lesson — not as a wall of comments inside hidden code, and not as prose with no code to anchor it.

## It must run — outputs don't ship

The anyscale/templates repo **strips notebook outputs** before commit (a `clear-notebook-outputs` pre-commit hook) and the test re-runs the notebook end-to-end with **papermill** to prove it executes. Consequences for how you author:

- **Committed defaults must execute in the test environment** — the template's compute at CI/smoke scale, usually **CPU**. Never commit the author's `use_gpu=True` or large-scale config as the default; expose scale-up through one obvious knob (`SCALE`, `num_workers`, `ScalingConfig`) and leave it at the runnable setting.
- **Don't rely on committed outputs to tell the story.** The reader sees code + prose first and runs it themselves. Render plots/numbers in-cell so they appear on run, *and* state the expected result in prose so a reader knows what "working" looks like before they execute.
- **The proof of correctness is a green papermill run, not a screenshot.** If a stage can't run at smoke scale in CI time, shrink it (fewer cards/epochs/rows) rather than committing a version only a GPU cluster can execute.

## Checklist

- [ ] Every inline cell carries a transferable Ray/Anyscale lesson; domain munging is imported from `src/`.
- [ ] No lesson is hidden inside an incidental wrapper.
- [ ] Inline-shown logic and any headless entry point compose the **same** `src/` helpers.
- [ ] Each shown step says *why it matters*, motivated from the data/problem.
- [ ] Committed defaults run top-to-bottom under papermill at CI/smoke scale (CPU); scale-up is one knob.
- [ ] Expected results are described in prose (outputs will be stripped on commit).
