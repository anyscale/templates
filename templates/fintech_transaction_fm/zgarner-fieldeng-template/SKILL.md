---
name: zgarner-fieldeng-template
description: Zach Garner's field-engineering craft for Anyscale demo templates — the notebook-specific layer on top of the zgarner-prose skill. Use when authoring or reviewing a template notebook: what to show vs hide, Ray visibility, notebook structure, outputs, and the collaboration protocol. REQUIRES zgarner-prose (sentence method, voice, tells, linter).
---

# Field-engineering template craft

The template-specific layer. **Load `zgarner-prose` first** — it owns the sentence method, voice, the tell catalog, truth rules, comment craft, and `scripts/prose_lint.py`. This skill owns what only exists in a template notebook.

1. **`references/notebook-authoring.md`** — Part 1: the show/hide boundary and Ray visibility (refactor src when Ray is buried; bit-match acceptance for refactors of validated code). Part 2: notebook structure (intro/arc pattern, one section per activity, Scaling factors, takeaways altitude, Next blurbs). Part 3: outputs, plots, papermill. Part 4: the collaboration protocol — file-safety rules that have destroyed his work when violated, and the seven-step hand-back. Part 5: counter-rules and unconfirmed notes.
2. **The show-or-hide audit**: `../zgarner-prose/scripts/prose_lint.py --imports <nb.ipynb>` — every src import with size and Ray content, run whenever code changes.

## The craft, in one line

Show the transferable Ray/Anyscale lesson inline; hide only what's incidental AND self-evident by name; motivate every step from the data; every number from a real run; verify by papermill's own exit; and follow the hand-back protocol every single time.
