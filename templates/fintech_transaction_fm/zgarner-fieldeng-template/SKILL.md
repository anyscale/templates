---
name: zgarner-fieldeng-template
description: Zach Garner's writing and reviewing method for Anyscale demo templates and workshop notebooks. Use when authoring or reviewing ANY notebook prose, code comments, or structure he will read — and before every hand-back to him. Owns the craft and the collaboration protocol; the repo's `template` skill owns mechanics (BUILD.yaml, compute configs, tests, publish).
---

# Field-engineering template craft

This skill is a complete, transferable method built with Zach across the fintech_transaction_fm review sessions. It has three parts, and all three are load-bearing:

1. **`references/notebook-authoring.md`** — the method. Read it fully before writing anything he reviews. Part 1 (job-label sentence validation, power positions, the review loop) is the procedure; Parts 2–6 are the standards; Part 7 is the collaboration protocol whose file-safety rules have destroyed his work when violated; Part 8 is where the rules do NOT apply.
2. **`scripts/prose_lint.py`** — the mechanical pre-flight. `prose_lint.py <nb.ipynb>` greps markdown and comments for the named tells; `prose_lint.py --imports <nb.ipynb>` audits every `src/` import for size and hidden Ray. Both run before every hand-back; zero hits or fix them.
3. **The hand-back protocol** (end of the reference, mirrored in its checklist): diff → wip-commit → write fresh → review loop to fixpoint with a WRITTEN audit → lint → verify by papermill's own exit (bit-match if validated code moved) → commit+push → hand back with the audit shown.

## The method, in one line

Label every sentence's job (claim / fact / consequence / pointer / gloss / instruction), check the label against its position and the content against the label — the tell blacklist is cleanup, not the method — and never hand back without the written audit, the linter run, and a green verified run.

## The craft, in one line

Show the transferable Ray/Anyscale lesson inline; hide only what's incidental AND self-evident by name; motivate every step from the data; every number from a real run; keep the reader in the loop across the whole series.
