---
name: zgarner-fieldeng-template
description: Zach Garner's field-engineering craft for building compelling Anyscale demo templates. Use when authoring or reviewing a template notebook or technical demo — what to show vs. hide, motivating each step, keeping defaults runnable. Complements the anyscale/templates repo's `template` skill, which owns the mechanics (BUILD.yaml, compute configs, tests, publish); this owns the craft.
---

# Field-engineering template craft

A good Anyscale demo template isn't just correct — it *teaches the reader the pattern they came for* and earns trust by running. The repo's `template` skill handles the mechanics (BUILD.yaml, configs, testing, publishing). This skill is the **craft**: the judgment calls that separate a template that lands from one that's merely accurate.

Apply it when authoring a notebook from scratch, reviewing a first draft (including one from `anyscale-template-agent`), or improving an existing template.

## The craft, in one line

**Show the transferable Ray/Anyscale lesson; hide everything incidental — and make sure the whole thing runs.**

## References

- `references/notebook-authoring.md` — the core discipline: what to show inline vs. hide in `src/`, composing instead of forking logic, showing *why* not just *what*, and keeping committed defaults runnable under the test. Read it before/while writing notebook content.

(Add more references here as the playbook grows — narrative arc, demo structure, the scale story, choosing the headline result.)
