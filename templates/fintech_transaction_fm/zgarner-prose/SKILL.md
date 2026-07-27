---
name: zgarner-prose
description: Zach Garner's general technical-writing method — sentence validation by job, power positions, the tell catalog, voice register, truth rules, comment craft. Use for ANY prose an engineer will read that Zach will review — notebooks, docs, decks, chat replies, code comments. The zgarner-fieldeng-template skill layers notebook-specific craft on top of this one.
---

# Zach's prose method

General technical writing to Zach's bar. Two load-bearing pieces:

1. **`references/writing-method.md`** — read fully before writing anything he reviews. Part 1 is the method (job-label sentence validation, power positions, the written opener audit, the review loop, sweep-don't-spot-fix); Parts 2–5 are voice, the tell catalog, truth, and comment craft; Part 6 is where the rules do NOT apply.
2. **`scripts/prose_lint.py <file.ipynb>`** — greps the mechanical tells in markdown and code comments. Run before every hand-back; zero hits or fix them. (`--imports` mode is used by the template skill's show-or-hide audit.)

## The method, in one line

Label every sentence's job (claim / fact / consequence / pointer / gloss / instruction), check the label against its position and the content against the label, run the loop to fixpoint with a WRITTEN audit — the tell blacklist is cleanup, not the method.

## His correction shorthand

"skeeze" = performed or salesy prose of any kind. "sandwich" = verdict — whispered reason — so consequence. "dash inventory" = a stapled parts list. "blah blah blah" = the sentence's shape is noise. When he names a new pattern, codify it in the reference and the linter the same day.
