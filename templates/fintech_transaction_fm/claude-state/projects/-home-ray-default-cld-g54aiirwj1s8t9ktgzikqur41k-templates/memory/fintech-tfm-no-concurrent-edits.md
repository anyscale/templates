---
name: fintech-tfm-no-concurrent-edits
description: File-safety protocol with Zach — now codified in the repo skill; this is the pointer + incident record
metadata:
  type: feedback
---

The full protocol is CODIFIED in the repo (durable):
`zgarner-ai-skills/zgarner-fieldeng-template/references/notebook-authoring.md`
(github.com/zachgarner/zgarner-ai-skills, checked out inside the template dir),
Part 7 "The file-safety rules" + "The hand-back protocol". Follow that document.

Incident record (why it exists): 2026-07-21, concurrent edit→papermill→commit cycles
plus "reload" instructions destroyed Zach's unsaved nb02 buffer (an HTML technical
note + intro rewrite) with no disk trace. 2026-07-23, his saved-to-buffer nb03 corpus
purge was silently lost the same way and recovered only because prose_lint re-caught
the banned word. His editor's saves LAG — always verify disk (`git status`) before
believing "it's saved".

See [[fintech-tfm-working-style]].
