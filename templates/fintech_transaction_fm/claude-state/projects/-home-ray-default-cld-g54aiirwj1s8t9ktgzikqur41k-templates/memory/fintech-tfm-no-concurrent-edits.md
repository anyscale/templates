---
name: fintech-tfm-no-concurrent-edits
description: NEVER write to a notebook Zach is actively reviewing/editing — chat-first patches only; his unsaved buffer got destroyed once (2026-07-21)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1a07737d-42d1-41f6-8e46-6e24a82d9b87
  modified: 2026-07-21T17:00:03.453Z
---

2026-07-21: While Zach was manually reviewing nb02 in his editor, I ran repeated
edit→papermill→commit cycles on the same file and told him to "reload" after each.
His unsaved buffer (a rewritten intro line + an HTML technical note) was discarded
on reload — the work was lost with no disk trace (verified: papermill snapshots,
Jupyter checkpoint, VS Code history, git all clean of it).

**Why:** concurrent writes to a file he has open guarantee one side loses; "reload"
instructions are exactly when unsaved work dies. He was rightly upset.

**How to apply (Zach 2026-07-21, furious: "you could have fucking committed or
anything. You need to reread the fucking notebook before blowing it over."):**
1. COMMIT FIRST: before ANY write to a repo notebook, commit whatever is on disk
   (even `wip: on-disk state before edits`) so disk state is always recoverable.
2. REREAD AT WRITE TIME: `git diff` + re-load the file immediately before dumping,
   not at hand-off time — NEVER hold a loaded copy across a background papermill
   and then json.dump it (that window silently destroys his saves).
3. While he is reviewing/editing a file: do NOT Write/Edit/json.dump it at all.
   Put proposed changes in chat as paste-ready blocks. Only touch the file after an
   explicit hand-off ("i finished my edits you go").
4. If he reports lost work, check (in order): papermill/scratchpad snapshots,
   .ipynb_checkpoints/, ~/.vscode-server/data/User/History/*/entries.json, git blobs.

See [[fintech-tfm-working-style]].
