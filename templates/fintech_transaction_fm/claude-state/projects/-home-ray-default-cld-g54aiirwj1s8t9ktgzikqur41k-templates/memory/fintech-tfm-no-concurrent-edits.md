---
name: fintech-tfm-no-concurrent-edits
description: NEVER write to a notebook Zach is actively reviewing/editing — chat-first patches only; his unsaved buffer got destroyed once (2026-07-21)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1a07737d-42d1-41f6-8e46-6e24a82d9b87
  modified: 2026-07-21T16:57:37.645Z
---

2026-07-21: While Zach was manually reviewing nb02 in his editor, I ran repeated
edit→papermill→commit cycles on the same file and told him to "reload" after each.
His unsaved buffer (a rewritten intro line + an HTML technical note) was discarded
on reload — the work was lost with no disk trace (verified: papermill snapshots,
Jupyter checkpoint, VS Code history, git all clean of it).

**Why:** concurrent writes to a file he has open guarantee one side loses; "reload"
instructions are exactly when unsaved work dies. He was rightly upset.

**How to apply:** While he is reviewing/editing a file: do NOT Write/Edit/json.dump
it. Put proposed changes in chat as paste-ready blocks. Only touch the file after an
explicit hand-off ("i finished my edits you go") — and even then, diff the on-disk
file against HEAD first and fold his changes in. If he reports lost work, check (in
order): papermill/scratchpad snapshots, .ipynb_checkpoints/, ~/.vscode-server/data/
User/History/*/entries.json, git blobs.

See [[fintech-tfm-working-style]].
