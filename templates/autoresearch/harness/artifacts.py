"""R8 — artifact safety: never delete, move aside.

Paid for with a real scar: a deleted embeddings dir cost a $4 / 50-min GPU re-extraction, and
a failed rerun once moved winning artifacts aside then died before writing replacements
(`AUTORESEARCH.md` Iron Rule #4). This module is the one primitive every rerun uses instead of
`rm -rf`: move each prior artifact to `<name>_old_<stamp>` under a single shared stamp, so a
rerun's model / embeddings / tokenized / downstream outputs all move together and stay
restorable.

Guarantees:
- **Never deletes.** The only operation is rename-aside; recovery is always possible.
- **Idempotent / guarded.** If the backup target already exists (a double-submit), it is a
  no-op — it does NOT clobber the existing backup. A double-fire is safe, not a disaster.
- **One stamp per rerun.** Pass one stamp for all artifact kinds so they're grouped.
"""

from __future__ import annotations

import os
import time


def make_stamp(t: float | None = None) -> str:
    """A filesystem-safe timestamp for one rerun. Pass `t` (epoch seconds) for determinism."""
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t))


def aside_path(path: str, stamp: str) -> str:
    return f"{path.rstrip('/')}_old_{stamp}"


def move_aside(paths, stamp: str) -> list:
    """Move each existing path to `<name>_old_<stamp>`. Returns the (src, dst) moves made.

    Missing paths are skipped (nothing to protect). If the destination already exists, the
    move is skipped rather than overwriting an existing backup — so re-running move_aside with
    the same stamp is a safe no-op, never data loss.
    """
    moves = []
    for p in paths:
        if not os.path.exists(p):
            continue
        dst = aside_path(p, stamp)
        if os.path.exists(dst):
            continue  # guarded: do not clobber an existing backup
        os.rename(p, dst)
        moves.append((p, dst))
    return moves


def restore(path: str, stamp: str) -> bool:
    """Idempotent restore: move `<name>_old_<stamp>` back to `<name>` if the current name is
    free. Returns True if a restore happened. A double-restore is a no-op."""
    src = aside_path(path, stamp)
    if not os.path.exists(src) or os.path.exists(path):
        return False
    os.rename(src, path)
    return True
