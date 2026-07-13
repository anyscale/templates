"""R7 (part) — content-hash the pinned eval so every registry row names what it scored on.

A device/library flip once moved a headline 0.05 AP and re-ranked conditions
(`AUTORESEARCH.md` §9.2). The defense: freeze the eval artifact (rows/prompts/episodes +
seeds + decoding/sampling params + metric implementation id), content-hash it, and store the
hash as `eval_pin` on every registry row. Rows are only comparable within one pin; the
registry's `check_eval_pin_consistency` makes a cross-pin comparison detectable — this module
produces the pin it keys on.

The hash is over a *canonical* JSON encoding, so key order and whitespace don't change it —
only the eval's actual content does.
"""

from __future__ import annotations

import hashlib
import json


def eval_pin(spec: dict) -> str:
    """Deterministic content hash of an eval spec. `spec` should include everything that can
    change a number: the row/prompt/episode ids or their hash, seeds, decoding/sampling params,
    and a metric-implementation identifier (e.g. `pytrec_eval==0.5`). Order-independent."""
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def pin_file(path: str, chunk: int = 1 << 20) -> str:
    """Content hash of a frozen eval file on disk (a parquet of eval rows, a prompt set)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()[:16]
