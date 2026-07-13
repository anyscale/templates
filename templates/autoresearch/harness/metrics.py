"""MEASURE/READ foundation — is the number real?

`BEAT_IT.md`'s first rule of reading a result: clear the noise floor before you believe a
gain. This is the tool that makes "real" computable — a decision metric, a bootstrap CI on it,
and a *paired* bootstrap that answers "did B actually beat A, or is it inside the noise?"
Everything in the climb keys off this: a move whose paired CI straddles zero is not a result.

Deterministic (seeded), stdlib-only. Metrics operate on record lists so they compose with
`erroranalysis` slicing; a record is {"y_true", "score" (ranking) or "y_pred" (classification)}.
"""

from __future__ import annotations

import random


# --- decision metrics (pass any of these, or your own, everywhere a metric_fn is wanted) ---
def average_precision(records) -> float:
    """AP / PR-AUC — the fraud metric; robust to heavy class imbalance where ROC saturates.

    Caveat: ties in `score` are broken by input order (Python's stable sort). If a model emits
    many identical scores, AP becomes order-dependent — pre-sort by a stable secondary key, or
    add small deterministic jitter, before trusting small AP deltas on tie-heavy outputs."""
    ranked = sorted(records, key=lambda r: r["score"], reverse=True)
    P = sum(1 for r in records if r["y_true"])
    if P == 0:
        return 0.0
    tp, ap = 0, 0.0
    for i, r in enumerate(ranked, 1):
        if r["y_true"]:
            tp += 1
            ap += tp / i                      # precision at each true positive
    return ap / P


def accuracy(records) -> float:
    return sum(1 for r in records if r["y_pred"] == r["y_true"]) / max(1, len(records))


# --- the noise floor ------------------------------------------------------------------
def bootstrap_ci(records, metric_fn, n=1000, alpha=0.05, seed=0) -> dict:
    """Point estimate + (1-alpha) bootstrap CI by resampling records with replacement."""
    if not records:
        raise ValueError("bootstrap_ci needs a non-empty record list (an empty slice?)")
    rng = random.Random(seed)
    N = len(records)
    stats = []
    for _ in range(n):
        sample = [records[rng.randrange(N)] for _ in range(N)]
        stats.append(metric_fn(sample))
    stats.sort()
    lo = stats[max(0, int(alpha / 2 * n))]
    hi = stats[min(n - 1, int((1 - alpha / 2) * n))]
    return {"value": round(metric_fn(records), 4), "lo": round(lo, 4), "hi": round(hi, 4), "n": N}


def paired_bootstrap(a, b, metric_fn, n=1000, alpha=0.05, seed=0) -> dict:
    """Did model B beat model A? `a` and `b` are aligned per-example (same examples, two
    scorings). Resample the *same* indices for both each draw (paired), so shared eval noise
    cancels. Returns the delta, its CI, and `prob_b_better` (fraction of resamples where B
    wins). Decision rule: a real win needs the CI to exclude 0 (equivalently prob_b_better
    near 1). A delta whose CI straddles 0 is inside the noise — do not bank it."""
    if len(a) != len(b):
        raise ValueError("paired_bootstrap needs aligned per-example records")
    if not a:
        raise ValueError("paired_bootstrap needs non-empty aligned record lists")
    rng = random.Random(seed)
    N = len(a)
    deltas = []
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        deltas.append(metric_fn([b[i] for i in idx]) - metric_fn([a[i] for i in idx]))
    deltas.sort()
    return {
        "delta": round(metric_fn(b) - metric_fn(a), 4),
        "lo": round(deltas[max(0, int(alpha / 2 * n))], 4),
        "hi": round(deltas[min(n - 1, int((1 - alpha / 2) * n))], 4),
        "prob_b_better": round(sum(1 for d in deltas if d > 0) / n, 3),
    }


def is_real_gain(paired: dict, higher_better=True) -> bool:
    """A real GAIN is directional: the whole paired CI must sit on the winning side of zero.
    For higher-better metrics that means lo > 0; for lower-better (WER/loss), hi < 0. A CI that
    straddles zero is noise; a CI entirely on the *losing* side is a reliable regression, NOT a
    gain (an earlier version wrongly counted those as wins — caught by the multiple-comparisons
    simulation)."""
    return paired["lo"] > 0 if higher_better else paired["hi"] < 0


def p_value_gain(a, b, metric_fn, n=1000, seed=0, higher_better=True) -> float:
    """One-sided bootstrap p-value for 'B beats A'. Null = B is not better. p = fraction of
    paired resamples where B fails to beat A. Feeds family-wise / FDR control across a campaign
    (see `significance.py`) so a pile of experiments doesn't manufacture a fake win."""
    if len(a) != len(b) or not a:
        raise ValueError("p_value_gain needs non-empty aligned record lists")
    rng = random.Random(seed)
    N = len(a)
    worse = 0
    for _ in range(n):
        idx = [rng.randrange(N) for _ in range(N)]
        delta = metric_fn([b[i] for i in idx]) - metric_fn([a[i] for i in idx])
        gain = delta if higher_better else -delta
        if gain <= 0:
            worse += 1
    return round(worse / n, 4)


def min_detectable_effect(records, metric_fn, power=0.8, alpha=0.05, n=500, seed=0) -> dict:
    """Power up front (`BEAT_IT.md` anti-fooling rule #6): given THIS eval, what's the smallest
    gain you could reliably detect? Estimates the metric's standard error by bootstrap, then
    MDE ≈ (z_alpha + z_power)·SE·√2 (paired, two-sided-ish). If the gains you're chasing are
    below the MDE, enlarge the eval before spending GPU to confirm you can't tell. `detectable`
    is a helper: is a target gain within reach?"""
    se = (bootstrap_ci(records, metric_fn, n=n, seed=seed)["hi"]
          - bootstrap_ci(records, metric_fn, n=n, seed=seed)["lo"]) / (2 * 1.96)
    z_alpha = 1.96 if alpha <= 0.05 else 1.64
    z_power = {0.8: 0.84, 0.9: 1.28, 0.95: 1.64}.get(power, 0.84)
    mde = (z_alpha + z_power) * se * (2 ** 0.5)
    return {"se": round(se, 4), "mde": round(mde, 4), "power": power, "alpha": alpha,
            "note": f"can reliably detect gains >= ~{mde:.3f}; smaller ones need a bigger eval"}


def detectable(mde: dict, target_gain: float) -> bool:
    return target_gain >= mde["mde"]


def positives_warning(n_pos: int, floor=50) -> str | None:
    """Few positives → point estimates are fragile and CIs wide (the fraud campaign's 112
    frauds). Returns a warning string when under the floor, else None."""
    if n_pos < floor:
        return (f"only {n_pos} positives — CIs will be wide and point estimates fragile; "
                f"CI-mandatory, and consider enlarging the eval before chasing small gains")
    return None
