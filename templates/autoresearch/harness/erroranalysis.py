"""The DIAGNOSE step, as code — error analysis that points at the next test.

`BEAT_IT.md` says: don't guess what to improve, look at the errors and let them point. This
turns a scored eval into (1) per-slice metrics that expose *where* the ceiling is held down,
and (2) ranked **hypothesis cards** — symptom → cheapest test → direction — straight from the
diagnosis table. The engine proposes; the human/PI dispositions.

Metric-agnostic: pass any `metric(examples) -> float` (accuracy, AP, nDCG, -WER, …). Set
`higher_better=False` for loss/error/WER and everything flips consistently.

An example is a dict: {"y_true", "y_pred" or "score", "slices": {feature: value}, ...}.
"""

from __future__ import annotations

from collections import defaultdict

RUNG_COST = {"smoke": 0.5, "proxy": 5.0, "full": 50.0}  # A10G-eq hr, for priority = lift/cost


# --- a couple of convenience metrics (callers usually pass their own) ------------------
def accuracy(examples) -> float:
    return sum(1 for e in examples if e["y_pred"] == e["y_true"]) / max(1, len(examples))


def error_rate(examples) -> float:
    return 1.0 - accuracy(examples)


# --- slicing ---------------------------------------------------------------------------
def slice_report(examples, metric, by: str) -> dict:
    """Per-value metric for one slice feature: {value: {metric, n, n_pos}}.

    Use this for **pointwise** metrics (accuracy, per-example error) where a subgroup's score
    is meaningful in isolation. Do NOT use it for **ranking** metrics (AP, nDCG): a slice
    defined by a label-correlated feature is often single-class, and a ranking metric on a
    single-class slice is degenerate (all-positives → 1.0, all-negatives → 0.0). For ranking
    tasks use `recall_by_subgroup`, which scores subgroups against the *global* ranking."""
    groups = defaultdict(list)
    for e in examples:
        groups[e.get("slices", {}).get(by)].append(e)
    return {val: {"metric": metric(g), "n": len(g),
                  "n_pos": sum(1 for e in g if e.get("y_true"))}
            for val, g in groups.items()}


def recall_by_subgroup(examples, by: str, k: int | None = None) -> dict:
    """Ranking-aware error analysis: rank ALL examples by score, take the global top-`k`
    (default k = number of positives), then report, per positive subgroup, the fraction of its
    positives that made the global cut. Same shape as `slice_report` ({value: {metric, n,
    n_pos}}) so it drops straight into `worst_slices`/`diagnose` — but it's valid for ranking
    metrics because the cutoff is global, not within-slice. Low recall on a subgroup =
    that subgroup's positives are ranked too low (e.g. burst-pattern frauds the model misses)."""
    order = sorted(range(len(examples)), key=lambda i: examples[i]["score"], reverse=True)
    P = sum(1 for e in examples if e.get("y_true"))
    k = k if k is not None else P
    caught = set(order[:k])
    groups = defaultdict(list)
    for i, e in enumerate(examples):
        if e.get("y_true"):
            groups[e.get("slices", {}).get(by)].append(i in caught)
    return {val: {"metric": (sum(flags) / len(flags)) if flags else 0.0,
                  "n": len(flags), "n_pos": len(flags)}
            for val, flags in groups.items()}


def worst_slices(report: dict, higher_better=True, min_n=1, min_pos=1, k=3) -> list:
    """The k worst-scoring slices with at least `min_n` examples AND `min_pos` positives.

    `min_pos` is load-bearing: a ranking metric (AP, nDCG) on a slice with no positives is
    *undefined*, not "bad" — without this guard the negative-class slice looks like a
    catastrophic failure and the controller forever proposes "fix the legit class." For a
    balanced accuracy task where negative-only slices are meaningful, pass `min_pos=0`."""
    items = [(v, info) for v, info in report.items()
             if info["n"] >= min_n and info["n_pos"] >= min_pos]
    items.sort(key=lambda kv: kv[1]["metric"], reverse=not higher_better)
    return items[:k]


# --- hypothesis cards ------------------------------------------------------------------
def discover_slices(examples, features=None, k=5, min_pos=10, max_order=2) -> list:
    """Auto-find the underperforming subgroups WITHOUT being told which feature to look at
    (SliceLine / DivExplorer in spirit). Enumerate predicates over the slice features — single
    values and up to `max_order`-way conjunctions — score each positive subgroup by global-cutoff
    recall, and return the worst coherent, well-supported ones ranked by
    `deficit × support-fraction` (the SliceLine tradeoff: big *and* bad matters most).

    Ranking-aware by construction: recall is measured against the GLOBAL top-k cut (k = number
    of positives), so it's valid where per-slice AP is degenerate. Returns
    `[{predicate, recall, n_pos, support_frac, score}]`, worst first.
    """
    from itertools import combinations
    order = sorted(range(len(examples)), key=lambda i: examples[i]["score"], reverse=True)
    total_pos = sum(1 for e in examples if e.get("y_true"))
    if total_pos == 0:
        return []
    caught = set(order[:total_pos])
    global_recall = sum(1 for i in order[:total_pos] if examples[i].get("y_true")) / total_pos
    feats = features or sorted({f for e in examples for f in e.get("slices", {})})

    # candidate predicates: single (feat=val) and conjunctions up to max_order
    singles = sorted({(f, e["slices"][f]) for e in examples for f in feats if f in e.get("slices", {})})
    preds = []
    for o in range(1, max_order + 1):
        preds += [frozenset(c) for c in combinations(singles, o)
                  if len({f for f, _ in c}) == o]        # don't AND two values of the same feature

    scored = []
    for pred in preds:
        members = [i for i, e in enumerate(examples)
                   if all(e.get("slices", {}).get(f) == v for f, v in pred) and e.get("y_true")]
        if len(members) < min_pos:
            continue
        recall = sum(1 for i in members if i in caught) / len(members)
        deficit = global_recall - recall
        if deficit <= 0:
            continue
        support_frac = len(members) / total_pos
        scored.append({"predicate": dict(pred), "recall": round(recall, 4),
                       "n_pos": len(members), "support_frac": round(support_frac, 3),
                       "score": round(deficit * support_frac, 4)})
    scored.sort(key=lambda s: s["score"], reverse=True)
    # drop a conjunction that's no better than a single it contains (prefer the simpler slice)
    return scored[:k]


def _card(symptom, cause, test, rung, lift, risk, flag, kind="opportunity") -> dict:
    return {"symptom": symptom, "cause": cause, "cheapest_test": test, "rung": rung,
            "expected_lift": round(lift, 4), "confound_risk": risk, "flag": flag,
            "kind": kind, "priority": round(lift / RUNG_COST[rung], 4)}


def diagnose(dev, baseline=None, train=None, slice_reports=None, noise=None,
             higher_better=True, overfit_gap=0.05, tie_margin=0.01,
             min_slice_n=30, min_slice_pos=1, total_n=None) -> list:
    """Emit ranked hypothesis cards from a result. Opportunities (ranked by lift-per-cost)
    come first; warnings (things that make a claimed win suspect) come after.

    - `dev`/`baseline`/`train`: the decision metric on dev, the frozen baseline, and train.
    - `noise`: the eval's noise band (bootstrap/seed std) — used to flag non-real gains.
    - `slice_reports`: {feature: slice_report(...)} to surface slice-specific failures.
    """
    s = 1 if higher_better else -1          # normalize so "bigger is better" everywhere
    d = s * dev
    b = s * baseline if baseline is not None else None
    t = s * train if train is not None else None
    cards = []

    # WARNING: a claimed gain over baseline sits inside the noise band → not real
    if b is not None and noise is not None and (d - b) <= noise:
        cards.append(_card(
            f"move over baseline ({dev - baseline:+.3f}) is within the noise band (±{noise})",
            "not a real gain — inside seed/bootstrap variance",
            "add seeds + paired bootstrap; add positives if the count is small",
            "proxy", 0.0, "n/a", "seeds", kind="warning"))

    # WARNING: a cheap/simple baseline is ~tied with the model → not earning its keep
    if b is not None and abs(d - b) <= tie_margin:
        cards.append(_card(
            f"the baseline ({baseline:.3f}) is ~tied with dev ({dev:.3f})",
            "model isn't earning its keep — or there's leakage, or the task is easy",
            "run the strongest cheap baseline to convergence; fire the leak-audit",
            "proxy", 0.0, "med", "baseline", kind="warning"))

    # OPPORTUNITY: train >> dev → overfitting
    if t is not None and (t - d) >= overfit_gap:
        cards.append(_card(
            f"train exceeds dev by {t - d:.3f} → overfitting",
            "overfitting: too little regularization / data for the capacity",
            "dropout/weight-decay sweep, or 2x data/augmentation, at proxy",
            "proxy", 0.5 * (t - d), "low", "reg"))

    # OPPORTUNITY: the worst subgroup trails the BEST subgroup (compared within one metric
    # space — not against `dev`, which may be a different metric than the slice report's).
    if slice_reports:
        tot = total_n or _infer_total(slice_reports)
        for feat, report in slice_reports.items():
            quals = [(v, info) for v, info in report.items()
                     if info["n"] >= min_slice_n and info["n_pos"] >= min_slice_pos]
            if len(quals) < 2:          # need at least two comparable subgroups to localize a gap
                continue
            quals.sort(key=lambda kv: kv[1]["metric"], reverse=not higher_better)  # worst first
            (wv, wi), (bv, bi) = quals[0], quals[-1]
            deficit = (bi["metric"] - wi["metric"]) if higher_better else (wi["metric"] - bi["metric"])
            if deficit > tie_margin and tot:
                frac = wi["n"] / tot
                cards.append(_card(
                    f"slice {feat}={wv!r} trails the best {feat} slice ({bv!r}) by "
                    f"{deficit:.3f} (n={wi['n']}, {frac:.0%} of eval)",
                    "slice-specific failure holding the ceiling down",
                    f"eval-only re-score of {feat}={wv!r} with one targeted tweak "
                    f"(feature / loss-weight / data)",
                    "smoke", deficit * frac, "low", f"slice_{feat}"))

    cards.sort(key=lambda c: (c["kind"] == "warning", -c["priority"]))
    return cards


def _infer_total(slice_reports) -> int:
    for report in slice_reports.values():          # any one feature partitions all examples
        return sum(info["n"] for info in report.values())
    return 0
