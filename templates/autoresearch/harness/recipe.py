"""DECIDE/BANK — the compounding engine (the whole point of hill-climbing).

Incremental improvement only *compounds* if wins stack and don't silently rot. A recipe is the
ordered stack of tricks that are ON, each with the marginal lift it added *on top of the
current best* (not the bare baseline — interactions are real). This module banks wins, computes
the marginal-lift waterfall, guards against stale tricks and regressions, and tracks the
dev-vs-holdout gap so you can see when you've started overfitting the eval you climb on.

Pure functions over a plain dict, so a recipe serializes straight to the registry / disk.
"""

from __future__ import annotations


def new_recipe(baseline: float, higher_better=True) -> dict:
    return {"baseline": baseline, "higher_better": higher_better, "tricks": []}


def _improves(recipe, new, old) -> bool:
    return new > old if recipe["higher_better"] else new < old


def best_score(recipe: dict) -> float:
    """The best score actually achieved so far — baseline advanced only by *improving* tricks.
    A dud (non-improving) trick must NOT lower the running best, or every later marginal would
    be measured against the wrong reference and compounding would silently break."""
    best = recipe["baseline"]
    for t in recipe["tricks"]:
        if _improves(recipe, t["score"], best):
            best = t["score"]
    return best


def add_trick(recipe: dict, name: str, flag: str, score: float) -> dict:
    """Bank a trick measured ON TOP of the current best. Returns
    `{name, flag, score, marginal, banked}`; `banked` is False when the trick didn't actually
    improve the stack (marginal ≤ 0) — a signal not to keep it on."""
    prev = best_score(recipe)
    marginal = round((score - prev) if recipe["higher_better"] else (prev - score), 6)
    entry = {"name": name, "flag": flag, "score": score, "marginal": marginal,
             "banked": marginal > 0}
    recipe["tricks"].append(entry)
    return entry


def waterfall(recipe: dict) -> list:
    """The marginal-lift waterfall: baseline then each trick's contribution — the view that
    makes a stale or negative trick jump out."""
    rows = [{"label": "baseline", "value": recipe["baseline"], "marginal": None}]
    for t in recipe["tricks"]:
        rows.append({"label": f"+{t['name']}", "value": t["score"], "marginal": t["marginal"]})
    return rows


def stale_tricks(recipe: dict, off_scores: dict, eps=1e-9) -> list:
    """Regression guard. `off_scores[name]` = the stack's score with that trick turned OFF.
    A trick is stale if removing it costs ~nothing now (its marginal has been absorbed by later
    tricks / interactions). Returns the names that no longer earn their slot."""
    best = best_score(recipe)
    stale = []
    for t in recipe["tricks"]:
        if t["name"] not in off_scores:
            continue
        cost_of_removing = (best - off_scores[t["name"]]) if recipe["higher_better"] \
            else (off_scores[t["name"]] - best)
        if cost_of_removing <= eps:
            stale.append(t["name"])
    return stale


def dev_holdout_gap(dev: float, holdout: float, higher_better=True) -> dict:
    """Two-eval discipline (`BEAT_IT.md` READ §2). A widening gap = you're overfitting the dev
    eval you climb on. `overfitting_dev` flags when dev is meaningfully ahead of holdout."""
    gap = (dev - holdout) if higher_better else (holdout - dev)
    return {"dev": dev, "holdout": holdout, "gap": round(gap, 4),
            "overfitting_dev": gap > 0.02}


def holdout_budget(queries_spent: int, cap: int = 10) -> dict:
    """The holdout is base camp — touch it rarely. Every score against it spends a query; the
    more you peek the less it means (reusable-holdout idea). Track the budget explicitly."""
    return {"spent": queries_spent, "cap": cap, "remaining": max(0, cap - queries_spent),
            "exhausted": queries_spent >= cap}
