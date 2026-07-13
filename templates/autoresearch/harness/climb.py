"""The controller — drives the loop end to end (up to the submit boundary).

This is the brain that ties the harness into an actual hill-climb: given where a campaign is
(its latest scored result + error-analysis slices + the recipe so far), it DIAGNOSEs, turns the
diagnosis into candidate experiments, drops already-tried ones, budget-checks each, and returns
a **ranked, runnable plan** — cheapest-high-lift first. It stops at `launcher.submit` (the PI /
irreversibility boundary); a human or a monitored auto-loop picks from the plan and launches.

That's the whole MEASURE→DIAGNOSE→HYPOTHESIZE→TEST loop as one call, working on ANY job that
speaks the `contract` (per-example eval output → metric → slices). `should_pivot` reads the
recipe to say when a lever is tapped out and it's time to change tack.
"""

from __future__ import annotations

import budget
import erroranalysis
import recipe as recipe_mod


def propose_next(base, campaign, wave, dev, *, train=None, baseline=None, noise=None,
                 slice_reports=None, gpu="A10G", hours=1.0, tried_flags=(),
                 higher_better=True) -> list:
    """One turn of the loop: diagnose the current result → ranked, budget-checked experiments.

    Each plan item: `{flag, rung, kind, rationale, test, expected_lift, priority, estimate,
    budget, runnable}`. `runnable` is True if the budget allows it (or it's a full run the PI
    can escalate); refused-outright items are dropped. Already-tried flags are skipped so the
    loop doesn't re-propose dead ends.
    """
    cards = erroranalysis.diagnose(
        dev=dev, train=train, baseline=baseline, noise=noise,
        slice_reports=slice_reports, higher_better=higher_better)

    plan = []
    for c in cards:
        if c["flag"] in tried_flags:
            continue
        rung = c["rung"]
        est_hours = min(hours, 0.4) if rung == "smoke" else hours
        est = budget.estimate(gpu, 1, est_hours)
        verdict = budget.preflight(base, campaign, wave, rung, est)
        runnable = verdict["allowed"] or verdict["escalate"]
        if not runnable and c["kind"] != "warning":
            continue                       # refused outright — don't put it on the plan
        plan.append({
            "flag": c["flag"], "rung": rung, "kind": c["kind"],
            "rationale": c["symptom"], "test": c["cheapest_test"],
            "expected_lift": c["expected_lift"], "priority": c["priority"],
            "estimate": est, "budget": verdict, "runnable": runnable,
        })
    return plan


def should_pivot(recipe: dict, window=3, eps=1e-6) -> dict:
    """Know when to change levers. If the last `window` banked tricks all added ~nothing, the
    current lever (readout, say) is tapped out — pivot to another (data → arch → objective →
    ensemble) or declare the local max. Returns `{pivot, reason, recent_marginals}`."""
    tricks = recipe["tricks"]
    if len(tricks) < window:
        return {"pivot": False, "reason": f"only {len(tricks)} tricks; keep climbing this lever",
                "recent_marginals": [t["marginal"] for t in tricks]}
    recent = [t["marginal"] for t in tricks[-window:]]
    flat = all(m <= eps for m in recent)
    return {"pivot": flat, "recent_marginals": recent,
            "reason": (f"last {window} tricks added ~0 ({recent}) — lever tapped out, change tack"
                       if flat else "still gaining on this lever, keep going")}


def climb_summary(recipe: dict) -> dict:
    """A one-glance status: where we started, where we are, total lift, per-trick contributions."""
    best = recipe_mod.best_score(recipe)
    total = (best - recipe["baseline"]) if recipe["higher_better"] else (recipe["baseline"] - best)
    banked = [t for t in recipe["tricks"] if t["banked"]]
    return {
        "baseline": recipe["baseline"], "current_best": best,
        "total_lift": round(total, 4), "tricks_banked": len(banked),
        "tricks_tried": len(recipe["tricks"]),
        "waterfall": recipe_mod.waterfall(recipe),
    }
