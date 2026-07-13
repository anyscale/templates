"""Feasibility — is $N even enough to make progress here? (de-risk before scaling)

This answers the second, harder budget question. Big labs don't guess whether an idea is worth
scaling — they de-risk empirically with three moves, all of which this module makes explicit:

  1. COMPUTE affordability — does the budget buy enough GPU-hours for the minimal viable
     campaign (reproduce the gate + one proxy pulse + at least one full run)?
  2. STATISTICAL power — even if affordable, is the eval big enough to *detect* the gain you're
     chasing? (`metrics.min_detectable_effect`.) If the smallest detectable gain is bigger than
     your target, $N buys you a run you can't read.
  3. LEARNING-CURVE extrapolation — run a few cheap pilot points, fit a power law
     (error = c + a·N^-b, the scaling-laws form), and extrapolate: how much data/compute to hit
     the target, and — crucially — is the target even *below the fitted floor* c? If it isn't,
     no budget reaches it and you need a better method, not more money.

Verdict: NOT_ENOUGH  >  PILOT_ONLY  >  GO (with caveats). Honest by construction — the
learning-curve piece needs real pilot data, and it says so.
"""

from __future__ import annotations

import math

import budget


# --- learning curve: error = c + a * N**(-b)  (data/compute scaling law) --------------
def fit_power_law(points) -> dict | None:
    """Fit error = c + a·N^(-b) to pilot points [(N, error), ...] (N = data size or steps).
    Stdlib-only: grid-search the asymptote c, linear-fit log(error-c) vs log(N) for each.
    Returns {a, b, c, rmse} or None if it can't fit (e.g. error not decreasing)."""
    if len(points) < 3:
        return None
    errs = [e for _, e in points]
    emin = min(errs)
    best = None
    for frac in [i / 20 for i in range(20)]:          # c in [0, 0.95*emin)
        c = emin * frac
        xs, ys, ok = [], [], True
        for N, e in points:
            d = e - c
            if d <= 0:
                ok = False
                break
            xs.append(math.log(N))
            ys.append(math.log(d))
        if not ok:
            continue
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        if denom == 0:
            continue
        m = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / denom
        q = my - m * mx
        b, a = -m, math.exp(q)
        rmse = (sum((c + a * N ** (-b) - e) ** 2 for N, e in points) / n) ** 0.5
        if best is None or rmse < best["rmse"]:
            best = {"a": a, "b": round(b, 4), "c": round(c, 5), "rmse": round(rmse, 5)}
    return best


def error_at(fit: dict, N: float) -> float:
    return round(fit["c"] + fit["a"] * N ** (-fit["b"]), 5)


def n_for_target(fit: dict, target_error: float) -> float:
    """Data/compute needed to reach `target_error`. inf if the target is at/below the fitted
    floor c — meaning no amount of data gets there with this method."""
    if target_error <= fit["c"]:
        return math.inf
    return (fit["a"] / (target_error - fit["c"])) ** (1 / fit["b"])


# --- the verdict ----------------------------------------------------------------------
def feasibility(usd: float, plan: dict, spot=True, target_gain=None, mde=None,
                pilot=None, target_error=None) -> dict:
    """`plan` = A10G-equivalent hours per stage: {gate, proxy, full, controls}. Returns a
    verdict with itemized reasons. `mde`+`target_gain` gate statistical power;
    `pilot`+`target_error` gate the learning-curve extrapolation (both optional — without them
    you get the compute-affordability verdict, honestly labeled as such)."""
    envelope_ah = budget.to_a10g_hours(usd, spot)
    gate, proxy, full = plan.get("gate", 0), plan.get("proxy", 0), plan.get("full", 0)
    need_min = gate + proxy + full                     # reproduce + rank + one full run
    need_all = sum(plan.values())
    verdict, reasons = "GO", []

    # 1. compute affordability
    if envelope_ah < gate:
        verdict = "NOT_ENOUGH"
        reasons.append(f"can't even clear the reproduction gate: it needs ~{gate} A10G-hr "
                       f"(≈${budget.to_usd(gate, spot)}), budget buys {envelope_ah} A10G-hr (${usd}).")
    elif envelope_ah < need_min:
        verdict = "PILOT_ONLY"
        reasons.append(f"affords the gate + a proxy pulse but NOT a full run — minimal viable "
                       f"campaign is ~{need_min} A10G-hr (≈${budget.to_usd(need_min, spot)}). "
                       f"Use it to de-risk, not to publish.")
    elif envelope_ah < need_all:
        reasons.append(f"affords a minimal campaign but not the full plan "
                       f"(~{need_all} A10G-hr ≈ ${budget.to_usd(need_all, spot)}); prioritize experiments.")
    else:
        reasons.append(f"comfortably affordable: plan ~{need_all} A10G-hr "
                       f"(≈${budget.to_usd(need_all, spot)}) fits in {envelope_ah} A10G-hr (${usd}).")

    # 2. statistical power
    if mde is not None and target_gain is not None and target_gain < mde:
        verdict = "NOT_ENOUGH"
        reasons.append(f"eval too small to READ the result: smallest detectable gain ~{mde}, but "
                       f"you're chasing {target_gain}. You'd spend the money and not be able to "
                       f"tell if it worked — enlarge the eval / positives first.")

    # 3. learning-curve extrapolation (needs pilot data)
    curve = None
    if pilot and target_error is not None:
        fit = fit_power_law(pilot)
        if not fit or fit["b"] <= 0:
            reasons.append("pilot points show no improving trend — more data likely won't help; "
                           "rethink the method, don't buy more compute.")
            if verdict == "GO":
                verdict = "PILOT_ONLY"
        elif target_error <= fit["c"]:
            verdict = "NOT_ENOUGH"
            reasons.append(f"target error {target_error} is BELOW the fitted floor {fit['c']} — no "
                           f"amount of data/compute reaches it with this method. Change the "
                           f"approach, not the budget.")
        else:
            need_N = n_for_target(fit, target_error)
            curve = {"fit": fit, "N_for_target": round(need_N)}
            reasons.append(f"learning curve (floor {fit['c']}) extrapolates ~{round(need_N):,} "
                           f"examples/steps to hit {target_error}. Convert that to GPU-hours to "
                           f"price it against the budget.")
    elif target_error is not None:
        reasons.append("no pilot data given — run 3 cheap points (N/8, N/4, N/2) first to fit the "
                       "curve; until then this is a compute-affordability estimate only.")

    return {"verdict": verdict, "usd": usd, "spot": spot,
            "envelope_a10g_hr": envelope_ah, "reasons": reasons, "curve": curve}
