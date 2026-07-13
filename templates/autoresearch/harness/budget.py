"""R4 — submit-time cost caps, enforced (not promised).

BUDGET_POLICY.md states rung caps and wave envelopes; before this module they were prose a
human was trusted to honor. This makes them a gate: estimate a run's cost *before* submit,
refuse it if it exceeds its rung cap or the campaign's remaining envelope, and escalate the
cases the PI must sign (crossing into `full`, exceeding an envelope). It reads real spend from
the registry (R1), so the envelope math is grounded in what actually ran, not a guess.

The estimate is the "prepaid card, not a pinky promise" half at submit time;
`wall_clock_timeout_s()` is the runtime hard-kill half — together they bound a run's cost from
both ends (an estimate can be wrong; a timeout can't overrun).

Everything is denominated in **A10G-equivalent hours** (see `registry.tier_weight`), the same
currency the wave thresholds use — so a cap can't be gamed by choosing a bigger GPU.
"""

from __future__ import annotations

import registry

# A10G-equivalent-hour caps (BUDGET_POLICY.md). `full` is envelope-bound + PI-gated.
RUNG_CAP = {"smoke": 0.5, "proxy": 10.0}
WAVE_ENVELOPE = {1: 60.0, 2: 400.0, 3: 2000.0}

# $/A10G-hr (BUDGET_POLICY conversion table). Spot is the default for research fleets.
USD_PER_A10G_HR = {"spot": 0.35, "on_demand": 1.01}


def to_usd(a10g_equiv_hours: float, spot=True) -> float:
    return round(a10g_equiv_hours * USD_PER_A10G_HR["spot" if spot else "on_demand"], 2)


def to_a10g_hours(usd: float, spot=True) -> float:
    """A dollar envelope, in the A10G-equivalent hours the caps are denominated in. NOTE: spot
    prices drift ±, so this is an estimate — pair it with the per-run wall-clock timeout and the
    registry's *actual* recorded cost, which is what real enforcement reads."""
    return round(usd / USD_PER_A10G_HR["spot" if spot else "on_demand"], 2)


def estimate(gpu_type: str, num_gpus: float, hours: float) -> dict:
    """A cost estimate for one run, in raw and A10G-equivalent GPU-hours."""
    raw = round(num_gpus * hours, 4)
    return {
        "gpu_type": gpu_type,
        "gpu_hours": raw,
        "a10g_equiv_hours": round(raw * registry.tier_weight(gpu_type), 4),
    }


def preflight(base: str, campaign: str, wave: int, rung: str, est: dict,
              envelope_ah: float | None = None) -> dict:
    """Decide whether a run may be submitted. Returns a verdict dict:
    `{allowed, escalate, reason, est_eq, spent, remaining, envelope}`.

    `envelope_ah` overrides the wave envelope — pass `to_a10g_hours($100)` to enforce a dollar
    budget directly. Enforcement is real *only if every run goes through here and reports its
    actual cost to the registry* (R1); the live `anyscale job submit` wiring is the last stubbed
    piece, so today this refuses over-budget runs by discipline + the harness, not by physics.

    - smoke : allowed iff ≤ the smoke cap (agent runs these freely).
    - proxy : allowed iff ≤ the per-idea cap AND fits the remaining envelope.
    - full  : always escalates (crossing into full is the PI boundary,
              `BUDGET_POLICY.md`) — never auto-allowed, and still refused outright if it
              can't fit the envelope even with approval.
    Any rung that would push cumulative spend past the wave envelope is refused/escalated.
    """
    if wave not in WAVE_ENVELOPE:
        raise ValueError(f"unknown wave {wave!r}")
    eq = est["a10g_equiv_hours"]
    spent = registry.campaign_spend(base, campaign)["a10g_equiv_hours"]
    envelope = envelope_ah if envelope_ah is not None else WAVE_ENVELOPE[wave]
    remaining = round(envelope - spent, 4)
    v = {"est_eq": eq, "spent": spent, "remaining": remaining, "envelope": envelope,
         "allowed": False, "escalate": False, "reason": ""}

    if rung == "smoke":
        if eq <= RUNG_CAP["smoke"]:
            v["allowed"] = True; v["reason"] = "within smoke cap; agent may run freely"
        else:
            v["reason"] = f"smoke est {eq} A10G-eq > cap {RUNG_CAP['smoke']} — re-scope to tiny data"
    elif rung == "proxy":
        if eq > RUNG_CAP["proxy"]:
            v["reason"] = f"proxy est {eq} > per-idea cap {RUNG_CAP['proxy']} A10G-eq"
        elif eq > remaining:
            v["reason"] = f"proxy est {eq} > remaining envelope {remaining} A10G-eq — refuse"
        else:
            v["allowed"] = True; v["reason"] = "within proxy cap and envelope"
    elif rung == "full":
        v["escalate"] = True
        if eq > remaining:
            v["reason"] = (f"full run {eq} A10G-eq exceeds remaining envelope {remaining} — "
                           f"needs a bigger envelope AND PI sign-off")
        else:
            v["reason"] = "full rung requires PI sign-off (money + is-this-real boundary)"
    else:
        raise ValueError(f"unknown rung {rung!r} (smoke|proxy|full)")
    return v


def wall_clock_timeout_s(hours: float, margin: float = 1.5) -> int:
    """Runtime hard-kill derived from the estimate — the second half of a real cap. A run
    that overruns its estimate by more than `margin` is killed rather than billed. Baked into
    the R2-generated job spec so a bad estimate can't become an unbounded bill."""
    return int(hours * 3600 * margin)
