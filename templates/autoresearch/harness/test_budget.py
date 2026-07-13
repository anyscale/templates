"""Tests for R4 budget enforcement. Self-runnable: `python3 test_budget.py`."""

import tempfile

import budget
import registry as R


def _seed_spend(base, campaign, eq_hours, gpu="A10G"):
    """Put a run on the registry so campaign_spend reflects it. eq = raw*tier; back out raw."""
    raw = eq_hours / R.tier_weight(gpu)
    R.append_run(base, {"campaign": campaign, "run_id": "seed_" + str(eq_hours), "commit": "c",
        "rung": "proxy", "cost": {"gpu_hours": raw, "gpu_type": gpu, "usd_est": 0, "spot": True},
        "eval_pin": "sha256:p", "seed_plan_commit": "s", "status": "SUCCEEDED"})


def test_estimate_tier_weighted():
    e = budget.estimate("H100", 8, 15)          # campaign 02's 7B run: 8 GPUs * 15h = 120 raw
    assert e["gpu_hours"] == 120
    assert e["a10g_equiv_hours"] == 660.0        # 120 * 5.5 (H100)
    a100 = budget.estimate("A100", 8, 15)        # same shape on A100 is cheaper
    assert a100["a10g_equiv_hours"] == 420.0     # 120 * 3.5
    print("ok  estimate is tier-weighted (8×H100×15h = 660, 8×A100×15h = 420 A10G-eq)")


def test_smoke_cap():
    with tempfile.TemporaryDirectory() as base:
        ok = budget.preflight(base, "c", 1, "smoke", budget.estimate("A10G", 1, 0.3))
        assert ok["allowed"] and not ok["escalate"]
        bad = budget.preflight(base, "c", 1, "smoke", budget.estimate("A10G", 1, 2))
        assert not bad["allowed"], bad
    print("ok  smoke allowed under 0.5 cap, refused over")


def test_proxy_within_and_over_envelope():
    with tempfile.TemporaryDirectory() as base:
        # fresh wave-1 campaign: 3 A10G-eq proxy fits
        v = budget.preflight(base, "c", 1, "proxy", budget.estimate("A10G", 1, 3))
        assert v["allowed"], v
        # now spend 58 of the 60 envelope; a 5-hr proxy no longer fits
        _seed_spend(base, "c", 58)
        v2 = budget.preflight(base, "c", 1, "proxy", budget.estimate("A10G", 1, 5))
        assert not v2["allowed"] and "envelope" in v2["reason"], v2
    print("ok  proxy fits fresh envelope, refused once envelope nearly spent")


def test_proxy_over_per_idea_cap():
    with tempfile.TemporaryDirectory() as base:
        v = budget.preflight(base, "c", 2, "proxy", budget.estimate("A100", 1, 4))  # 14 A10G-eq
        assert not v["allowed"] and "per-idea cap" in v["reason"], v
    print("ok  proxy over the 10 A10G-eq per-idea cap is refused")


def test_full_always_escalates():
    with tempfile.TemporaryDirectory() as base:
        v = budget.preflight(base, "02-rl", 3, "full", budget.estimate("H100", 8, 15))  # 660
        assert v["escalate"] and not v["allowed"], v      # PI must sign
        assert v["est_eq"] == 660.0
        # same run mislabelled Wave 2 would blow the envelope outright
        v2 = budget.preflight(base, "02-rl", 2, "full", budget.estimate("H100", 8, 15))
        assert "exceeds remaining envelope" in v2["reason"], v2
    print("ok  full always escalates; the 660 run refused outright under a Wave-2 envelope")


def test_wall_clock_timeout():
    assert budget.wall_clock_timeout_s(2, 1.5) == 10800   # 2h * 1.5 = 3h
    print("ok  wall-clock timeout = hours * margin (runtime hard-kill)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
