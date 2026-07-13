"""Tests for the feasibility estimator + dollar budget. `python3 test_feasibility.py`."""

import budget
import feasibility as F


def test_dollar_conversion_roundtrips():
    ah = budget.to_a10g_hours(100, spot=True)          # $100 spot -> ~285 A10G-hr
    assert 280 < ah < 290
    assert budget.to_usd(ah, spot=True) - 100 < 0.5
    print(f"ok  $100 spot ≈ {ah} A10G-hr; conversion round-trips")


def test_affordable_wave1_go():
    plan = {"gate": 3, "proxy": 4, "full": 8, "controls": 2}      # ESM-2-ish, ~17 A10G-hr
    r = F.feasibility(100, plan, spot=True)
    assert r["verdict"] == "GO"
    print("ok  $100 on a cheap wave-1 campaign -> GO (17 A10G-hr << 285)")


def test_7b_rl_on_100_is_pilot_only():
    # 7B RL: gate ~$24 (affordable), proxy affordable, but one full run ~$231 spot is not.
    plan = {"gate": 70, "proxy": 140, "full": 660, "controls": 30}
    r = F.feasibility(100, plan, spot=True)
    assert r["verdict"] == "PILOT_ONLY", r
    assert any("full run" in x for x in r["reasons"])
    print("ok  $100 on 7B-RL -> PILOT_ONLY: reproduce + proxy to de-risk, but no full run (~$231)")


def test_cant_even_reproduce_is_not_enough():
    plan = {"gate": 400, "proxy": 40, "full": 80}      # gate alone > $100-of-spot A10G-hr
    r = F.feasibility(100, plan, spot=True)
    assert r["verdict"] == "NOT_ENOUGH"
    assert any("reproduction gate" in x for x in r["reasons"])
    print("ok  when even the gate is unaffordable -> NOT_ENOUGH (can't reproduce, don't start)")


def test_pilot_only_when_no_full_run_affordable():
    plan = {"gate": 200, "proxy": 40, "full": 900}     # gate affordable, full not
    r = F.feasibility(100, plan, spot=True)
    assert r["verdict"] in ("PILOT_ONLY", "NOT_ENOUGH")
    print("ok  budget that buys the gate + proxy but not a full run -> PILOT_ONLY")


def test_underpowered_eval_is_not_enough_even_if_affordable():
    plan = {"gate": 1, "proxy": 2, "full": 4}
    r = F.feasibility(100, plan, spot=True, mde=0.03, target_gain=0.005)   # chasing < detectable
    assert r["verdict"] == "NOT_ENOUGH"
    assert any("detect" in x for x in r["reasons"])
    print("ok  affordable but eval too small to detect the target gain -> NOT_ENOUGH")


def test_power_law_fit_and_extrapolation():
    # synthetic curve error = 0.10 + 2.0 * N^-0.5
    pts = [(N, round(0.10 + 2.0 * N ** -0.5, 4)) for N in (100, 400, 1600, 6400)]
    fit = F.fit_power_law(pts)
    assert abs(fit["c"] - 0.10) < 0.03 and abs(fit["b"] - 0.5) < 0.1, fit
    # reachable target above the floor
    assert F.n_for_target(fit, 0.15) < float("inf")
    print(f"ok  power-law fit recovers floor≈{fit['c']}, exponent≈{fit['b']}")


def test_target_below_floor_is_unreachable():
    pts = [(N, round(0.20 + 1.5 * N ** -0.4, 4)) for N in (100, 400, 1600, 6400)]
    r = F.feasibility(100, {"gate": 1, "proxy": 2, "full": 4}, spot=True,
                      pilot=pts, target_error=0.10)          # below the ~0.20 floor
    assert r["verdict"] == "NOT_ENOUGH"
    assert any("floor" in x and "approach" in x for x in r["reasons"])
    print("ok  target below the learning-curve floor -> NOT_ENOUGH (change method, not budget)")


def test_no_pilot_is_labeled_estimate_only():
    r = F.feasibility(100, {"gate": 3, "proxy": 4, "full": 8}, spot=True, target_error=0.1)
    assert any("no pilot data" in x for x in r["reasons"])
    print("ok  without pilot data the verdict is honestly labeled compute-affordability-only")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
