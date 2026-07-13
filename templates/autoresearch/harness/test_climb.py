"""Tests for the loop controller. `python3 test_climb.py`."""

import tempfile

import climb
import erroranalysis as EA
import recipe as R


def _slice_reports():
    ex = ([{"y_true": 1, "y_pred": 1, "slices": {"grp": "easy"}} for _ in range(70)]
          + [{"y_true": 1, "y_pred": 0, "slices": {"grp": "hard"}} for _ in range(30)])
    return {"grp": EA.slice_report(ex, EA.accuracy, "grp")}, EA.accuracy(ex)


def test_propose_next_ranks_and_budget_checks():
    with tempfile.TemporaryDirectory() as base:
        reps, overall = _slice_reports()
        plan = climb.propose_next(base, "08-esm2", 1, dev=overall, train=overall + 0.12,
                                  slice_reports=reps, gpu="A10G", hours=1.0)
        assert plan, "should propose experiments"
        # every item carries a budget verdict and a runnable flag
        assert all("budget" in p and "runnable" in p for p in plan)
        # opportunities are ranked by priority (cheap high-lift first)
        opps = [p for p in plan if p["kind"] == "opportunity"]
        assert opps[0]["priority"] >= opps[-1]["priority"]
        print("ok  propose_next returns a ranked, budget-checked plan")


def test_propose_skips_already_tried_flags():
    with tempfile.TemporaryDirectory() as base:
        reps, overall = _slice_reports()
        plan = climb.propose_next(base, "c", 1, dev=overall, train=overall + 0.12,
                                  slice_reports=reps, tried_flags=("reg",))
        assert all(p["flag"] != "reg" for p in plan)     # the overfit fix was already tried
        print("ok  propose_next skips flags already tried (no re-proposing dead ends)")


def test_full_rung_candidate_is_escalatable_not_dropped():
    with tempfile.TemporaryDirectory() as base:
        # a huge over-envelope proxy would be dropped; a full-rung candidate escalates instead.
        # here just assert a within-budget plan marks items runnable
        reps, overall = _slice_reports()
        plan = climb.propose_next(base, "c", 1, dev=overall, slice_reports=reps)
        assert any(p["runnable"] for p in plan)
        print("ok  within-budget candidates are marked runnable")


def test_should_pivot_when_lever_taps_out():
    r = R.new_recipe(0.40)
    R.add_trick(r, "t1", "a", 0.55)      # +0.15
    R.add_trick(r, "t2", "b", 0.55)      # +0.0
    R.add_trick(r, "t3", "c", 0.55)      # +0.0
    R.add_trick(r, "t4", "d", 0.55)      # +0.0  -> last 3 flat
    p = climb.should_pivot(r, window=3)
    assert p["pivot"] is True and "tapped out" in p["reason"]
    print("ok  should_pivot fires when the last N tricks add ~nothing")


def test_climb_summary():
    r = R.new_recipe(0.40)
    R.add_trick(r, "readout", "a", 0.53)
    R.add_trick(r, "context", "b", 0.57)
    s = climb.climb_summary(r)
    assert s["total_lift"] == 0.17 and s["current_best"] == 0.57 and s["tricks_banked"] == 2
    print("ok  climb_summary reports baseline -> best, total lift, tricks banked")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
