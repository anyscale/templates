"""Tests for the compounding engine. `python3 test_recipe.py`."""

import recipe as R


def test_marginal_lift_on_top_of_current_best():
    r = R.new_recipe(baseline=0.40)
    e1 = R.add_trick(r, "readout=mlp", "readout", 0.53)
    e2 = R.add_trick(r, "context=1024", "seq_len", 0.57)
    assert e1["marginal"] == 0.13 and e2["marginal"] == 0.04    # each vs the running best
    assert R.best_score(r) == 0.57
    print("ok  marginal lift is measured on top of the current best, not the baseline")


def test_non_improving_trick_is_not_banked():
    r = R.new_recipe(0.50)
    R.add_trick(r, "good", "a", 0.60)
    dud = R.add_trick(r, "dud", "b", 0.59)                       # worse than current best
    assert dud["marginal"] < 0 and dud["banked"] is False
    print("ok  a trick that doesn't beat the current best is flagged not-banked")


def test_dud_does_not_corrupt_the_running_best():
    """Regression (found by simulation): a dud must not lower best_score, or the NEXT trick's
    marginal is measured against the wrong reference and compounding breaks."""
    r = R.new_recipe(0.50)
    R.add_trick(r, "good", "a", 0.60)
    R.add_trick(r, "dud", "b", 0.59)                             # non-improving
    assert R.best_score(r) == 0.60, "dud must not lower the best"
    nxt = R.add_trick(r, "next", "c", 0.63)
    assert nxt["marginal"] == 0.03, "marginal must be vs the true best (0.60), not the dud"
    print("ok  a dud trick doesn't corrupt best_score or the next marginal (bug fix)")


def test_waterfall_shape():
    r = R.new_recipe(0.40)
    R.add_trick(r, "t1", "a", 0.50)
    w = R.waterfall(r)
    assert w[0]["label"] == "baseline" and w[0]["value"] == 0.40
    assert w[1]["label"] == "+t1" and w[1]["marginal"] == 0.10
    print("ok  waterfall = baseline then each trick's marginal contribution")


def test_stale_trick_detected():
    r = R.new_recipe(0.40)
    R.add_trick(r, "t1", "a", 0.55)
    R.add_trick(r, "t2", "b", 0.60)                             # best now 0.60
    # turning t1 off still leaves 0.60 (t2 absorbed it) -> t1 is stale; t2 off drops to 0.50
    stale = R.stale_tricks(r, {"t1": 0.60, "t2": 0.50})
    assert stale == ["t1"]
    print("ok  regression guard flags a trick that no longer earns its slot")


def test_dev_holdout_gap_and_budget():
    g = R.dev_holdout_gap(dev=0.62, holdout=0.55)
    assert g["gap"] == 0.07 and g["overfitting_dev"] is True     # climbing dev, holdout stalls
    b = R.holdout_budget(queries_spent=10, cap=10)
    assert b["exhausted"] and b["remaining"] == 0
    print("ok  dev-holdout gap flags eval overfitting; holdout query budget tracked")


def test_lower_is_better_recipe():
    r = R.new_recipe(0.20, higher_better=False)                 # WER: lower better
    e = R.add_trick(r, "finetune", "ft", 0.14)
    assert e["marginal"] == 0.06 and e["banked"]                # WER dropped 0.06 -> a win
    print("ok  higher_better=False recipe banks a WER *drop* as positive lift")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
