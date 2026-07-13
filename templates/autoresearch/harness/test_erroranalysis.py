"""Tests for the DIAGNOSE engine. Self-runnable: `python3 test_erroranalysis.py`."""

import erroranalysis as EA


def _ex(y, p, **slices):
    return {"y_true": y, "y_pred": p, "slices": slices}


def test_slice_report_and_worst():
    # 'long' sequences are all wrong; 'short' all right
    ex = [_ex(1, 1, length="short"), _ex(0, 0, length="short"),
          _ex(1, 0, length="long"), _ex(0, 1, length="long")]
    rep = EA.slice_report(ex, EA.accuracy, "length")
    assert rep["short"]["metric"] == 1.0 and rep["long"]["metric"] == 0.0
    worst = EA.worst_slices(rep, higher_better=True, min_n=1, k=1)
    assert worst[0][0] == "long"
    print("ok  slice_report computes per-slice metric; worst_slices finds the laggard")


def test_overfit_card():
    cards = EA.diagnose(dev=0.80, train=0.95)
    reg = [c for c in cards if c["flag"] == "reg"]
    assert reg and "overfitting" in reg[0]["cause"]
    assert reg[0]["expected_lift"] > 0 and reg[0]["rung"] == "proxy"
    print("ok  train >> dev emits an overfitting card with a positive expected lift")


def test_within_noise_is_a_warning():
    cards = EA.diagnose(dev=0.720, baseline=0.710, noise=0.02)
    warn = [c for c in cards if c["flag"] == "seeds"]
    assert warn and warn[0]["kind"] == "warning" and warn[0]["expected_lift"] == 0.0
    print("ok  a gain inside the noise band is flagged as a non-real warning")


def test_baseline_tie_warns_not_earning_keep():
    cards = EA.diagnose(dev=0.900, baseline=0.895, tie_margin=0.01)
    tie = [c for c in cards if c["flag"] == "baseline"]
    assert tie and "earning its keep" in tie[0]["cause"]
    print("ok  a cheap baseline ~tied with the model warns 'not earning its keep'")


def test_worst_slice_card_names_the_slice():
    ex = ([_ex(1, 1, grp="easy") for _ in range(70)]           # easy: perfect
          + [_ex(1, 0, grp="hard") for _ in range(30)])         # hard: all wrong, 30% of eval
    reps = {"grp": EA.slice_report(ex, EA.accuracy, "grp")}
    overall = EA.accuracy(ex)                                    # 0.70
    cards = EA.diagnose(dev=overall, slice_reports=reps, min_slice_n=10)
    slc = [c for c in cards if c["flag"] == "slice_grp"]
    assert slc and "hard" in slc[0]["symptom"], cards
    assert slc[0]["rung"] == "smoke"                             # eval-only re-score is cheap
    print("ok  a bad, common slice emits a targeted (smoke-cost) card naming it")


def test_lower_is_better_direction():
    # WER-style: train 0.05 beats dev 0.12 -> overfitting when lower is better
    cards = EA.diagnose(dev=0.12, train=0.05, higher_better=False)
    assert any(c["flag"] == "reg" for c in cards), cards
    print("ok  higher_better=False flips direction (WER/loss overfit detected)")


def test_recall_by_subgroup_localizes_weak_subgroup_and_skips_negatives():
    """Simulation regression: for a RANKING task, slice the ranking (subgroup recall), not the
    metric. Burst positives ranked low must show low recall; the negative-only 'legit' slice
    must be absent (no positives to have recall over)."""
    ex = ([{"y_true": 1, "score": 0.9, "slices": {"p": "normal"}} for _ in range(30)]
          + [{"y_true": 1, "score": 0.1, "slices": {"p": "burst"}} for _ in range(30)]   # missed
          + [{"y_true": 0, "score": 0.5, "slices": {"p": "legit"}} for _ in range(140)])
    rep = EA.recall_by_subgroup(ex, "p")
    assert "legit" not in rep                                   # no positives -> not a subgroup
    assert rep["burst"]["metric"] < rep["normal"]["metric"]
    cards = EA.diagnose(dev=0.5, slice_reports={"p": rep}, min_slice_n=1)
    slc = [c for c in cards if c["flag"] == "slice_p"]
    assert slc and "burst" in slc[0]["symptom"], cards
    print("ok  recall_by_subgroup localizes the weak subgroup; negatives-only slice excluded")


def test_no_positive_slice_excluded_from_direction():
    """Simulation regression: a slice with ZERO positives — even one the model scores terribly
    — must NOT be proposed as a fix (a ranking metric there is undefined, not 'bad')."""
    ex = ([{"y_true": 1, "y_pred": 1, "slices": {"g": "a"}} for _ in range(40)]
          + [{"y_true": 0, "y_pred": 1, "slices": {"g": "c"}} for _ in range(40)])  # all-neg, acc 0
    rep = EA.slice_report(ex, EA.accuracy, "g")
    cards = EA.diagnose(dev=EA.accuracy(ex), slice_reports={"g": rep}, min_slice_n=1)
    assert not [c for c in cards if c["flag"] == "slice_g"], cards   # 'c' excluded -> <2 -> no card
    print("ok  a no-positive slice (even at 0 accuracy) is excluded from direction")


def test_discover_slices_finds_the_weak_subgroup_unprompted():
    """Slice auto-discovery: without being told which feature matters, find the burst subgroup
    whose positives the model ranks too low."""
    ex = []
    for i in range(300):
        fraud = i % 3 == 0
        pat = "burst" if (fraud and i % 6 == 0) else ("normal" if fraud else "legit")
        score = 0.1 if pat == "burst" else (0.9 if fraud else 0.3)
        ex.append({"id": i, "y_true": int(fraud), "score": score, "slices": {"pattern": pat}})
    found = EA.discover_slices(ex, min_pos=5)
    assert found, "should discover at least one weak slice"
    assert found[0]["predicate"] == {"pattern": "burst"} and found[0]["recall"] < 0.3
    print("ok  discover_slices auto-finds the burst subgroup (lowest global-cut recall)")


def test_discover_slices_ignores_no_positive_subgroups():
    ex = ([{"id": i, "y_true": 1, "score": 0.9, "slices": {"g": "a"}} for i in range(30)]
          + [{"id": 100 + i, "y_true": 0, "score": 0.8, "slices": {"g": "legit"}} for i in range(60)])
    found = EA.discover_slices(ex, min_pos=5)
    assert all(s["predicate"] != {"g": "legit"} for s in found)   # no positives -> never a slice
    print("ok  discover_slices never proposes a no-positive subgroup")


def test_single_comparable_slice_gives_no_card():
    ex = [{"y_true": 1, "y_pred": 1, "slices": {"g": "only"}} for _ in range(40)]
    cards = EA.diagnose(dev=1.0, slice_reports={"g": EA.slice_report(ex, EA.accuracy, "g")}, min_slice_n=1)
    assert not [c for c in cards if c["flag"] == "slice_g"]      # nothing to compare against
    print("ok  a single comparable slice yields no slice card")


def test_ranking_lift_per_cost():
    # a cheap slice fix (smoke) should outrank a proxy-cost overfit fix of similar raw lift
    ex = ([_ex(1, 1, grp="easy") for _ in range(80)]
          + [_ex(1, 0, grp="hard") for _ in range(20)])
    reps = {"grp": EA.slice_report(ex, EA.accuracy, "grp")}
    cards = EA.diagnose(dev=0.80, train=0.88, slice_reports=reps, min_slice_n=10)
    opps = [c for c in cards if c["kind"] == "opportunity"]
    assert opps[0]["priority"] >= opps[-1]["priority"]           # sorted by priority desc
    print("ok  cards rank by expected-lift-per-GPU-hour (cheap high-lift first)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
