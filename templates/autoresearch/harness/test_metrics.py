"""Tests for the noise-floor foundation. Self-runnable: `python3 test_metrics.py`."""

import metrics as M


def _rank(*pairs):  # (y_true, score) -> records
    return [{"y_true": y, "score": s} for y, s in pairs]


def test_average_precision_perfect_and_known():
    perfect = _rank((1, .9), (1, .8), (0, .2), (0, .1))
    assert M.average_precision(perfect) == 1.0
    # one negative slips above one positive: AP = mean(precision@each pos)
    mixed = _rank((1, .9), (0, .8), (1, .7))          # pos at ranks 1 and 3
    assert abs(M.average_precision(mixed) - ((1/1) + (2/3)) / 2) < 1e-9
    print("ok  average_precision: 1.0 on perfect ranking, matches hand calc on mixed")


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    recs = _rank(*[(i % 2, i / 10) for i in range(40)])
    a = M.bootstrap_ci(recs, M.average_precision, n=200, seed=1)
    b = M.bootstrap_ci(recs, M.average_precision, n=200, seed=1)
    assert a == b                                     # seeded -> deterministic
    assert a["lo"] <= a["value"] <= a["hi"]
    print("ok  bootstrap_ci brackets the point estimate and is deterministic")


def test_paired_bootstrap_detects_real_gain_on_a_realistic_eval():
    # 20 pos + 20 neg. B ranks every positive above every negative (AP=1); A does the reverse.
    # With enough examples the resamples almost always contain both classes -> gain is clean.
    a = _rank(*([(1, 0.1 + i / 1000) for i in range(20)] + [(0, 0.9 - i / 1000) for i in range(20)]))
    b = _rank(*([(1, 0.9 - i / 1000) for i in range(20)] + [(0, 0.1 + i / 1000) for i in range(20)]))
    real = M.paired_bootstrap(a, b, M.average_precision, n=400, seed=2)
    assert real["delta"] > 0 and M.is_real_gain(real) and real["prob_b_better"] > 0.95, real
    print("ok  paired_bootstrap certifies a clean win on a realistically-sized eval")


def test_paired_bootstrap_rejects_a_nongain_and_tiny_evals_cant_certify():
    a = _rank((1, .9), (1, .8), (0, .2), (0, .1))
    same = M.paired_bootstrap(a, a, M.average_precision, n=300, seed=2)
    assert same["delta"] == 0.0 and not M.is_real_gain(same)   # identical models -> not real
    # a genuine but tiny 4-example eval CANNOT certify (CI touches 0) -- the few-positives lesson
    b = _rank((1, .9), (1, .8), (0, .2), (0, .1))
    tiny = M.paired_bootstrap(_rank((1, .1), (1, .2), (0, .8), (0, .9)), b,
                              M.average_precision, n=300, seed=2)
    assert tiny["delta"] > 0 and not M.is_real_gain(tiny)      # real direction, uncertifiable
    print("ok  rejects a non-gain, and a tiny eval can't certify even a true gain (correct!)")


def test_is_real_gain_is_directional():
    """Regression from the multiple-comparisons sim: a reliable LOSS (CI entirely below 0) is
    NOT a gain. Earlier code counted those as wins."""
    loss = {"delta": -0.15, "lo": -0.22, "hi": -0.08}       # reliably worse
    win = {"delta": 0.15, "lo": 0.08, "hi": 0.22}
    assert not M.is_real_gain(loss) and M.is_real_gain(win)
    assert M.is_real_gain(loss, higher_better=False)         # for WER, a drop IS the gain
    print("ok  is_real_gain is directional — a reliable regression is not a 'win'")


def test_p_value_gain():
    a = _rank(*([(1, 0.1 + i / 1000) for i in range(20)] + [(0, 0.9 - i / 1000) for i in range(20)]))
    b = _rank(*([(1, 0.9 - i / 1000) for i in range(20)] + [(0, 0.1 + i / 1000) for i in range(20)]))
    assert M.p_value_gain(a, b, M.average_precision, n=300, seed=1) < 0.05    # B clearly better
    assert M.p_value_gain(a, a, M.average_precision, n=300, seed=1) > 0.20    # no gain
    print("ok  p_value_gain is small for a real gain, large for none (feeds FDR)")


def test_min_detectable_effect_shrinks_with_more_data():
    small = [{"y_true": i % 4 == 0, "score": (i * 7) % 11 / 11} for i in range(40)]
    big = [{"y_true": i % 4 == 0, "score": (i * 7) % 11 / 11} for i in range(400)]
    mde_s = M.min_detectable_effect(small, M.average_precision, n=200, seed=1)
    mde_b = M.min_detectable_effect(big, M.average_precision, n=200, seed=1)
    assert mde_b["mde"] < mde_s["mde"]                       # more data -> detect smaller gains
    assert M.detectable(mde_b, target_gain=0.5) and not M.detectable(mde_s, target_gain=0.001)
    print("ok  MDE shrinks with eval size; detectable() gates a target gain")


def test_empty_input_guards():
    """Simulation: an empty slice must fail loud, not crash on randrange(0)."""
    for call in (lambda: M.bootstrap_ci([], M.accuracy),
                 lambda: M.paired_bootstrap([], [], M.accuracy)):
        try:
            call(); assert False, "empty input should raise"
        except ValueError:
            pass
    print("ok  empty record lists raise a clear error, not an opaque crash")


def test_positives_warning():
    assert M.positives_warning(12) is not None        # fraud-campaign territory
    assert M.positives_warning(500) is None
    print("ok  few-positives warning fires under the floor, silent above")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
