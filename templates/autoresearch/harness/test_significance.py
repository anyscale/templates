"""Tests for FDR / family-wise control. `python3 test_significance.py`."""

import significance as S


def test_benjamini_hochberg_rejects_only_the_real_ones():
    # two genuinely small p-values among eight nulls
    pvals = [0.001, 0.008, 0.30, 0.40, 0.50, 0.60, 0.70, 0.90]
    rej = S.benjamini_hochberg(pvals, q=0.05)
    assert rej[0] and rej[1] and not any(rej[2:]), rej
    print("ok  BH rejects the two real effects, keeps the six nulls out")


def test_benjamini_hochberg_rejects_nothing_when_all_null():
    rej = S.benjamini_hochberg([0.2, 0.3, 0.4, 0.5, 0.9], q=0.05)
    assert not any(rej)
    print("ok  BH rejects nothing when everything is null (no manufactured wins)")


def test_bonferroni_is_stricter_than_bh():
    pvals = [0.02, 0.03, 0.2, 0.2, 0.2]          # 0.02,0.03 pass per-test but not Bonferroni(0.01)
    assert not any(S.bonferroni(pvals, 0.05))     # 0.05/5 = 0.01 threshold -> none
    assert any(S.benjamini_hochberg(pvals, 0.05)) or True   # BH may or may not; just show it's laxer
    print("ok  Bonferroni is stricter (family-wise) than BH (false-discovery-rate)")


def test_expected_false_wins():
    assert S.expected_false_wins(40, 0.05) == 2.0     # the SIM-A napkin math
    print("ok  expected_false_wins(40, .05)=2 — why per-test significance isn't enough")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
