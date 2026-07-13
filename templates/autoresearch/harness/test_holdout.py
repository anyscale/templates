"""Tests for Thresholdout. `python3 test_holdout.py`."""

import holdout as H


def test_agreement_returns_dev_free():
    t = H.Thresholdout(threshold=0.02, noise=0.0, budget=3, seed=0)
    for _ in range(5):
        r = t.query(dev_stat=0.300, holdout_stat=0.305)     # agree within threshold
        assert r["value"] == 0.3 and not r["overfit"]
    assert t.budget == 3                                     # no queries spent
    print("ok  when dev and holdout agree, Thresholdout returns dev for free")


def test_divergence_reveals_holdout_and_spends_budget():
    t = H.Thresholdout(threshold=0.02, noise=0.0, budget=3, seed=0)
    r = t.query(dev_stat=0.462, holdout_stat=0.283)          # overfit dev by selection
    assert r["overfit"] and abs(r["value"] - 0.283) < 1e-6 and t.budget == 2
    print("ok  divergence flags overfitting, reveals the honest holdout, spends a query")


def test_budget_exhaustion_falls_back_to_dev_with_warning():
    t = H.Thresholdout(threshold=0.02, noise=0.0, budget=1, seed=0)
    t.query(0.9, 0.3)                                        # spends the only query
    r = t.query(0.9, 0.3)
    assert t.budget == 0 and not r["overfit"] and "exhausted" in r["note"]
    print("ok  once the budget is spent, only dev is returned (with a warning)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
