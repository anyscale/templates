"""Tests for the leak / target-leak canaries. `python3 test_canary.py`."""

import canary as C


def test_shuffled_label_control():
    # fraud base rate ~0.012; a healthy pipeline scores ~there after shuffling labels
    assert C.shuffled_label_control(0.015, prevalence_floor=0.012)["verdict"] == "CLEAN"
    # scores way above the floor with labels shuffled -> the pipeline is reading a leak
    bad = C.shuffled_label_control(0.40, prevalence_floor=0.012)
    assert bad["leaked"] and bad["verdict"] == "BROKEN"
    print("ok  shuffled-label control: collapses = CLEAN, stays high = BROKEN (leak)")


def test_too_good_too_early():
    assert C.too_good_too_early(0.995, plausible_ceiling=1.0)["suspicious"]      # ~perfect at step 1
    assert not C.too_good_too_early(0.55, plausible_ceiling=1.0)["suspicious"]
    print("ok  near-ceiling metric at step 1 is flagged SUSPICIOUS")


def test_audit_reports_worst_severity():
    a = C.audit(shuffled={"metric": 0.4, "floor": 0.01}, step1=0.5, ceiling=1.0)
    assert a["verdict"] == "BROKEN"                          # shuffle leak dominates
    clean = C.audit(shuffled={"metric": 0.012, "floor": 0.01}, step1=0.5, ceiling=1.0)
    assert clean["verdict"] == "CLEAN"
    print("ok  audit surfaces the worst-severity finding (BROKEN > SUSPICIOUS > CLEAN)")


def test_embedding_collapse():
    assert C.embedding_collapse(0.99)["collapsed"] and not C.embedding_collapse(0.3)["collapsed"]
    print("ok  embedding collapse fires when mean pairwise cosine -> 1")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
