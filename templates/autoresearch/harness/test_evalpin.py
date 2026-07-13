"""Tests for R7 eval pinning. Self-runnable: `python3 test_evalpin.py`."""

import os
import tempfile

import evalpin as E


def test_same_spec_same_pin_order_independent():
    a = E.eval_pin({"rows": "gb1_three_vs_rest", "seed": 42, "metric": "spearman@scipy1.13"})
    b = E.eval_pin({"metric": "spearman@scipy1.13", "seed": 42, "rows": "gb1_three_vs_rest"})
    assert a == b                                    # key order must not matter
    print("ok  identical spec -> identical pin, regardless of key order")


def test_different_spec_different_pin():
    a = E.eval_pin({"rows": "msmarco", "seed": 0})
    b = E.eval_pin({"rows": "msmarco", "seed": 1})   # a seed change is a different eval
    assert a != b
    print("ok  any content change -> different pin (seed flip caught)")


def test_pin_shape():
    p = E.eval_pin({"x": 1})
    assert p.startswith("sha256:") and len(p) == len("sha256:") + 16
    print("ok  pin is a short sha256 handle")


def test_pin_file():
    with tempfile.TemporaryDirectory() as base:
        f = os.path.join(base, "benchmark.parquet")
        with open(f, "wb") as fh:
            fh.write(b"pretend-parquet-bytes")
        p1 = E.pin_file(f)
        with open(f, "ab") as fh:
            fh.write(b"x")                            # mutate the frozen eval
        assert E.pin_file(f) != p1                    # detected
    print("ok  file pin changes when the frozen eval file changes")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
