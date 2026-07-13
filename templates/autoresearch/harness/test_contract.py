"""Tests for the job contract + confound firewall. `python3 test_contract.py`."""

import contract as K


def test_valid_eval_output_passes():
    recs = [{"id": 1, "y_true": 1, "score": 0.9, "slices": {"len": "long"}},
            {"id": 2, "y_true": 0, "score": 0.1}]
    assert K.validate_eval_output(recs) == []
    print("ok  a conforming eval output validates clean")


def test_eval_output_problems_are_reported():
    bad = [{"id": 1, "y_true": 1},                      # no score/y_pred
           {"id": 1, "y_pred": 0, "y_true": 0}]         # duplicate id
    probs = K.validate_eval_output(bad)
    assert any("one of" in p for p in probs) and any("duplicate id" in p for p in probs)
    assert K.validate_eval_output([]) == ["eval output must be a non-empty list of records"]
    print("ok  missing score, duplicate id, and empty output are all reported")


def test_single_delta_enforces_one_change():
    base = {"readout": "linear", "seq_len": 512, "lr": 1e-3}
    ok = {"readout": "mlp", "seq_len": 512, "lr": 1e-3}
    assert K.single_delta(base, ok) == "readout"
    print("ok  single_delta returns the one changed flag")


def test_confounded_experiment_is_rejected():
    base = {"readout": "linear", "seq_len": 512}
    two = {"readout": "mlp", "seq_len": 1024}           # two changes -> can't attribute
    try:
        K.single_delta(base, two)
        assert False, "should reject a 2-flag diff"
    except ValueError as e:
        assert "confounded" in str(e)
    try:
        K.single_delta(base, dict(base))                # no change
        assert False, "should reject a no-op diff"
    except ValueError as e:
        assert "no delta" in str(e)
    print("ok  a confounded (2-flag) or no-op experiment is rejected at the firewall")


def test_describe_delta():
    assert K.describe_delta({"seq_len": 512}, {"seq_len": 1024}) == "seq_len: 512 -> 1024"
    print("ok  describe_delta renders the one change")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
