"""Tests for R8 artifact safety. Self-runnable: `python3 test_artifacts.py`."""

import os
import tempfile

import artifacts as A


def _touch(p, body="x"):
    with open(p, "w") as f:
        f.write(body)


def test_move_aside_renames_never_deletes():
    with tempfile.TemporaryDirectory() as base:
        model = os.path.join(base, "model")
        _touch(model, "weights")
        moves = A.move_aside([model], "20260711_120000")
        assert len(moves) == 1
        assert not os.path.exists(model)                     # original name freed
        backup = A.aside_path(model, "20260711_120000")
        assert os.path.exists(backup)
        assert open(backup).read() == "weights"              # content preserved, not deleted
    print("ok  move_aside renames to _old_<stamp>, never deletes")


def test_shared_stamp_groups_artifacts():
    with tempfile.TemporaryDirectory() as base:
        kinds = [os.path.join(base, k) for k in ("model", "embeddings", "tokenized")]
        for p in kinds:
            _touch(p)
        stamp = "20260711_130000"
        A.move_aside(kinds, stamp)
        for p in kinds:
            assert os.path.exists(A.aside_path(p, stamp))     # all moved under one stamp
    print("ok  one shared stamp groups every artifact kind of a rerun")


def test_double_move_is_guarded_noop():
    """A double-submit must not clobber the first backup."""
    with tempfile.TemporaryDirectory() as base:
        model = os.path.join(base, "model")
        _touch(model, "run1")
        A.move_aside([model], "stampA")
        _touch(model, "run2")                                 # a second run writes a new model
        moves2 = A.move_aside([model], "stampA")              # same stamp double-fire
        assert moves2 == []                                   # no-op: backup already exists
        assert open(A.aside_path(model, "stampA")).read() == "run1"  # first backup intact
        assert open(model).read() == "run2"                   # current model untouched
    print("ok  double move_aside with same stamp is a guarded no-op (no clobber)")


def test_missing_path_skipped():
    with tempfile.TemporaryDirectory() as base:
        moves = A.move_aside([os.path.join(base, "nope")], "s")
        assert moves == []
    print("ok  a missing artifact is skipped, not an error")


def test_restore_is_idempotent():
    with tempfile.TemporaryDirectory() as base:
        model = os.path.join(base, "model")
        _touch(model, "v1")
        A.move_aside([model], "s1")
        assert A.restore(model, "s1") is True
        assert open(model).read() == "v1"
        assert A.restore(model, "s1") is False                # already restored -> no-op
    print("ok  restore moves backup back, idempotent on second call")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
