"""Tests for R1. Self-runnable (no pytest needed): `python test_registry.py`.

Each test encodes a REQUIREMENTS.md R1 clause — or a critiques.md fix — as an executable
assertion, so the design decisions can't silently rot.
"""

import os
import tempfile

import registry as R


def _run(campaign, run_id, status, gpu_hours, gpu_type, eval_pin="sha256:pinA",
         decision=None, **extra):
    return {
        "campaign": campaign, "run_id": run_id, "commit": "abc123", "rung": "proxy",
        "cost": {"gpu_hours": gpu_hours, "gpu_type": gpu_type, "usd_est": 0, "spot": True},
        "eval_pin": eval_pin, "seed_plan_commit": "seed01", "status": status,
        "decision": decision, **extra,
    }


def test_tier_weighting_is_stored():
    """critiques.md #14: cost carries A10G-equivalent hours weighted by GPU tier."""
    with tempfile.TemporaryDirectory() as base:
        R.append_run(base, _run("c", "r1", "SUCCEEDED", 10, "A100"))
        spend = R.campaign_spend(base, "c")
        assert spend["gpu_hours"] == 10
        assert spend["a10g_equiv_hours"] == 35.0, spend  # 10 * 3.5
    print("ok  tier weighting stored (A100 10h -> 35 A10G-eq)")


def test_the_7b_rl_run_is_miswaved():
    """critiques.md #14/#20, in running code: campaign 02's '~120 H100-hr' full run is
    labelled Wave 2 (60-400), but weighted it is 660 A10G-eq -> Wave 3. This assertion is
    the argument for the fix."""
    with tempfile.TemporaryDirectory() as base:
        R.append_run(base, _run("02-rl", "full", "SUCCEEDED", 120, "H100"))
        eq = R.campaign_spend(base, "02-rl")["a10g_equiv_hours"]
        assert eq == 660.0, eq
        assert eq > 400, "660 A10G-eq is Wave 3, not the Wave 2 the plan claims"
    print("ok  7B RL run is Wave 3 once tier-weighted (660 A10G-eq > 400)")


def test_idempotent_on_run_id_and_status():
    """REQUIREMENTS R1 / critiques.md #4: a double-fire of the same (run_id,status) is a
    no-op, so R4's envelope math never double-counts."""
    with tempfile.TemporaryDirectory() as base:
        assert R.append_run(base, _run("c", "r1", "SUCCEEDED", 4, "A10G")) is True
        assert R.append_run(base, _run("c", "r1", "SUCCEEDED", 4, "A10G")) is False
        assert R.campaign_spend(base, "c")["gpu_hours"] == 4  # counted once
    print("ok  idempotent on (run_id, status) — no double count")


def test_running_then_terminal_counts_once():
    """The single-writer model: a RUNNING heartbeat then a terminal row = one run, costed
    at the terminal row (critiques.md #4)."""
    with tempfile.TemporaryDirectory() as base:
        R.append_run(base, _run("c", "r1", "RUNNING", 0, "A10G"))
        R.append_run(base, _run("c", "r1", "SUCCEEDED", 5, "A10G"))
        assert R.campaign_spend(base, "c") == {"gpu_hours": 5, "a10g_equiv_hours": 5.0,
                                               "n_runs": 1}
    print("ok  RUNNING heartbeat + terminal row -> one run, terminal cost")


def test_hard_crash_row_written_by_monitor():
    """A run that dies to spot preemption can't write its own FAILED row; the monitor does.
    The registry accepts a terminal row for a run that only ever heartbeated RUNNING."""
    with tempfile.TemporaryDirectory() as base:
        R.append_run(base, _run("c", "r1", "RUNNING", 0, "A10G"))
        R.append_run(base, _run("c", "r1", "TERMINATED", 2, "A10G"))  # monitor-written
        assert R.reconstruct(base, "c")["runs"] == {"r1": "TERMINATED"}
    print("ok  monitor writes terminal row after a hard crash")


def test_decisions_are_typed_not_prose():
    """critiques.md #3: promote/kill/hold is a typed row with a mandatory reason, readable
    programmatically — not buried in a run's notes."""
    with tempfile.TemporaryDirectory() as base:
        R.append_run(base, _run("c", "r1", "SUCCEEDED", 4, "A10G"))
        R.append_decision(base, "c", ["r1"], "kill", "AP within CI of baseline", "seed01")
        state = R.reconstruct(base, "c")
        assert state["killed"] == {("r1",): "AP within CI of baseline"}
    print("ok  decisions are typed rows with a recorded reason")


def test_decision_requires_reason():
    with tempfile.TemporaryDirectory() as base:
        try:
            R.append_decision(base, "c", ["r1"], "kill", "", "seed01")
            assert False, "empty reason should raise"
        except ValueError:
            pass
    print("ok  a kill without a reason is rejected")


def test_cross_pin_comparison_is_detectable():
    """REQUIREMENTS R1: rows are only comparable within one eval_pin; a split must be
    detectable (the FM campaign's 112-fraud vs 2724-fraud tables)."""
    with tempfile.TemporaryDirectory() as base:
        R.append_run(base, _run("c", "r1", "SUCCEEDED", 1, "A10G", eval_pin="sha256:small"))
        R.append_run(base, _run("c", "r2", "SUCCEEDED", 1, "A10G", eval_pin="sha256:full"))
        assert R.check_eval_pin_consistency(base, "c") == ["sha256:full", "sha256:small"]
    print("ok  cross-pin comparison is detectable (2 distinct pins surfaced)")


def test_unknown_gpu_raises():
    """An un-priced GPU must not silently count as free budget."""
    with tempfile.TemporaryDirectory() as base:
        try:
            R.append_run(base, _run("c", "r1", "SUCCEEDED", 1, "B200"))
            assert False, "unknown gpu should raise"
        except ValueError:
            pass
    print("ok  unknown gpu_type is rejected, not costed as free")


def test_missing_required_field_raises():
    with tempfile.TemporaryDirectory() as base:
        bad = _run("c", "r1", "SUCCEEDED", 1, "A10G")
        del bad["eval_pin"]
        try:
            R.append_run(base, bad)
            assert False, "missing eval_pin should raise"
        except ValueError:
            pass
    print("ok  missing required field is rejected")


def test_conflicting_terminal_rows_are_detectable():
    """Adversarial self-review: idempotency dedupes (run_id,status) but a buggy monitor
    could write two *different* terminal statuses for one run. That must be surfaced, not
    silently resolved to the last writer."""
    with tempfile.TemporaryDirectory() as base:
        R.append_run(base, _run("c", "r1", "FAILED", 2, "A10G"))
        R.append_run(base, _run("c", "r1", "SUCCEEDED", 5, "A10G"))  # conflict
        assert R.conflicting_terminal_runs(base, "c") == ["r1"]
        assert R.reconstruct(base, "c")["conflicting_terminal_runs"] == ["r1"]
    print("ok  conflicting terminal statuses for one run are detectable")


def test_reconstruct_from_disk_alone():
    """The R1 'Done when': a fresh context rebuilds full campaign state from JSONL alone."""
    with tempfile.TemporaryDirectory() as base:
        R.append_run(base, _run("01-hstu", "r1", "SUCCEEDED", 8, "A10G"))
        R.append_run(base, _run("01-hstu", "r2", "SUCCEEDED", 40, "A100"))
        R.append_decision(base, "01-hstu", ["r2"], "promote", "beats baseline", "seed01")
        R.append_decision(base, "01-hstu", ["r1"], "kill", "no lift over freq prior", "seed01")
        state = R.reconstruct(base, "01-hstu")
        assert state["spend"]["a10g_equiv_hours"] == 8 + 140  # 40 * 3.5
        assert state["promoted"] == [["r2"]]
        assert state["killed"] == {("r1",): "no lift over freq prior"}
        assert state["runs"] == {"r1": "SUCCEEDED", "r2": "SUCCEEDED"}
    print("ok  full campaign state reconstructs from JSONL alone")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
