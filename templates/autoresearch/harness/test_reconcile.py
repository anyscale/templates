"""Tests for R6 reconciliation. Self-runnable: `python3 test_reconcile.py`."""

import tempfile

import reconcile
import registry as R

STATUS_DONE = """\
id: prodjob_abc123
name: fm-xl-test
state: SUCCEEDED
created_at: 2026-07-08 12:59:40.757184+00:00
updated_at: 2026-07-08 14:24:52.471673+00:00
"""

STATUS_RUNNING = STATUS_DONE.replace("SUCCEEDED", "RUNNING")


def _seed(base):
    R.append_run(base, {
        "campaign": "fm", "run_id": "prodjob_abc123", "commit": "c", "rung": "proxy",
        "eval_pin": "sha256:abc", "status": "RUNNING",
        "cost": {"gpu_hours": 12.0, "gpu_type": "A10G", "usd_est": 12.12, "spot": False},
        "fleet": {"gpu_type": "A10G", "num_gpus": 8, "spot": False},
    })


def test_parse_status_wall_clock():
    s = reconcile.parse_status(STATUS_DONE)
    assert s["state"] == "SUCCEEDED"
    assert s["hours"] == 1.4199  # 12:59:40.76 -> 14:24:52.47
    assert reconcile.parse_status("garbage") is None
    print("ok  status parse: state + wall-clock hours from created/updated")


def test_fallback_writes_terminal_row_from_wall_clock():
    with tempfile.TemporaryDirectory() as base:
        _seed(base)
        out = reconcile.sweep(base, fetch=lambda job_id: STATUS_DONE, measure=None)
        assert "terminal row written (fallback): SUCCEEDED" in out["prodjob_abc123"], out
        rows = R.terminal_runs(base, "fm")
        assert len(rows) == 1 and rows[0]["status"] == "SUCCEEDED"
        assert rows[0]["cost"]["gpu_hours"] == round(8 * 1.4199, 4)  # actuals, not estimate
        assert rows[0]["cost"]["wall_clock_hours"] == 1.4199
        assert "upper bound" in rows[0]["cost_basis"]
        # sweep again: nothing open, spend unchanged (idempotent end to end)
        assert reconcile.sweep(base, fetch=lambda j: STATUS_DONE, measure=None) == {}
        assert R.campaign_spend(base, "fm")["gpu_hours"] == round(8 * 1.4199, 4)
    print("ok  fallback: no cluster events -> wall-clock x fleet, labeled upper bound")


def test_sweep_leaves_running_jobs_open():
    with tempfile.TemporaryDirectory() as base:
        _seed(base)
        out = reconcile.sweep(base, fetch=lambda j: STATUS_RUNNING, measure=None)
        assert out["prodjob_abc123"] == "still RUNNING"
        assert R.terminal_runs(base, "fm")[0]["status"] == "RUNNING"
    print("ok  sweep: non-terminal jobs stay open for the next pass")


MEASURED = {"state": "SUCCEEDED", "cluster_id": "ses_x", "cluster_terminated": True,
            "cost": {"gpu_hours": 2.0, "cpu_hours": 40.0, "gpu_type": "mixed",
                     "by_tier": {"A10G": 1.0, "L40S": 1.0}, "usd": 3.7, "spot": False},
            "cost_basis": "actual: node lifetimes from the cluster event log"}


def test_sweep_writes_measured_actuals_and_frees_the_budget():
    with tempfile.TemporaryDirectory() as base:
        _seed(base)  # estimate: 12 GPU-h committed
        assert R.campaign_spend(base, "fm")["a10g_equiv_hours"] == 12.0
        out = reconcile.sweep(base, fetch=None, measure=lambda j: MEASURED)
        assert "terminal row written: SUCCEEDED (2.0 GPU-h, 40.0 CPU-h, $3.7)" == \
            out["prodjob_abc123"], out
        row = R.terminal_runs(base, "fm")[0]
        assert row["cost_basis"].startswith("actual") and row["estimate"]["gpu_hours"] == 12.0
        spend = R.campaign_spend(base, "fm")
        assert spend["a10g_equiv_hours"] == 2.8 and spend["usd"] == 3.7  # actual replaces estimate
    print("ok  sweep: measured actuals replace the estimate; spend drops from 12 to 2.8 A10G-eq")


def test_sweep_waits_for_running_jobs_and_teardown():
    with tempfile.TemporaryDirectory() as base:
        _seed(base)
        out = reconcile.sweep(base, measure=lambda j: {**MEASURED, "state": "RUNNING"})
        assert out["prodjob_abc123"] == "still RUNNING"
        out = reconcile.sweep(base, measure=lambda j: {**MEASURED, "cluster_terminated": False})
        assert out["prodjob_abc123"] == "waiting for cluster teardown"
        assert R.terminal_runs(base, "fm")[0]["status"] == "RUNNING"
    print("ok  sweep: no terminal row until the job ends and its cluster is gone")


if __name__ == "__main__":
    test_sweep_writes_measured_actuals_and_frees_the_budget()
    test_sweep_waits_for_running_jobs_and_teardown()
    test_parse_status_wall_clock()
    test_fallback_writes_terminal_row_from_wall_clock()
    test_sweep_leaves_running_jobs_open()
    print("all reconcile tests passed")
