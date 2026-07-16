"""Tests for the Claude Code submit hooks. Self-runnable: `python3 test_hooks.py`."""

import os
import tempfile

import hooks
import registry as R

JOB_YAML = """\
# autoresearch: campaign=fm wave=1 rung=proxy est_hours=1.0 eval_pin=sha256:abc
name: fm-xl-test
entrypoint: python train.py
compute_config:
  cloud: aws
  head_node:
    instance_type: m5.4xlarge
  worker_nodes:
  - instance_type: g5.xlarge
    name: gpu-1x
    min_nodes: 0
    max_nodes: 8
    market_type: ON_DEMAND
  - instance_type: m5.4xlarge
    name: cpu-workers
    min_nodes: 0
    max_nodes: 4
    market_type: ON_DEMAND
"""


def _payload(cmd, response=""):
    return {"tool_name": "Bash", "tool_input": {"command": cmd},
            "tool_response": {"output": response}}


def test_parse_compute_worst_case_fleet():
    f = hooks.parse_compute(JOB_YAML)
    assert f == {"gpu_type": "A10G", "num_gpus": 8, "instance_type": "g5.xlarge",
                 "spot": False}, f
    print("ok  compute parse: 8x g5.xlarge worst case, CPU groups ignored, on-demand seen")


def test_multi_tier_refused_and_unknown_raises():
    two = JOB_YAML + "  - instance_type: p5.48xlarge\n    max_nodes: 1\n"
    try:
        hooks.parse_compute(two); assert False, "multi-tier must raise"
    except ValueError as e:
        assert "separate runs" in str(e)
    try:
        hooks.gpu_shape("trn1.32xlarge"); assert False, "unpriced GPU must raise"
    except ValueError as e:
        assert "unknown instance family" in str(e)
    print("ok  multi-tier configs and un-priced GPUs are refused, not guessed")


def test_pre_gate_allows_declared_and_refuses_over_cap():
    with tempfile.TemporaryDirectory() as tmp:
        yml = os.path.join(tmp, "job.yaml")
        open(yml, "w").write(JOB_YAML)
        base = os.path.join(tmp, "base")
        assert hooks.pre(_payload(f"anyscale job submit {yml}"), base) == 0  # 8 eq < cap 10
        big = JOB_YAML.replace("est_hours=1.0", "est_hours=10")  # 80 eq > proxy cap 10
        open(yml, "w").write(big)
        assert hooks.pre(_payload(f"anyscale job submit {yml}"), base) == 2
        open(yml, "w").write(JOB_YAML.replace("# autoresearch:", "# nope:"))
        assert hooks.pre(_payload(f"anyscale job submit {yml}"), base) == 2  # undeclared
        assert hooks.pre(_payload("anyscale job list"), base) == 0  # non-submit untouched
    print("ok  pre gate: declared+affordable passes, over-cap and undeclared are exit 2")


def test_post_ledgers_running_row_idempotently():
    with tempfile.TemporaryDirectory() as tmp:
        yml = os.path.join(tmp, "job.yaml")
        open(yml, "w").write(JOB_YAML)
        base = os.path.join(tmp, "base")
        p = _payload(f"anyscale job submit {yml}", "submitted prodjob_abc123 ok")
        assert hooks.post(p, base) == 0
        assert hooks.post(p, base) == 0  # duplicate fire = registry no-op
        rows = R.terminal_runs(base, "fm")
        assert len(rows) == 1 and rows[0]["run_id"] == "prodjob_abc123"
        assert rows[0]["status"] == "RUNNING" and rows[0]["cost"]["gpu_hours"] == 8.0
        assert rows[0]["cost"]["a10g_equiv_hours"] == 8.0
        assert rows[0]["eval_pin"] == "sha256:abc"
    print("ok  post hook: RUNNING row with committed estimate, idempotent on refire")


if __name__ == "__main__":
    test_parse_compute_worst_case_fleet()
    test_multi_tier_refused_and_unknown_raises()
    test_pre_gate_allows_declared_and_refuses_over_cap()
    test_post_ledgers_running_row_idempotently()
    print("all hooks tests passed")
