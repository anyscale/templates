"""Tests for measured usage. Self-runnable: `python3 test_usage.py`.

The event fixtures are real cluster events from the pathology PoC (Oct 2026), trimmed to the
lines usage.py reads and to whole seconds. A live run on 2026-10-21 (full timestamps) gave
the same numbers to within those dropped fractions of a second.
"""

from datetime import datetime, timezone

import usage

TABLE = {"instances": {
    "m5.2xlarge": {"vcpus": 8, "gpus": 0, "on_demand_usd_hr": 0.384},
    "g5.2xlarge": {"vcpus": 8, "gpus": 1, "on_demand_usd_hr": 1.212},
    "g6.2xlarge": {"vcpus": 8, "gpus": 1, "on_demand_usd_hr": 0.9776},
}}


def _events(pairs):
    return [{"created_at": t + "+00:00", "message": m} for t, m in pairs]


# prodjob_12gvzqe3x8xuv1hbfwxuvgs85v: 4 GPU nodes on two tiers, an on-demand -> spot
# replacement whose launch the event log never recorded.
REPLACEMENT = _events([
    ("2026-10-08T03:59:13", "[autoscaler] [head-node|m5.2xlarge] [us-west-2a] [on-demand] Launched 1 instance."),
    ("2026-10-08T04:02:28", "[autoscaler] [g6.2xlarge/on-demand|g6.2xlarge] [us-west-2c] [on-demand] Launched 1 instance."),
    ("2026-10-08T04:02:50", "[autoscaler] [g5.2xlarge/on-demand|g5.2xlarge] [us-west-2c] [on-demand] Launched 3 instances."),
    ("2026-10-08T04:04:04", "[autoscaler] Cluster upscaled to {10 CPU, 1 GPU}."),
    ("2026-10-08T04:04:54", "[autoscaler] Cluster upscaled to {34 CPU, 4 GPU}."),
    ("2026-10-08T05:12:11", "[autoscaler] Cluster upscaled to {42 CPU, 5 GPU}."),
    ("2026-10-08T05:12:11", "[autoscaler] Terminating 1 g6.2xlarge/on-demand instances (replaced with g6.2xlarge/spot instances)."),
    ("2026-10-08T05:12:12", "[autoscaler] Cluster resized to {34 CPU, 4 GPU}."),
    ("2026-10-08T05:13:24", "Cluster is terminated (reason: job or service complete)."),
])

# prodjob_hzvirw64b8pmkaqwig3ua3tz8l: a 3-hour GPU job whose autoscaler never logged a total.
NO_TOTALS = _events([
    ("2026-10-06T19:28:32", "[autoscaler] [head-node|m5.2xlarge] [us-west-2c] [on-demand] Launched 1 instance."),
    ("2026-10-06T19:34:00", "[autoscaler] [g5.2xlarge/on-demand|g5.2xlarge] [us-west-2b] [on-demand] Launched 1 instance."),
    ("2026-10-06T22:43:56", "Cluster is terminated (reason: job or service complete)."),
])

# A cluster that lost nodes and restarted under the same job (shape of
# prodjob_bwbw7g1gbnhkni8vdhg7neqw1f), plus a spot node.
RESTART = _events([
    ("2026-10-09T06:00:00", "[autoscaler] [head-node|m5.2xlarge] [us-west-2d] [on-demand] Launched 1 instance."),
    ("2026-10-09T06:00:00", "[autoscaler] [g5.2xlarge/spot|g5.2xlarge] [us-west-2d] [spot] Launched 2 instances."),
    ("2026-10-09T06:01:00", "[autoscaler] Cluster upscaled to {24 CPU, 2 GPU}."),
    ("2026-10-09T07:00:00", "[autoscaler] Cluster resized to {16 CPU, 1 GPU}."),
    ("2026-10-09T08:00:00", "[autoscaler] [head-node|m5.2xlarge] [us-west-2d] [on-demand] Launched 1 instance."),
    ("2026-10-09T08:00:00", "[autoscaler] [g5.2xlarge/on-demand|g5.2xlarge] [us-west-2d] [on-demand] Launched 1 instance."),
    ("2026-10-09T09:00:00", "Cluster is terminated (reason: job or service complete)."),
])


def test_replacement_counts_every_node_and_the_unlogged_one():
    nodes, steps = usage.node_intervals(REPLACEMENT, TABLE)
    cost = usage.summarize(nodes, steps, TABLE)
    assert cost["by_tier"] == {"L4": 1.1619, "A10G": 3.5489}, cost["by_tier"]
    assert cost["unattributed_gpu_hours"] == 0.0206  # the spot node, 05:12:11 -> 05:13:24
    assert cost["gpu_hours"] == 4.7109 and cost["gpu_type"] == "mixed"
    assert cost["instances"]["g6.2xlarge/on-demand"]["nodes"] == 1
    assert cost["autoscaler_gpu_hours"] == 4.5808  # cross-check agrees within boot time
    assert cost["usd"] == 5.91 and "flags" not in cost
    print("ok  replacement: two tiers split, unlogged spot node counted, $ from the table")


def test_job_with_no_autoscaler_totals_still_counted():
    nodes, steps = usage.node_intervals(NO_TOTALS, TABLE)
    cost = usage.summarize(nodes, steps, TABLE)
    assert cost["autoscaler_gpu_hours"] == 0  # the totals alone would say "free"
    assert cost["gpu_hours"] == 3.1656 and cost["by_tier"] == {"A10G": 3.1656}
    assert cost["cpu_hours"] == 51.3778  # 8 vCPU head for 3.26 h + 8 vCPU GPU node for 3.17 h
    print("ok  no autoscaler totals: launch ledger still gives 3.17 GPU-h")


def test_restart_and_resize_close_nodes_and_spot_uses_history():
    nodes, steps = usage.node_intervals(RESTART, TABLE)
    ends = sorted((n["itype"], n["market"], n["end_reason"]) for n in nodes)
    assert ("g5.2xlarge", "spot", "cluster resized down") in ends
    assert ("g5.2xlarge", "spot", "cluster restarted") in ends
    spot = usage.summarize(nodes, steps, TABLE, spot_price=lambda *a: 0.40)
    cost = usage.summarize(nodes, steps, TABLE)  # no spot history -> on-demand, flagged
    assert spot["gpu_hours"] == cost["gpu_hours"] == 4.0  # 1 + 2 spot, 1 on-demand
    assert spot["instances"]["g5.2xlarge/spot"]["usd"] == round(3 * 0.40, 2)
    assert any("spot priced at on-demand" in f for f in cost["flags"])
    print("ok  restart + resize close the right nodes; spot priced from history or flagged")


def test_unknown_instance_is_flagged_not_guessed():
    evs = _events([
        ("2026-10-09T06:00:00", "[autoscaler] [head-node|m5.2xlarge] [us-west-2d] [on-demand] Launched 1 instance."),
        ("2026-10-09T06:00:00", "[autoscaler] [g6e.12xlarge|g6e.12xlarge] [us-west-2d] [on-demand] Launched 1 instance."),
        ("2026-10-09T07:00:00", "Cluster is terminated (reason: user action)."),
    ])
    nodes, steps = usage.node_intervals(evs, TABLE)
    cost = usage.summarize(nodes, steps, TABLE)
    assert cost["gpu_hours"] == 4.0  # GPU count from the instance family, not the table
    assert cost["usd"] == 0.38 and any("unpriced" in f for f in cost["flags"])
    print("ok  instance missing from the price table: GPU-hours kept, $ flagged as unpriced")


def test_events_page_past_the_api_cap():
    page1 = [{"created_at": f"2026-10-09T06:{i // 60:02d}:{i % 60:02d}+00:00", "message": f"e{i}"}
             for i in range(usage.EVENTS_PAGE)]
    page2 = page1[-1:] + [{"created_at": "2026-10-09T23:00:00+00:00", "message": "last"}]
    calls = []

    def fake_get(path):
        calls.append(path)
        return {"results": page1 if len(calls) == 1 else page2}

    real, usage._get = usage._get, fake_get
    try:
        evs = usage.cluster_events("ses_x")
    finally:
        usage._get = real
    assert len(evs) == usage.EVENTS_PAGE + 1 and evs[-1]["message"] == "last"
    assert "start_time=" in calls[1]
    print("ok  events: pages by start_time past the 1,000 cap, no duplicates")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
