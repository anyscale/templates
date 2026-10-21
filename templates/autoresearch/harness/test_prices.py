"""Tests for the price lookup. Self-runnable: `python3 test_prices.py`."""

from datetime import datetime, timezone

import prices

COLS = ["SKU", "TermType", "Unit", "PricePerUnit", "Instance Type", "vCPU", "Tenancy",
        "Operating System", "Pre Installed S/W", "CapacityStatus", "GPU"]


def _row(itype, price, **over):
    r = {"SKU": "x", "TermType": "OnDemand", "Unit": "Hrs", "PricePerUnit": price,
         "Instance Type": itype, "vCPU": "8", "Tenancy": "Shared", "Operating System": "Linux",
         "Pre Installed S/W": "NA", "CapacityStatus": "Used", "GPU": "1", **over}
    return ",".join(f'"{r[c]}"' for c in COLS)


OFFER = "\n".join([
    '"FormatVersion","v1.0"',
    '"Publication Date","2026-10-15T19:13:16Z"',
    ",".join(f'"{c}"' for c in COLS),
    _row("g5.2xlarge", "1.2120000000"),
    _row("g5.2xlarge", "0.9000000000", TermType="Reserved"),        # not on-demand
    _row("g5.2xlarge", "1.5000000000", **{"Operating System": "Windows"}),
    _row("g5.2xlarge", "1.3000000000", Tenancy="Dedicated"),
    _row("m5.2xlarge", "0.3840000000", GPU=""),
    _row("x2iedn.xlarge", "1.0000000000"),                           # family not priced
]).splitlines()


def test_offer_csv_keeps_only_the_price_a_node_pays():
    table, pub = prices.parse_offer_csv(OFFER)
    assert pub == "2026-10-15T19:13:16Z"
    assert table == {"g5.2xlarge": {"vcpus": 8, "gpus": 1, "on_demand_usd_hr": 1.212},
                     "m5.2xlarge": {"vcpus": 8, "gpus": 0, "on_demand_usd_hr": 0.384}}, table
    print("ok  offer file: shared-tenancy Linux on-demand only, publication date kept")


def test_spot_price_is_time_weighted_over_the_node_window():
    t = lambda h, m=0: datetime(2026, 10, 9, h, m, tzinfo=timezone.utc)
    hist = [{"Timestamp": "2026-10-09T05:00:00+00:00", "SpotPrice": "0.40"},  # in effect at start
            {"Timestamp": "2026-10-09T07:00:00+00:00", "SpotPrice": "0.60"}]
    assert prices.time_weighted(hist, t(6), t(8)) == 0.5   # 1 h at 0.40, 1 h at 0.60
    assert prices.time_weighted(hist, t(7, 30), t(8)) == 0.6
    assert prices.time_weighted([], t(6), t(8)) is None    # no history -> caller flags it
    print("ok  spot: time-weighted over the node's own window; no history -> None, not a guess")


def test_get_falls_back_to_the_snapshot_when_offline():
    import json, os, tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prices_test.json")
        old = {"source": "x", "publication_date": "2026-10-15", "region": "us-west-2",
               "fetched_at": "2026-01-01T00:00:00+00:00", "instances": {"g5.2xlarge": {
                   "vcpus": 8, "gpus": 1, "on_demand_usd_hr": 1.212}}}
        json.dump(old, open(path, "w"))

        def offline(region, path=None):
            raise OSError("no network")

        t = prices.get(path=path, fetch=offline)
        assert t["basis"].startswith("snapshot (offline") and "g5.2xlarge" in t["instances"]
        fresh = {**old, "fetched_at": datetime.now(timezone.utc).isoformat()}
        json.dump(fresh, open(path, "w"))
        assert prices.get(path=path, fetch=offline)["basis"] == "snapshot"  # no fetch needed
    shipped = prices.get(fetch=offline, max_age_days=float("inf"))
    assert shipped and shipped["instances"]["g5.2xlarge"]["gpus"] == 1  # committed snapshot
    print("ok  offline: stale snapshot used and labeled; committed us-west-2 table works as is")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)} passed")
