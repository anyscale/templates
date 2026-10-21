"""Instance prices from a published source, never from the model's memory (Stephen's idea).

An agent asked "what does a g6e.4xlarge cost?" will answer confidently and may be wrong. The
ledger can't rest on that, so every price the harness uses comes from one of two places,
each recorded on the row it priced:

- **On-demand:** AWS's public EC2 price list (no credentials needed). `refresh()` streams the
  region's offer file once and writes `prices_<region>.json`: per instance type, vCPUs, GPUs
  and the Linux on-demand $/hr, stamped with AWS's publication date.
- **Spot:** EC2 spot price history (`aws ec2 describe-spot-price-history`) for the node's own
  AZ and its own run window, time-weighted. Needs `ec2:DescribeSpotPriceHistory` (Anyscale
  cluster node roles have it). Without it, spot nodes are priced at on-demand and the row
  says so: an upper bound, never a guess.

An instance type missing from the table is left unpriced and flagged, not filled in.
These are AWS's prices for the instances. The Anyscale platform charge is separate.

Works offline: `get()` refreshes when the saved snapshot is over a week old and the network
is up, and otherwise uses the snapshot committed next to this file (`prices_us-west-2.json`).

    python3 prices.py refresh [region]        # ~300 MB streamed, seconds on a fast link
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

OFFER_URL = ("https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/"
             "{region}/index.csv")
HERE = os.path.dirname(os.path.abspath(__file__))
# GPU families the harness prices (hooks._FAMILY) plus the CPU families heads and CPU workers use.
FAMILIES = ("g4dn", "g5", "g6", "g6e", "p4d", "p4de", "p5", "m5", "m6i", "m6a", "m7i", "m7a",
            "c5", "c6i", "c7i", "r5", "r6i", "r7i")


def table_path(region: str = "us-west-2") -> str:
    return os.path.join(HERE, f"prices_{region}.json")


def parse_offer_csv(lines, families=FAMILIES) -> tuple[dict, str]:
    """({instance_type: {vcpus, gpus, on_demand_usd_hr}}, publication_date) from the AWS EC2
    offer CSV. Keeps only shared-tenancy Linux on-demand rows with no pre-installed software:
    the price an Anyscale node actually pays."""
    lines = iter(lines)
    pub = ""
    for line in lines:  # metadata rows, then the header
        if line.startswith('"Publication Date"'):
            pub = next(csv.reader([line]))[1]
        if line.startswith('"SKU"'):
            header = next(csv.reader([line]))
            break
    else:
        raise ValueError("no header row in the offer file")
    col = {name: i for i, name in enumerate(header)}
    want = {"TermType": "OnDemand", "Operating System": "Linux", "Tenancy": "Shared",
            "Pre Installed S/W": "NA", "CapacityStatus": "Used", "Unit": "Hrs"}
    out: dict[str, dict] = {}
    for row in csv.reader(lines):
        if len(row) < len(header):
            continue
        itype = row[col["Instance Type"]]
        if not itype or itype.split(".")[0] not in families:
            continue
        if any(row[col[k]] != v for k, v in want.items()):
            continue
        price = float(row[col["PricePerUnit"]] or 0)
        if price <= 0:
            continue
        out[itype] = {"vcpus": int(row[col["vCPU"]] or 0),
                      "gpus": int(float(row[col["GPU"]] or 0)),
                      "on_demand_usd_hr": price}
    return out, pub


def refresh(region: str = "us-west-2", path: str | None = None) -> dict:
    """Download the region's offer file and write the lookup table. The only network call."""
    url = OFFER_URL.format(region=region)
    with urllib.request.urlopen(url, timeout=600) as resp:
        instances, pub = parse_offer_csv(io.TextIOWrapper(resp, encoding="utf-8"))
    table = {"source": url, "publication_date": pub, "region": region,
             "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "instances": dict(sorted(instances.items()))}
    with open(path or table_path(region), "w") as f:
        json.dump(table, f, indent=1)
    return table


def load(region: str = "us-west-2", path: str | None = None) -> dict | None:
    p = path or table_path(region)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _age_days(table: dict) -> float:
    fetched = datetime.fromisoformat(table["fetched_at"])
    return (datetime.now(timezone.utc) - fetched).total_seconds() / 86400


def get(region: str = "us-west-2", max_age_days: float = 7, path: str | None = None,
        fetch=refresh) -> dict | None:
    """The price table to use, online or not.

    Uses the saved snapshot while it's younger than `max_age_days`. Otherwise tries to refresh
    it from AWS; with no internet (or any download error) it falls back to the saved snapshot,
    and for a region with no snapshot, to the committed us-west-2 one. The table's `basis`
    says which was used, so a stale or borrowed price is visible on the row it priced."""
    saved = load(region, path)
    if saved and _age_days(saved) < max_age_days:
        return {**saved, "basis": "snapshot"}
    try:
        return {**fetch(region, path), "basis": "refreshed"}
    except (OSError, ValueError):  # URLError, timeouts and DNS failures are OSErrors
        pass
    if saved:
        return {**saved, "basis": "snapshot (offline, refresh failed)"}
    default = load("us-west-2")
    if default:
        return {**default, "basis": f"us-west-2 snapshot used for {region} (offline)"}
    return None


def spot_history(instance_type: str, az: str, start: datetime, end: datetime) -> list[dict]:
    """Raw spot price history for one type in one AZ over [start, end]; [] if unavailable."""
    cmd = ["aws", "ec2", "describe-spot-price-history", "--region", az[:-1],
           "--availability-zone", az, "--instance-types", instance_type,
           "--product-descriptions", "Linux/UNIX", "--start-time", start.isoformat(),
           "--end-time", end.isoformat(), "--output", "json"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return json.loads(out.stdout)["SpotPriceHistory"] if out.returncode == 0 else []
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        return []


def time_weighted(history: list[dict], start: datetime, end: datetime) -> float | None:
    """Average $/hr over [start, end]. Each history entry is the price from its timestamp until
    the next change; the entry in effect at `start` (at or before it) is included."""
    pts = sorted((datetime.fromisoformat(h["Timestamp"].replace("Z", "+00:00")),
                  float(h["SpotPrice"])) for h in history)
    if not pts or end <= start:
        return None
    before = [p for p in pts if p[0] <= start]
    pts = ([(start, before[-1][1])] if before else []) + [p for p in pts if start < p[0] < end]
    if not pts:
        return None
    total = 0.0
    for (t, price), nxt in zip(pts, pts[1:] + [(end, None)]):
        total += price * (nxt[0] - max(t, start)).total_seconds()
    covered = (end - max(pts[0][0], start)).total_seconds()
    return round(total / covered, 6) if covered > 0 else None


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "refresh":
        t = refresh(sys.argv[2] if len(sys.argv) > 2 else "us-west-2")
        print(f"{len(t['instances'])} instance types, AWS publication {t['publication_date']} "
              f"-> {table_path(t['region'])}")
    else:
        print(__doc__)
