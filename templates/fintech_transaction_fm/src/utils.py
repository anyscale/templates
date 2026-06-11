"""Small shared helpers."""

from __future__ import annotations

import ray


def print_cluster_resources() -> None:
    resources = ray.cluster_resources()
    print("Ray cluster resources:")
    for resource, count in sorted(resources.items()):
        if not resource.startswith("node:"):
            print(f"  {resource:<20} {count}")
    nodes = ray.nodes()
    print(f"\nCluster nodes: {len(nodes)}")
    for n in nodes:
        res = ", ".join(
            f"{k}={v}" for k, v in n["Resources"].items() if not k.startswith("node:")
        )
        print(f"  {n['NodeManagerAddress']:<20} alive={n['Alive']}  {res}")


def sample_serve_payload(tokenized_path: str) -> dict:
    """Build one realistic request payload from the tokenized dataset."""
    import json

    ds = ray.data.read_parquet(tokenized_path)
    row = ds.take(1)[0]
    payload = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in row.items()}
    for k in ("label", "split", "weight", "kind",
              "raw_amount", "raw_hour", "raw_dow", "raw_mcc", "raw_ts"):
        payload.pop(k, None)
    return json.loads(json.dumps(payload))  # ensure JSON-serializable
