"""
Load test for the recommendation service.

Usage (from multi-model-serving/):
    python scripts/load_test.py --url http://localhost:8000 --concurrency 20 --requests 100
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

DEMO_QUERIES = [
    {"query": "wireless bluetooth headphones noise cancelling"},
    {"query": "women's running shoes lightweight breathable"},
    {"query": "daily face moisturizer SPF sensitive skin"},
    {"query": "instant pot pressure cooker 6 quart"},
    {"query": "adjustable dumbbell set home gym"},
    {"query": "vitamin c serum brightening anti-aging"},
    {"query": "insulated water bottle stainless steel 32oz"},
    {"query": "mechanical keyboard rgb gaming"},
    {"query": "yoga mat non-slip thick eco-friendly"},
    {"query": "air fryer digital compact kitchen"},
]


def _send_request(url: str, payload: dict) -> dict:
    start = time.time()
    resp = requests.post(f"{url}/recommend", json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    data["_wall_ms"] = (time.time() - start) * 1000
    return data


def run_load_test(url: str, concurrency: int, total_requests: int) -> None:
    print(f"\nLoad test: {total_requests} requests at concurrency={concurrency}")
    print(f"Target: {url}/recommend\n")

    payloads = [
        DEMO_QUERIES[i % len(DEMO_QUERIES)] for i in range(total_requests)
    ]
    latencies = []
    stage_latencies = {k: [] for k in ["encode_ms", "retrieve_ms", "rerank_ms"]}
    errors = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_send_request, url, p): p for p in payloads}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                latencies.append(result["_wall_ms"])
                for key in stage_latencies:
                    stage_latencies[key].append(result["stages"].get(key, 0))
                if i % 10 == 0 or i == total_requests:
                    print(f"  Completed {i}/{total_requests}...")
            except Exception as e:
                errors += 1
                print(f"  ERROR: {e}")

    if not latencies:
        print("No successful requests.")
        return

    latencies = np.array(latencies)
    print(f"\n{'─' * 55}")
    print(f"  Requests completed : {len(latencies)} / {total_requests}")
    print(f"  Errors             : {errors}")
    print(f"  Throughput         : {len(latencies) / (latencies.sum() / 1000 / concurrency):.1f} RPS")
    print(f"\n  End-to-end latency (ms):")
    print(f"    p50  : {np.percentile(latencies, 50):.1f}")
    print(f"    p95  : {np.percentile(latencies, 95):.1f}")
    print(f"    p99  : {np.percentile(latencies, 99):.1f}")
    print(f"    max  : {latencies.max():.1f}")
    print(f"\n  Per-stage averages (ms):")
    for key, vals in stage_latencies.items():
        print(f"    {key:<18}: {np.mean(vals):.1f}")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--requests", dest="total_requests", type=int, default=100)
    args = parser.parse_args()
    run_load_test(args.url, args.concurrency, args.total_requests)
