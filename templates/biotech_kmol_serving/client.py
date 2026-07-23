"""Test / benchmark client for the kMoL ensemble service.

Usage:
    # Local (serve run):
    python client.py

    # Anyscale service:
    python client.py --url https://<service-url> --token <bearer-token>
"""

import argparse
import time
from typing import Optional

import requests

# A few valid SMILES to smoke-test with.
SAMPLE_SMILES = [
    "CCO",                          # ethanol
    "c1ccccc1",                     # benzene
    "CC(=O)OC1=CC=CC=C1C(=O)O",     # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # caffeine
]


def _headers(token: Optional[str]) -> dict:
    return {"Authorization": f"Bearer {token}"} if token else {}


def single(url: str, token: Optional[str]) -> None:
    for smi in SAMPLE_SMILES:
        r = requests.post(url, json={"smiles": smi}, headers=_headers(token), timeout=30)
        r.raise_for_status()
        print(f"{smi:35s} -> {r.json()}")


def batch(url: str, token: Optional[str]) -> None:
    r = requests.post(url, json={"smiles": SAMPLE_SMILES}, headers=_headers(token), timeout=60)
    r.raise_for_status()
    print(f"batch of {len(SAMPLE_SMILES)} -> {len(r.json())} predictions")


def bench(url: str, token: Optional[str], n: int, concurrency: int) -> None:
    """Crude open-loop-ish throughput probe. For real scaling numbers, drive this
    from a *separate node* with a proper load generator (see README REC 4)."""
    from concurrent.futures import ThreadPoolExecutor

    payloads = [{"smiles": SAMPLE_SMILES[i % len(SAMPLE_SMILES)]} for i in range(n)]
    hdrs = _headers(token)

    def _one(p):
        return requests.post(url, json=p, headers=hdrs, timeout=60).status_code

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        codes = list(pool.map(_one, payloads))
    elapsed = time.perf_counter() - start
    ok = sum(c == 200 for c in codes)
    print(f"{ok}/{n} ok in {elapsed:.2f}s -> {n / elapsed:,.1f} molecules/sec "
          f"(concurrency={concurrency})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/")
    ap.add_argument("--token", default=None)
    ap.add_argument("--mode", choices=["single", "batch", "bench"], default="single")
    ap.add_argument("--n", type=int, default=2000, help="bench: total requests")
    ap.add_argument("--concurrency", type=int, default=64, help="bench: parallel clients")
    args = ap.parse_args()

    if args.mode == "single":
        single(args.url, args.token)
    elif args.mode == "batch":
        batch(args.url, args.token)
    else:
        bench(args.url, args.token, args.n, args.concurrency)
