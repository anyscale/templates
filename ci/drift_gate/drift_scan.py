#!/usr/bin/env python3
"""Emit a JSON array of drifted template names to stdout.

Drift = current effective_hash differs from `tmpl_effective_hash` in
templates.ci.ray.io/templates/<name>/latest/channel.json (404 or
missing field = drifted)."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from effective_hash import effective_hash, load_entries  # noqa: E402
from validate_build_yaml import Entry  # noqa: E402


CHANNEL_URL = "https://templates.ci.ray.io/templates/{name}/latest/channel.json"
_SESSION = requests.Session()


def fetch_published_hash(name: str, *, timeout: float = 5.0) -> Optional[str]:
    """None on first-publish (404), missing field, or transient failure
    (treated as drifted — safe default; false-positive drift just means an
    extra publish)."""
    try:
        resp = _SESSION.get(CHANNEL_URL.format(name=name), timeout=timeout)
    except requests.exceptions.RequestException as exc:
        print(f"warning: fetch failed for {name}: {exc}", file=sys.stderr)
        return None
    if resp.status_code == 404:
        return None
    if not resp.ok:
        print(f"warning: {name} returned HTTP {resp.status_code}", file=sys.stderr)
        return None
    try:
        return resp.json().get("tmpl_effective_hash")
    except ValueError as exc:
        print(f"warning: {name} channel.json invalid: {exc}", file=sys.stderr)
        return None


def compute_drift(entries: list[Entry]) -> list[str]:
    with ThreadPoolExecutor(max_workers=10) as pool:
        published = list(pool.map(lambda e: fetch_published_hash(e.name), entries))
    return [
        e.name
        for e, p in zip(entries, published)
        if p is None or effective_hash(e) != p
    ]


def main() -> int:
    print(json.dumps(compute_drift(load_entries())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
