#!/usr/bin/env python3
"""Emit a JSON array of drifted template names to stdout.

Drift = current effective_hash differs from `tmpl_effective_hash` in
templates.ci.ray.io/templates/<name>/latest/channel.json (404 or
missing field = drifted)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from effective_hash import effective_hash, load_entries  # noqa: E402
from validate_build_yaml import Entry  # noqa: E402


CHANNEL_URL = "https://templates.ci.ray.io/templates/{name}/latest/channel.json"


def fetch_published_hash(name: str, *, timeout: float = 10.0) -> Optional[str]:
    """None on first-publish (404) or if channel.json lacks the field."""
    resp = requests.get(CHANNEL_URL.format(name=name), timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json().get("tmpl_effective_hash")


def compute_drift(entries: list[Entry]) -> list[str]:
    drifted: list[str] = []
    for e in entries:
        published = fetch_published_hash(e.name)
        if published is None or effective_hash(e) != published:
            drifted.append(e.name)
    return drifted


def main() -> int:
    print(json.dumps(compute_drift(load_entries())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
