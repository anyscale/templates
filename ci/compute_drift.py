#!/usr/bin/env python3
"""Compute the drifted-template list for the current HEAD.

Compares each template's `effective_hash` against `tmpl_effective_hash` in
`https://templates.ci.ray.io/templates/<name>/latest/channel.json`. Emits a
JSON array of drifted names to stdout."""

from __future__ import annotations

import json
import sys
from typing import Optional

import requests

from compute_effective_hash import effective_hash, load_entries
from validate_build_yaml import Entry


CHANNEL_URL = "https://templates.ci.ray.io/templates/{name}/latest/channel.json"


def fetch_published_hash(name: str, *, timeout: float = 10.0) -> Optional[str]:
    """Returns the published `tmpl_effective_hash` for `name`, or None if
    the template has never published (404) or channel.json predates the
    field (returned by product PR 39637 — treated as drifted)."""
    resp = requests.get(CHANNEL_URL.format(name=name), timeout=timeout)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json().get("tmpl_effective_hash")


def compute_drift(entries: list[Entry]) -> list[str]:
    drifted: list[str] = []
    for e in entries:
        current = effective_hash(e)
        published = fetch_published_hash(e.name)
        if published is None or current != published:
            drifted.append(e.name)
    return drifted


def main() -> int:
    entries = load_entries()
    drifted = compute_drift(entries)
    print(json.dumps(drifted))
    return 0


if __name__ == "__main__":
    sys.exit(main())
