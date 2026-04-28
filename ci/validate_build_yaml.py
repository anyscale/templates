#!/usr/bin/env python3
"""Validate BUILD.yaml: referenced paths exist and GCP images are publicly accessible."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent

ACCEPT = ", ".join([
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
])


def is_gcp_uri(uri):
    host = uri.split("/", 1)[0]
    return host.endswith(".pkg.dev") or host.endswith("gcr.io")


def parse_ref(uri):
    repo, _, tag = uri.partition(":")
    if not tag:
        tag = "latest"
    if "/" not in repo:
        host, repo_path = "registry-1.docker.io", repo
    else:
        host, repo_path = repo.split("/", 1)
    return host, repo_path, tag


def manifest_accessible(uri):
    host, repo, tag = parse_ref(uri)
    url = f"https://{host}/v2/{repo}/manifests/{tag}"
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("Accept", ACCEPT)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200, "ok"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return False, f"network error: {e.reason}"


def validate_entry(entry, idx, errors):
    name = entry.get("name", f"<entry #{idx}>")

    d = entry.get("dir")
    if not d:
        errors.append(f"{name}: missing 'dir'")
    elif not (REPO_ROOT / d).is_dir():
        errors.append(f"{name}: dir does not exist: {d}")

    cc = entry.get("compute_config") or {}
    for cloud in ("GCP", "AWS"):
        path = cc.get(cloud)
        if path and not (REPO_ROOT / path).is_file():
            errors.append(f"{name}: compute_config.{cloud} not found: {path}")

    test = entry.get("test") or {}
    tp = test.get("tests_path")
    if tp and not (REPO_ROOT / tp).is_dir():
        errors.append(f"{name}: test.tests_path not found: {tp}")

    ce = entry.get("cluster_env") or {}
    images = []
    if ce.get("image_uri"):
        images.append(ce["image_uri"])
    byod = ce.get("byod") or {}
    if byod.get("docker_image"):
        images.append(byod["docker_image"])

    for img in images:
        if is_gcp_uri(img):
            ok, msg = manifest_accessible(img)
            if not ok:
                errors.append(f"{name}: GCP image not accessible: {img} ({msg})")


def main():
    build_yaml = REPO_ROOT / "BUILD.yaml"
    with build_yaml.open() as f:
        entries = yaml.safe_load(f)

    if not isinstance(entries, list):
        print("ERROR: BUILD.yaml top-level must be a list", file=sys.stderr)
        return 1

    errors = []
    for idx, entry in enumerate(entries):
        validate_entry(entry, idx, errors)

    if errors:
        print(f"FAIL: {len(errors)} validation error(s):", file=sys.stderr)
        for err in errors:
            print(f"::error::{err}", file=sys.stderr)
        return 1

    print(f"OK: validated {len(entries)} BUILD.yaml entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
