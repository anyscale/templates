#!/usr/bin/env python3
"""Fan out the "Template update" Cursor automation to bump templates to a Ray version.

One POST to the automation's webhook -> one agent -> one template -> one draft PR.
The agent prompt lives on the Cursor automation dashboard (single source of truth);
this script only fans a {template_name, ray_version} payload out over the maintained
BUILD.yaml entries. The automation runs .claude/skills/template/workflows/bump-ray-version.md.

Credentials come from the environment — never hardcoded, never committed (needed only to --execute):
  CURSOR_TEMPLATE_UPDATER_WEBHOOK     the automation's inbound webhook URL (the POST target).
  CURSOR_TEMPLATE_UPDATER_AUTH_TOKEN  Bearer for the webhook — the automation's
                                      "Generate auth header" value.

Template selection: pass names explicitly, or --all for every *maintained* BUILD.yaml
entry. `maintained: false` entries (archived templates) are always skipped; a named
entry that is unmaintained or absent from BUILD.yaml is skipped with a warning.

SAFE BY DEFAULT: without --execute the script only PREVIEWS (prints the payloads it
would POST, makes zero API calls). Firing real agents requires an explicit --execute.

Examples:
  # Preview (no --execute -> no POST, no creds needed):
  ci/trigger-cursor-bump.py -v 2.56.0 job-intro object-detection-video-processing skyrl
  ci/trigger-cursor-bump.py --all --exclude job-intro,object-detection-video-processing,skyrl --list
  # Test batch, then fanout -- add --execute to actually fire:
  ci/trigger-cursor-bump.py -v 2.56.0 job-intro object-detection-video-processing skyrl --execute
  ci/trigger-cursor-bump.py -v 2.56.0 --all --exclude job-intro,object-detection-video-processing,skyrl --execute
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

BUILD_YAML = Path(__file__).resolve().parent.parent / "BUILD.yaml"


def warn(msg: str) -> None:
    print(msg, file=sys.stderr)


def resolve_templates(
    entries: list[dict], requested: list[str], exclude: set[str]
) -> list[str]:
    """Filter `requested` down to the launch set: drop excluded, unknown,
    unmaintained, and duplicate names (each skip is reported to stderr)."""
    by_name = {e["name"]: e for e in entries}
    final: list[str] = []
    for name in requested:
        if name in exclude:
            warn(f"skip (excluded): {name}")
        elif name not in by_name:
            warn(f"skip (not in BUILD.yaml): {name}")
        elif not by_name[name].get("maintained", True):
            warn(f"skip (maintained: false): {name}")
        elif name not in final:
            final.append(name)
    return final


def trigger(url: str, token: str, payload: dict) -> tuple[bool, str]:
    """POST one payload to the automation webhook. Returns (ok, one-line result).
    Never raises — a single failure must not abort the batch."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status, body = resp.status, resp.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return False, f"  unexpected response: HTTP {e.code} {e.read().decode(errors='replace')[:300]}"
    except (urllib.error.URLError, TimeoutError) as e:
        return False, f"  unexpected response: {e}"
    # Best-effort: surface an id/url if the response is JSON (shape isn't documented).
    detail = ""
    try:
        data = json.loads(body)
        if isinstance(data, dict):
            ident = data.get("id") or data.get("agentId") or data.get("runId") or "?"
            link = data.get("url") or (data.get("target") or {}).get("url")
            detail = f"  id: {ident}" + (f"   url: {link}" if link else "")
    except json.JSONDecodeError:
        detail = f"  response: {body[:200]}" if body.strip() else ""
    return True, f"  triggered (HTTP {status}){detail}"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("templates", nargs="*", help="template names to bump")
    p.add_argument("-v", "--ray-version", default="2.56.0")
    p.add_argument("--all", dest="all_", action="store_true",
                   help="bump every maintained BUILD.yaml entry")
    p.add_argument("--exclude", default="", help="comma-separated names to skip")
    p.add_argument("--list", dest="list_only", action="store_true",
                   help="print the resolved template set and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="print payloads without firing (implied unless --execute)")
    p.add_argument("--execute", action="store_true",
                   help="actually POST to the automation webhook (required to fire)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not BUILD_YAML.is_file():
        warn(f"error: BUILD.yaml not found at {BUILD_YAML}")
        return 2
    entries = yaml.safe_load(BUILD_YAML.read_text())
    maintained = [e["name"] for e in entries if e.get("maintained", True)]

    # Choose the requested set: --all (every maintained entry) or explicit names.
    if args.all_:
        if args.templates:
            warn("error: pass template names OR --all, not both")
            return 2
        requested = maintained
    elif args.templates:
        requested = args.templates
    else:
        warn("error: pass at least one template name, or --all")
        return 2

    exclude = {name for name in args.exclude.split(",") if name}
    final = resolve_templates(entries, requested, exclude)
    if not final:
        warn("error: no templates to fire after filtering")
        return 2

    if args.list_only:
        warn(f"resolved {len(final)} template(s):")
        print("\n".join(final))
        return 0

    # SAFE BY DEFAULT: only --execute performs real POSTs; anything else previews.
    # An explicit --dry-run also wins, so a stray --execute can't override it.
    preview = args.dry_run or not args.execute
    if preview and not args.dry_run:
        warn("note: preview only — nothing fired. Re-run with --execute to fire for real.")

    url = os.environ.get("CURSOR_TEMPLATE_UPDATER_WEBHOOK", "")
    token = os.environ.get("CURSOR_TEMPLATE_UPDATER_AUTH_TOKEN", "")
    if not preview and not (url and token):
        warn("error: set CURSOR_TEMPLATE_UPDATER_WEBHOOK and "
             "CURSOR_TEMPLATE_UPDATER_AUTH_TOKEN to fire")
        return 2

    warn(f"== {'PREVIEW — ' if preview else ''}firing {len(final)} agent(s) via the "
         f"Template-update automation: Ray {args.ray_version} ==")

    fired = 0
    for template in final:
        payload = {"template_name": template, "ray_version": args.ray_version}
        if preview:
            warn(f"-- preview: {template} --")
            print(json.dumps(payload, indent=2))  # payload holds no secrets
            continue
        warn(f"-- firing: {template} (Ray {args.ray_version}) --")
        ok, line = trigger(url, token, payload)
        print(line)
        fired += ok

    if preview:
        warn(f"== preview complete: {len(final)} payload(s) ==")
    else:
        warn(f"== fired {fired}/{len(final)} agent(s) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
