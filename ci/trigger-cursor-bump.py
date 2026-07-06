#!/usr/bin/env python3
"""Trigger Cursor Cloud "template-updater" agents to bump templates to a Ray version.

One agent -> one template -> one draft PR (each runs the non-interactive workflow
.claude/skills/template/workflows/bump-ray-version.md, per AGENTS.md "Cursor Cloud").

Credentials come from the environment — never hardcoded, never committed:
  CURSOR_TEMPLATE_UPDATER_AUTH_TOKEN      Cursor API key, sent as Bearer (falls back to
                                          CURSOR_API_KEY). Needed only to --execute.
  CURSOR_TEMPLATE_UPDATER_WEBHOOK         optional  https URL for statusChange callbacks.
  CURSOR_TEMPLATE_UPDATER_WEBHOOK_SECRET  optional  shared secret for the signature HMAC.
  CURSOR_AGENTS_URL                       optional  default https://api.cursor.com/v0/agents.
  TEMPLATES_REPO                          optional  default https://github.com/anyscale/templates.

Template selection: pass names explicitly, or --all for every *maintained* BUILD.yaml
entry. `maintained: false` entries (archived templates) are always skipped; a named
entry that is unmaintained or absent from BUILD.yaml is skipped with a warning.

SAFE BY DEFAULT: without --execute the script only PREVIEWS (prints the payloads it
would POST, makes zero API calls). Launching real agents requires an explicit --execute.

Examples:
  # Preview (no --execute -> no launch, no creds needed):
  ci/trigger-cursor-bump.py -v 2.56.0 job-intro object-detection-video-processing skyrl
  ci/trigger-cursor-bump.py --all --exclude job-intro,object-detection-video-processing,skyrl --list
  # Test batch, then fanout -- add --execute to actually launch:
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
DEFAULT_AGENTS_URL = "https://api.cursor.com/v0/agents"  # v0 supports webhooks
DEFAULT_REPO = "https://github.com/anyscale/templates"


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


def build_payload(
    template: str, ray_version: str, ref: str, repo: str,
    webhook_url: str, webhook_secret: str,
) -> dict:
    prompt = (
        f"You are the non-interactive `template-updater` Cursor Cloud agent. "
        f"Run the Ray-version bump workflow at "
        f".claude/skills/template/workflows/bump-ray-version.md for template "
        f"'{template}', target Ray version {ray_version}. Follow AGENTS.md "
        f"'Cursor Cloud' (preflight, cursor/ branch naming, "
        f"GH_TOKEN=$ANYSCALE_GH_TOKEN on gh writes, ray-update + cursor-cloud PR "
        f"labels). One template, one draft PR. Do not read or act on PR comments. "
        f"Stop-and-report on any blocked precondition."
    )
    body: dict = {
        "prompt": {"text": prompt},
        "source": {"repository": repo, "ref": ref},
    }
    if webhook_url:
        body["webhook"] = {"url": webhook_url}
        if webhook_secret:
            body["webhook"]["secret"] = webhook_secret
    return body


def launch(url: str, token: str, payload: dict) -> tuple[bool, str]:
    """POST one agent. Returns (ok, one-line result). Never raises — a single
    failure must not abort the batch."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        return False, f"  unexpected response: HTTP {e.code} {e.read().decode(errors='replace')[:500]}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return False, f"  unexpected response: {e}"
    target = data.get("target") or {}
    return True, (
        f"  agent: {data.get('id', '?')}   status: {data.get('status', '?')}   "
        f"url: {target.get('url') or data.get('url', '?')}"
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("templates", nargs="*", help="template names to bump")
    p.add_argument("-v", "--ray-version", default="2.56.0")
    p.add_argument("-r", "--ref", default="main")
    p.add_argument("--all", dest="all_", action="store_true",
                   help="bump every maintained BUILD.yaml entry")
    p.add_argument("--exclude", default="", help="comma-separated names to skip")
    p.add_argument("--list", dest="list_only", action="store_true",
                   help="print the resolved template set and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="print payloads without launching (implied unless --execute)")
    p.add_argument("--execute", action="store_true",
                   help="actually POST to the Cursor API (required to launch)")
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
        warn("error: no templates to launch after filtering")
        return 2

    if args.list_only:
        warn(f"resolved {len(final)} template(s):")
        print("\n".join(final))
        return 0

    # SAFE BY DEFAULT: only --execute performs real API calls; anything else previews.
    # An explicit --dry-run also wins, so a stray --execute can't override it.
    preview = args.dry_run or not args.execute
    if preview and not args.dry_run:
        warn("note: preview only — no agents launched. Re-run with --execute to launch for real.")

    token = os.environ.get("CURSOR_TEMPLATE_UPDATER_AUTH_TOKEN") or os.environ.get("CURSOR_API_KEY", "")
    if not preview and not token:
        warn("error: set CURSOR_TEMPLATE_UPDATER_AUTH_TOKEN (or CURSOR_API_KEY) to launch")
        return 2

    webhook_url = os.environ.get("CURSOR_TEMPLATE_UPDATER_WEBHOOK", "")
    webhook_secret = os.environ.get("CURSOR_TEMPLATE_UPDATER_WEBHOOK_SECRET", "")
    repo = os.environ.get("TEMPLATES_REPO", DEFAULT_REPO)
    url = os.environ.get("CURSOR_AGENTS_URL", DEFAULT_AGENTS_URL)

    warn(f"== {'PREVIEW — ' if preview else ''}launching {len(final)} agent(s): "
         f"Ray {args.ray_version}, ref {args.ref} ==")
    warn("   webhook: on" if webhook_url else
         "   webhook: off (set CURSOR_TEMPLATE_UPDATER_WEBHOOK to enable statusChange callbacks)")

    launched = 0
    for template in final:
        if preview:
            # Redact the webhook values so nothing sensitive is printed; stdout
            # stays pure JSON (pipe it to jq). Markers go to stderr.
            warn(f"-- preview: {template} --")
            payload = build_payload(
                template, args.ray_version, args.ref, repo,
                "[redacted CURSOR_TEMPLATE_UPDATER_WEBHOOK]" if webhook_url else "",
                "[redacted CURSOR_TEMPLATE_UPDATER_WEBHOOK_SECRET]" if webhook_secret else "",
            )
            print(json.dumps(payload, indent=2))
            continue
        warn(f"-- launching: {template} (Ray {args.ray_version}) --")
        payload = build_payload(
            template, args.ray_version, args.ref, repo, webhook_url, webhook_secret
        )
        ok, line = launch(url, token, payload)
        print(line)
        if ok:
            launched += 1

    if preview:
        warn(f"== preview complete: {len(final)} payload(s) ==")
    else:
        warn(f"== launched {launched}/{len(final)} agent(s) ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
