#!/usr/bin/env python3
"""Grep a notebook's markdown and code comments for the mechanical voice tells.

Usage: python prose_lint.py <notebook.ipynb> [...]

Catches the recurring, greppable violations from notebook-authoring.md. It cannot
judge openers or altitude — the written opener audit still runs by hand — but it
catches the patterns that keep leaking back. Zero exit = no hits.
"""
import json
import re
import sys

RULES = [
    # (name, regex, applies_to)  applies_to: md, comment, or both
    ("animate-verb", re.compile(
        r"\b(live[s]? in|rides? along|sits? (in|on|at)|carries|carry\b|owns\b|journey|"
        r"lands? (as|in|on)|comes? home|breathes?|wants? to)\b", re.I), "both"),
    ("banned-term", re.compile(
        r"\b(corpus|smoke test|smoke run|de-facto|payoff|skeeze|full stop)\b", re.I), "md"),
    ("fm-abbrev", re.compile(r"(?<![`/\w])fm(?![`\w])"), "md"),
    ("movie-preview", re.compile(
        r"\b(the one (line|knob|thing)|that's (all|it) it takes|is all it takes|"
        r"the magic|the beauty of|the payoff is)\b", re.I), "both"),
    ("announce-label", re.compile(
        r"^#*\s*(The |A |One |Two |Three |Four |Note:|Important:|Key )\w[\w '\-]{0,30}:\s", ), "comment"),
    ("negative-opener", re.compile(
        r"^(No |Nothing |Never |Not |Neither |Nobody )"), "both"),
    ("because-tail", re.compile(
        r", because [^.]{10,}\.\s*$"), "md"),
    ("notation-as-prose", re.compile(
        r"`[^`]+`\s*\+\s*`[^`]+`"), "md"),
    ("label-bullet", re.compile(
        r"^\s*[-*]\s+\*\*[\w /()]+\*\*\s*:"), "md"),
    ("announced-contrast", re.compile(
        r"\b(with one (big )?difference|but here's the (catch|twist)|the catch is)\b", re.I), "both"),
    ("punctuation-pile", re.compile(
        r"[^.!?]*:[^.!?]*\([^)]*\)[^.!?]*;"), "md"),
    ("grandstand", re.compile(
        r"\b(the artifact every|is what makes .{0,40} possible|this is the moment|"
        r"the heart of|the whole (game|point|story) is)\b", re.I), "md"),
]


def lint_notebook(path):
    hits = []
    nb = json.load(open(path))
    for c in nb.get("cells", []):
        cid = c.get("id", "?")
        src = "".join(c.get("source", []))
        if c.get("cell_type") == "markdown":
            for ln, line in enumerate(src.splitlines(), 1):
                for name, rx, scope in RULES:
                    if scope in ("md", "both") and rx.search(line):
                        hits.append((path, cid, f"md:{ln}", name, line.strip()[:90]))
        elif c.get("cell_type") == "code":
            for ln, line in enumerate(src.splitlines(), 1):
                if "#" not in line:
                    continue
                comment = line[line.index("#"):]
                for name, rx, scope in RULES:
                    if scope in ("comment", "both") and rx.search(comment):
                        hits.append((path, cid, f"code:{ln}", name, comment.strip()[:90]))
    return hits


def main(paths):
    all_hits = []
    for p in paths:
        all_hits += lint_notebook(p)
    for path, cid, loc, name, text in all_hits:
        print(f"{path}  cell={cid}  {loc}  [{name}]  {text}")
    print(f"{len(all_hits)} hit(s)")
    return 1 if all_hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
