#!/usr/bin/env bash
# Report what a template's lock changes about its image.
#
# Every line printed is a package pip overwrites in the running image. Each one
# should be traceable to the template's requirements.txt — an unexplained entry
# is drift.
#
# Usage: lock-vs-image.sh [template-name ...]     (no args = every template)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATES=("$@")
TEMPLATES="${TEMPLATES[*]:-}" ROOT="$ROOT" python3 - <<'PY'
import os, pathlib, re, sys, yaml

root = pathlib.Path(os.environ["ROOT"])
wanted = set(os.environ["TEMPLATES"].split())
config = yaml.safe_load((root / "dependencies/template.depsets.yaml").read_text())
arg_sets = config["build_arg_sets"]

pin = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)")
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()


def pins(path):
    out = {}
    for line in path.read_text().splitlines():
        if m := pin.match(line):
            out[norm(m.group(1))] = m.group(2)
    return out


def expand(text, args):
    for key, value in args.items():
        text = text.replace("${%s}" % key, value)
    return text


rc = 0
for entry in config["depsets"]:
    seed = next((h for h in entry.get("pre_hooks", []) if "seed-image-freeze.sh" in h), None)
    if not seed:
        continue
    name = entry["output"].split("/")[1]
    if wanted and name not in wanted:
        continue
    for arg_set in entry["build_arg_sets"]:
        args = arg_sets[arg_set]
        freeze = root / expand(seed.split()[1], args)
        lock = root / expand(entry["output"], args)
        if not freeze.exists():
            print(f"{name}: MISSING FREEZE {freeze.relative_to(root)}", file=sys.stderr)
            rc = 1
            continue
        image, locked = pins(freeze), pins(lock)
        diff = sorted((p, image[p], v) for p, v in locked.items() if p in image and image[p] != v)
        added = sorted(p for p in locked if p not in image)
        print(f"\n=== {name}  ({freeze.stem.removesuffix('.freeze')})")
        print(f"    {len(diff)} overwritten, {len(added)} added, {len(locked)} locked")
        for p, was, now in diff:
            print(f"    ! {p}: image {was} -> lock {now}")
sys.exit(rc)
PY
