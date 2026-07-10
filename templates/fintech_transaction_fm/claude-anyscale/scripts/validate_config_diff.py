#!/usr/bin/env python3
"""
validate_config_diff.py — SAFETY GATE for the scale-to-zero enforce.

Asserts the ONLY difference between a workspace's CURRENT compute config and a PROPOSED
config is min_nodes value(s) on worker group(s). Prints every difference and exits 0 (pass)
or 1 (fail). The enforce prompt REQUIRES running this and seeing PASS before it
terminates / updates / starts anything.

Usage: python3 validate_config_diff.py <workspace_id> <proposed_config.yaml|json>
"""
import json, sys, subprocess


def load(path):
    txt = open(path).read()
    try:
        return json.loads(txt)              # our proposed files are JSON (valid YAML)
    except json.JSONDecodeError:
        import yaml
        return yaml.safe_load(txt)


def live_config(wsid):
    r = subprocess.run(["anyscale", "workspace_v2", "get", "--id", wsid, "--json", "--verbose"],
                       capture_output=True, text=True)
    return (json.loads(r.stdout or "{}").get("config") or {}).get("compute_config") or {}


def walk(a, b, path=""):
    """Yield (path, a_val, b_val) for every differing leaf. Lists of dicts matched by 'name'."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            yield from walk(a.get(k, "<absent>"), b.get(k, "<absent>"),
                            f"{path}.{k}" if path else k)
    elif isinstance(a, list) and isinstance(b, list) and \
            all(isinstance(x, dict) and "name" in x for x in a + b):
        an = {x["name"]: x for x in a}
        bn = {x["name"]: x for x in b}
        for name in sorted(set(an) | set(bn)):
            yield from walk(an.get(name, "<absent>"), bn.get(name, "<absent>"), f"{path}[{name}]")
    elif isinstance(a, list) and isinstance(b, list):
        for i in range(max(len(a), len(b))):
            av = a[i] if i < len(a) else "<absent>"
            bv = b[i] if i < len(b) else "<absent>"
            yield from walk(av, bv, f"{path}[{i}]")
    elif a != b:
        yield (path, a, b)


def main():
    if len(sys.argv) != 3:
        print("usage: validate_config_diff.py <workspace_id> <proposed.yaml|json>")
        sys.exit(2)
    wsid, proposed_path = sys.argv[1], sys.argv[2]
    current, proposed = live_config(wsid), load(proposed_path)

    all_diffs = list(walk(current, proposed))
    min_node = [d for d in all_diffs if d[0].endswith(".min_nodes")]
    other = [d for d in all_diffs if not d[0].endswith(".min_nodes")]

    print(f"=== config diff: workspace {wsid} vs {proposed_path} ===")
    for p, a, b in min_node:
        print(f"  OK   {p}: {a} -> {b}")
    for p, a, b in other:
        print(f"  BAD  {p}: {a} -> {b}")

    if other:
        print(f"\nFAIL: {len(other)} field(s) other than min_nodes differ. Refusing to apply.")
        sys.exit(1)
    if not min_node:
        print("\nFAIL: no min_nodes change at all — nothing to enforce.")
        sys.exit(1)
    print(f"\nPASS: only min_nodes changed ({len(min_node)} group(s)). Safe to apply.")
    sys.exit(0)


if __name__ == "__main__":
    main()
