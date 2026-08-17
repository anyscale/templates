#!/usr/bin/env bash
# Seed a lock with resolution preferences before compiling.
#
# uv reads the -o file as preferences: a listed package keeps its version unless a
# requirement forces otherwise. Seed it with the image freeze, then carry over the
# previous lock's pins for packages the image does NOT ship.
#
#   image packages  -> the image's version
#   everything else -> its previously locked version
#
# Without the carry-over, regenerating re-resolves every non-image package to whatever
# is newest that day — the drift this whole setup exists to prevent.
#
# Usage: seed-image-freeze.sh <freeze-file> <lock-output-path>
set -euo pipefail

FREEZE="${1:?Usage: seed-image-freeze.sh <freeze-file> <lock-output-path>}"
OUTPUT="${2:?Usage: seed-image-freeze.sh <freeze-file> <lock-output-path>}"

[ -f "$FREEZE" ] || { echo "Image freeze not found: $FREEZE" >&2; exit 1; }

mkdir -p "$(dirname "$OUTPUT")"
FREEZE="$FREEZE" OUTPUT="$OUTPUT" python3 - <<'PY'
import os, re, pathlib

freeze = pathlib.Path(os.environ["FREEZE"])
output = pathlib.Path(os.environ["OUTPUT"])
# `[extras]` are tolerated so a stanza is never mistaken for an unnamed line and
# silently dropped; `@` catches the direct-URL form pip freeze uses for VCS installs.
pin = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*(?:==|@)")
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()

lines = freeze.read_text().splitlines()
have = {norm(m.group(1)) for l in lines if (m := pin.match(l))}
if len(have) < 50:
    raise SystemExit(
        f"{freeze} lists {len(have)} packages; a real image has hundreds. "
        "Refusing to seed from a truncated or corrupt freeze."
    )

carried = 0
if output.exists():
    # keep each stanza (pin line + its indented --hash/# via continuations) whose
    # package the freeze doesn't already cover
    keep = False
    for line in output.read_text().splitlines():
        # a blank line continues the stanza: treating it as a new header would cut
        # the remaining --hash lines and leave a dangling continuation backslash
        if line.strip() and line[:1] not in (" ", "\t"):
            m = pin.match(line)
            keep = bool(m) and norm(m.group(1)) not in have
            carried += keep
        if keep:
            lines.append(line)

output.write_text("\n".join(lines) + "\n")
print(f"Seeded {output} with {len(have)} image packages from {freeze.name}"
      + (f" + {carried} carried from the previous lock" if carried else ""))
PY
