#!/usr/bin/env bash
#
# Call DeepVariant from its quarantined environment.
#
# DeepVariant pins `python <3.11`; the cluster runs 3.12, and a Ray driver and its
# workers must agree on the interpreter. So it lives in its own conda environment
# at /opt/nf-tools/envs/deepvariant, is never on PATH, and is reached only through
# this script.
#
# The isolation is not decorative. The WDL sibling template carried a tool with
# exactly this constraint (medaka) and ended up shipping it disabled, because the
# alternative was letting a 3.10 interpreter shadow the cluster's. Everything here
# exists to make sure that cannot happen by accident: no PATH edit, no
# PYTHONPATH inheritance, and an assertion that the wrapper really did get the
# interpreter it meant to.
#
#   bash tools/run_deepvariant.sh --model_type=WGS --ref=... --reads=... --output_vcf=...
set -euo pipefail

DV_PREFIX="${NF_RAY_DEEPVARIANT_PREFIX:-/opt/nf-tools/envs/deepvariant}"
DV_PYTHON="$DV_PREFIX/bin/python"
DV_RUN="$DV_PREFIX/bin/run_deepvariant"

if [[ ! -x "$DV_PYTHON" ]]; then
    echo "run_deepvariant.sh: no DeepVariant environment at $DV_PREFIX" >&2
    echo "  Built by tools/build_tools.sh from tools/env.deepvariant.yml." >&2
    exit 127
fi

# Do not inherit the caller's Python environment. A PYTHONPATH or PYTHONHOME set
# for the cluster's 3.12 would be read by this 3.10 interpreter and produce import
# errors that name neither interpreter.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP

# Confirm the quarantine actually holds, rather than trusting the layout.
"$DV_PYTHON" - <<'PY'
import sys
major, minor = sys.version_info[:2]
assert (major, minor) == (3, 10), f"DeepVariant env is python {major}.{minor}, expected 3.10"
PY

exec "$DV_RUN" "$@"
