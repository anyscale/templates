#!/usr/bin/env bash
#
# Call DeepVariant from its own environment, off PATH.
#
# DeepVariant pins `python <3.11`; the cluster runs 3.12, and a Ray driver and its
# workers must agree on the interpreter. So a DeepVariant install has to live in
# its own environment at $NF_RAY_DEEPVARIANT_PREFIX (default
# /opt/nf-tools/envs/deepvariant), never on PATH, reached only through this
# script.
#
# This image has no such install. bioconda's deepvariant 1.10.0 (pyh697b589_0) was
# the plan, and it cannot work: the package is three dv_*.py wrappers pointing at
# the literal placeholder `$PREFIX/BINARYSUB`, because the recipe's steps that
# install Google's binaries and models are commented out. No `run_deepvariant`,
# no make_examples.zip, no model checkpoint. So today this script explains that
# and exits 127, and main.nf refuses `--deepvariant` at startup for the same
# reason.
#
# To plug a real install in, put Google's DeepVariant 1.10.0 into a python 3.10
# environment so that $NF_RAY_DEEPVARIANT_PREFIX/bin/run_deepvariant exists. The
# isolation below still applies: no PATH edit, no PYTHONPATH inheritance, and an
# assertion that the interpreter really is the one it meant to get.
#
#   bash tools/run_deepvariant.sh --model_type=WGS --ref=... --reads=... --output_vcf=...
set -euo pipefail

DV_PREFIX="${NF_RAY_DEEPVARIANT_PREFIX:-/opt/nf-tools/envs/deepvariant}"
DV_PYTHON="$DV_PREFIX/bin/python"
DV_RUN="$DV_PREFIX/bin/run_deepvariant"

if [[ ! -x "$DV_PYTHON" || ! -x "$DV_RUN" ]]; then
    echo "run_deepvariant.sh: no DeepVariant at $DV_PREFIX (need bin/python and bin/run_deepvariant)" >&2
    echo "  This image does not include one: bioconda's deepvariant 1.10.0 ships wrappers" >&2
    echo "  but not the binaries or models they call. See pipeline/PIPELINE.md." >&2
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
