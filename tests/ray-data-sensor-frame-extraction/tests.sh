#!/usr/bin/env bash
# CI test for ray-data-sensor-frame-extraction.
#
# Runs the notebook top to bottom, the way a user does.
#
# WHAT THIS GUARDS: two directions, no ratio.
#
# The template's claim is no longer "the read binds and the layout wins an order of magnitude" --
# its own fleet refuted the magnitude and the binding stage, and the notebook says so up front.
# What survived measurement is:
#
#   1. the WRITE side wins, ~4.1x, replicated across fleet runs; the range and run count are in
#      the README's results table. This is the headline, and the fixture cell asserts its
#      DIRECTION.
#   2. the READ side wins END TO END, ~1.6x. The A/B cell asserts that direction too. Note the
#      channel: Arrow's DECODE of the blob column differs by ~11x from byte-identical inputs, and
#      what shrinks it to 1.6x here is Amdahl -- this pipeline has a GPU stage. The on-disk gap is
#      1.00x under pyarrow's defaults because dictionary encoding, not the codec, removes the
#      per-value bookkeeping; `measure_layout.py --sweep on-disk` prints all 36 cells.
#
# Both assertions are on the SIGN, never on a threshold: a number here would be fleet-dependent
# and would go stale, and a CI test guarding a magnitude the template no longer claims is worse
# than no assertion at all. An inversion in either direction means the fixture, the codec or the
# hardware has changed enough that the figures need re-measuring.
#
# Magnitudes come from measure_layout.py, which takes replicates and is not run here: CI runs
# each arm once, and one run cannot separate a real effect from a cold cache.
set -euo pipefail

uv pip install -q --system papermill

# The fixture is ~16.7 MB a row by design -- that is the regime the lesson lives in. Shrink the
# frame COUNT for CI, never the geometry.
#
# Exported, not inlined: the notebook reads the same variables for an interactive run.
export FIXTURE_DIR="${FIXTURE_DIR:-/mnt/cluster_storage/sensor-fixture}"
export FRAMES="${FRAMES:-24}"
export FILES="${FILES:-4}"

papermill README.ipynb /tmp/ray-data-sensor-frame-extraction.out.ipynb \
    --log-output --kernel python3 --cwd .
