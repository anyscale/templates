#!/usr/bin/env bash
# CI test for ray-data-sensor-frame-extraction.
#
# Runs the notebook top to bottom, the way a user does.
#
# WHAT THIS GUARDS, and why it is two directions rather than one ratio.
#
# The template's claim is no longer "the read binds and the layout wins an order of magnitude" --
# its own fleet refuted the magnitude and the binding stage, and the notebook says so up front.
# What survived measurement is:
#
#   1. the WRITE side wins, ~4.1x, replicated (4.08x / 4.13x / 4.19x, within 3%). This is the
#      headline, and the fixture cell asserts its DIRECTION.
#   2. the READ side wins, ~1.6x rather than 7x-23x, because compression closes most of the gap.
#      The A/B cell asserts that direction too, with the small expected margin spelled out.
#
# Both assertions are on the SIGN, never on a threshold: a number here would be fleet-dependent
# and would go stale, and a CI test guarding a magnitude the template no longer claims is worse
# than no assertion at all. An inversion in either direction means the fixture, the codec or the
# hardware has changed enough that the lesson needs re-measuring -- which is exactly when a
# reader should be stopped.
#
# Magnitudes come from measure_layout.py, which takes replicates and is not run here: CI runs
# each arm once, and one run cannot separate a real effect from a cold cache.
set -euo pipefail

uv pip install -q --system papermill

# The fixture is ~16.7 MB a row by design -- that is the regime the lesson lives in. Shrink the
# frame COUNT for CI, never the geometry.
#
# Exported rather than inlined because the notebook reads the same variables for an interactive
# run: this is the template's configuration, not test scaffolding.
export FIXTURE_DIR="${FIXTURE_DIR:-/mnt/cluster_storage/sensor-fixture}"
export FRAMES="${FRAMES:-24}"
export FILES="${FILES:-4}"

papermill README.ipynb /tmp/ray-data-sensor-frame-extraction.out.ipynb \
    --log-output --kernel python3 --cwd .
