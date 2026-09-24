#!/usr/bin/env bash
set -euxo pipefail

# Pinned: an unpinned test dependency makes "the template broke" and "papermill changed"
# the same red build, and only one of those is worth waking up for.
uv pip install -q --system 'papermill==2.7.0'

# The notebook's default is `standard`, three samples x 10 Mbp of chr20, which is what a
# reader gets and is deliberately more than CI should spend. CI runs `quick`, 2 Mbp: same
# pipeline, same tools, same resource requests, same cohort shape, only fewer bases. The
# Buildkite step is capped at max(75, timeout_in_sec/60 + 30) minutes, 90 for this template.
export NF_DEMO_SCALE=quick

# The GPU leg stays in the template and out of CI: ANNOTATE_VARIANTS asks for an L4, and a run
# that waits on L4 capacity in the test cloud's zone is red for a reason the template cannot
# fix. The notebook reads this and skips the process and its Ray Data step (Step 7).
export NF_ANNOTATE=false

# Offline first, and deliberately before anything expensive. Plain scripts, not pytest
# (pytest collects none of them), each exiting non-zero on a failed check:
#
#   test_nf_ray            the #RAY header, directive-to-Ray mapping, exit codes, config
#   test_daemon            the daemon's task table, against a stand-in ray module
#   test_config_agreement  the node ceiling, scale regions and plugin version, which live in
#                          several files each (its compute-config checks skip here: rayapp
#                          does not ship configs/ to the cluster)
#   test_make_samplesheet  per-sample truth wiring, and a manifest that disagrees with disk
#   test_collect_vcfeval   the vcfeval readout, against summaries captured from rtg itself
#   test_score_variants    the reference and VCF plumbing around the scorer's model
#
# rayapp flattens templates/<name>/ and tests/<name>/ into one directory, so these sit beside
# README.ipynb, and each finds nf_ray/ and pipeline/ from the working directory.
here="$(dirname "$0")"
for check in test_nf_ray test_daemon test_config_agreement test_make_samplesheet \
             test_collect_vcfeval test_score_variants; do
  python "$here/$check.py"
done

papermill README.ipynb /tmp/nextflow-genomics-on-ray.out.ipynb --log-output --kernel python3 --cwd .
