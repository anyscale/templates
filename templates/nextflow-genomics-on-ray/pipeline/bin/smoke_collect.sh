#!/usr/bin/env bash
#
# smoke.nf's gather step: shard reports in, the summary and the checksum oracle out.
#
#   smoke_collect.sh shard-0.tsv shard-1.tsv ...
#
# A bin/ script rather than inline shell so that the smoke run checks one more thing
# the executor owns: that the pipeline's bin/ reaches the node a task lands on. See
# COLLECT in smoke.nf.
set -euo pipefail
export LC_ALL=C

cat "$@" | sort > smoke-summary.tsv

# The oracle. Compare this file across dispatch modes; it must not change.
grep '^checksum' smoke-summary.tsv | cut -f2 | sort -n > checksums.txt

echo "shards: $(grep -c '^shard' smoke-summary.tsv)"
echo "distinct hosts: $(grep '^hostname' smoke-summary.tsv | cut -f2 | sort -u | wc -l)"
