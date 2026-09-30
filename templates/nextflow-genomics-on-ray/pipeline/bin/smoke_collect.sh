#!/usr/bin/env bash
# smoke.nf's gather. A bin/ script so the smoke run checks that bin/ reaches the workers.
set -euo pipefail
export LC_ALL=C

cat "$@" | sort > smoke-summary.tsv

# The oracle. Compare this file across dispatch modes; it must not change.
grep '^checksum' smoke-summary.tsv | cut -f2 | sort -n > checksums.txt

echo "shards: $(grep -c '^shard' smoke-summary.tsv)"
echo "distinct hosts: $(grep '^hostname' smoke-summary.tsv | cut -f2 | sort -u | wc -l)"
