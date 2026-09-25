# Evidence for the README

"Cluster" is the shipped compute config, one `g6.4xlarge` worker (L4 GPU, 16 vCPUs) and an
`m5.2xlarge` head, with torch 2.9.1+cu129 and 96 frames of 16.7 MB per layout in 4 files.

## Cluster runs

Ray 2.57.0 runs were `tests.sh` or `measure_layout.py`; Ray 2.58.0 runs were 3 notebook test runs on
2026-09-24 on `anyscale/ray:2.58.0-py312-cu129`. A/B rates are 1 timed run per arm after the warmup;
spans are from the timed `binary(N)` run's `ds.stats()`.

| | Ray 2.57.0 | Ray 2.58.0 |
|---|---|---|
| Write ratio | 4.07x to 4.19x, 8 runs, spread 2.9% | 4.01x to 4.18x, 3 runs: 12.65-13.39 s against 52.94-54.14 s |
| End to end, rows/s | 15.92-16.81 against 9.94-10.21, 1.56x to 1.65x, 5 runs | 16.19-16.72 against 9.20-9.61, 1.69x to 1.78x, 3 runs |
| Read span, read UDF time | 2.44 s, 0us | 2.39-2.6 s, 0us |
| GPU stage span, UDF across 2 actors, saturation | 1.99 s, 3.4 s, about 0.85 | 1.9-2.06 s, 3.33-3.63 s, 0.81-0.96 |
| First run on a fresh cluster, then the next | 85.1-94.0 s, then 5.8 s and 5.9 s, 3 runs | 89.1-96.8 s, then 5.7-5.9 s, 3 runs |
| A/B with no warmup | 1.07 against 10.07 rows/s, 0.11x | not measured |
| Whole test run, workspace start to teardown | not measured | 8 min 58 s to 9 min 26 s, 3 runs |

On Ray 2.58.0 all 3 end-to-end ratios sat above the 2.57.0 range, since `list<uint8>` ran slower;
1 timed run per arm, cause unmeasured.

## Size on disk

`list<uint8>` is Parquet INT32, 4 bytes per payload byte before encoding; dictionary encoding turns
each byte into an index of about 1 byte before any codec runs. Footer `total_uncompressed_size`
(encoded, before the codec) for one frame, pyarrow 23.0.1:

| | encoded bytes | against `binary(N)` |
|---|---|---|
| `binary(N)` `required`, no dictionary | 16,684,961 | |
| `list<uint8>`, dictionary on, pyarrow's default | 16,719,140 | 1.00x |
| `list<uint8>`, dictionary off | 66,739,776 | 4.00x |

The `--sweep on-disk` grid: 6 codecs, both dictionary settings, 3 synthetic payloads. The encoded
gap was 4.00x in every dictionary-off configuration, and 1.00x with the dictionary on, or 0.50x for
`quantized`, whose 16 byte values pack into 4-bit indices. Row groups of 1, 4 and 16 rows didn't
change it. The gap closes with codec `none` and on incompressible `random` data, so the codec plays
no part. With the dictionary off, zstd shrinks the list column to 0.86x of `binary(N)`; on pyarrow
25.0.1 it was 1.28x, and the encoded gaps held.

## Decode on a laptop

`pq.read_table`, warm cache, 3 timed runs after a warmup per arm, macOS arm64, pyarrow 23.0.1.
Ranges don't overlap. Decompression costs both arms and dominates the fast one, so a ratio measured
under compression understates the decode gap.

| configuration | bytes on disk | `binary(N)`, MB/s | `list<uint8>`, MB/s | ratio |
|---|---|---|---|---|
| no codec, dictionary on | identical | 12,936-13,783 | 1,065-1,384 | 10.94x |
| zstd, dictionary on, as shipped | 1.00x | 6,355-9,356 | 1,102-1,503 | 4.51x |
| no codec, dictionary off | list 4x | 10,459-12,259 | 1,039-1,378 | 8.30x |

## Storage on the cluster worker

`measure_layout.py --sweep storage`: codec `none`, 12 frames, 2 runs per arm, every comparison
non-overlapping. Cold is `fsync` then `posix_fadvise(POSIX_FADV_DONTNEED)`. Cold throughput:
`/mnt/local_storage` 1906.6 MB/s, `/mnt/cluster_storage` 138.2 MB/s, or 141.5 MB/s from the head.

| mount | list dictionary | cold ratio | warm ratio |
|---|---|---|---|
| `/mnt/local_storage` | on | 12.16-22.11x | 31.92-32.33x |
| `/mnt/local_storage` | off | 21.61-21.74x | 31.30-31.55x |
| `/mnt/cluster_storage` | on | 3.17-3.23x | 32.40-32.73x |
| `/mnt/cluster_storage` | off | 4.48-4.72x | 31.16-31.67x |

The shared mount limits `binary(N)`, 3,980 MB/s warm and 296 MB/s cold, while `list<uint8>` is
CPU-bound at 65.3 to 127.5 MB/s of payload in all 8 arms. Not timed: zstd, the full pipeline on
these mounts, and more than 12 frames.

## Source workload

The pipeline shape and levers come from a customer workload whose data is private. It reported these
at production scale, on different hardware and data; none is reproduced here.

| Lever | Reported |
|---|---|
| Layout, read side | 7x to 23x, depending on the pair compared |
| Capping read concurrency | 1.66x to 1.91x with a CPU stage downstream binding; 0.56x on a read-bound pipeline |
| Decoder threads | up to 2.15x on local disk, about 1.6x from object storage, +52% wall clock at 8 threads |
| GPU packing | 8 actors at 0.5 GPU each, on 4 of 8 GPUs |
| Object-store fraction | 0.6, peak 69.4 GiB, no spill |
| Operator spans | summed to 2.1x the pipeline's wall clock |
