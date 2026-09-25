# Measurements behind the README

The README gives each figure with its scope. This file keeps the runs and tables behind them.

## Co-resident vs serial

One L4, this template's models, 96 frames at 640x480, one actor per stage. Each figure is
end-to-end wall clock, including model loads from a warm cache. `measure_packing.py` warmed the
Hugging Face cache on the GPU worker, discarded one untimed pass of each arm, then interleaved
the timed runs.

| arm | rows/s, 3 timed runs | spread |
|---|---|---|
| co-resident: four stages sharing the card | 1.2619 / 1.2631 / 1.2792 | 1.4% |
| serial: each stage alone with the whole GPU, materialized between stages | 0.9882 / 0.9903 / 0.9979 | 1.0% |

Two arms count as separable when the worst run of the faster arm beats the best run of the
slower one (1.2619 > 0.9979), and the margin is that gap, a lower bound: 26.5%. Ranges that
overlap are not separable however far apart the means sit, and an arm needs at least two runs.
`tests/test_packing.py` (class `Separability`) holds these runs and checks the rule on them.

This was the third of three rounds, and it excluded nothing. The first two read at least 27.7%
and 27.9%, but only after a slow run was dropped post hoc; on their full data neither was
separable.

## VRAM per actor

`python packing.py` prints the shipped configuration: `pipeline.py`'s default fractions and
actor counts, with per-actor VRAM measured on the production workload's own models and 48 GiB
card.

| stage | `num_gpus` | GiB/actor | by fraction | by VRAM | binds |
|---|---|---|---|---|---|
| detector | 0.20 | 9.60, unmeasured | 5 | 5 | both |
| object embedder | 0.05 | 7.65 | 20 | 6 | VRAM |
| image embedder | 0.05 | 0.96 | 20 | 50 | fraction |
| metrics | 0.02 | 2.80 | 50 | 17 | VRAM |

The detector's 9.60 is its fraction share of the card, not a measurement, and `packing.py`
prints `(UNMEASURED)` beside it. The two embedders take the same fraction and differ 8x in
memory. One actor of each stage holds 21.01 GiB: 56% headroom on 48 GiB, but 5% on an L4, where
`python packing.py --vram 22.03` also reports the shipped actor counts over-committed.
`packing.py` flags headroom under 10%, the band where the production workload hit its batch-192
OOM.

`--stages measured` is this template's SAM 3, DINOv3 and SigLIP2, from two probe jobs on a g6 L4
on 2026-08-17: 640x480 frames, bf16, no `torch.compile`, peak allocated above the pre-stage
baseline. One actor of each holds 5.53 GiB, 75% headroom. An L4's 24 GB is 22.35 GiB, of which
torch reports 22.03 usable. The object embedder's 0.59 GiB was measured at one crop per frame, a
floor for a nearly empty batch rather than a budget for a busy one. `packing.py` skips its
cost-ordering check on this table, though its last line still mentions the ordering.

## Detector labels

Boxes per frame on `make_fixture.py`'s frames, on an L4 at the shipped threshold of 0.4:

| prompt | boxes per frame |
|---|---|
| `object` / `shape` / `region` | 0 0 0 0 |
| `rectangle` | 1 0 0 3 |
| `bright square` | 1 1 1 2 |

`object,shape,region` finds nothing at any threshold down to 0.05, while the same weights, GPU
and threshold find 2 cats, 2 remote controls and 1 couch in a COCO photograph.

## The base image

The cells before the install need only numpy, pyarrow and Ray. The base image ships no torch,
torchvision or transformers: measured on the 2.57.0 image, and the 2.58.0 package freeze lists
the same.

## Where the defaults come from

`pipeline.py`'s defaults are the production workload's best full run (2026-06-09), on its own
models and fleet. Its end-to-end figure, ~49.7 rows/s on a fixed 131k x 2 confirmation run, does
not compare with this template's rows/s: a row there was not a frame (39.65 rows/s was 9.9
images/s). The comment above each setting in `pipeline.py` gives that workload's measured
effect.
