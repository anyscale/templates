# Measurement notes

The detail behind the README's numbers. The demo-scale and trio runs used this template's AWS
compute config (`m5.8xlarge` workers, an `m5.2xlarge` head with `CPU: 0`) on the Ray 2.56.0 image.

## Demo scales

Measured as Anyscale Jobs, provisioning and read staging included:

| scale | notebook, end to end | cohort workflow | slowest sample in that run |
|---|---|---|---|
| `quick` | ~10 min | 8m05s | 7m52s |
| `standard` | ~17 min | 14m00s | 13m32s |

The cohort takes 1.03x its slowest sample at both scales, and 1.006x over all of chr20, because
each assembly gets a worker once three are up. In the `quick` run the three `Assemble` tasks queued
32 s, 93 s and 93 s by the backend's `seconds_queued`; the 93 s is the autoscaler adding workers two
and three from `min_nodes: 1`. At `max_nodes: 2` the third assembly queued 7m37s behind the first
and the run took 13m16s. `seconds_queued` stops when the driver sees `ray_placement.json` on shared
storage, so it overstates long waits.

## Full chr20: one queue time

In the trio run (three on-demand workers), HG004's `Assemble` waited about 587 s from dispatch to
Flye's first log line, with all three workers up inside 100 s; `seconds_queued` read 639.1 s. All
three `MeasureDivergence` tasks had landed on the first worker, and HG002's ran 669.8 s. `Assemble`
reserves 30 of a worker's 32 cores, so any 4-core neighbour blocks it. It cost the cohort nothing:
HG004 is the smallest sample and still finished 1410 s before HG002. A fourth node would not have
helped, because HG004's request was passed over twice as the other workers came up.

## Read mode

An earlier single-sample run of HG002 used upstream's `--nano-raw` on the same instance type and
reads (97x, uncapped, Flye's one polishing round):

| HG002, full chr20, `m5.8xlarge` | `--nano-raw` (earlier run) | `--nano-hq` (trio run) |
|---|---|---|
| Flye, total | 14h44m | 1h55m33s |
| polishing stage | 9h51m | 31m15s |
| everything before polishing | 4h53m | 1h24m |
| N50 | 33,279,582 | 33,263,886 |

`--nano-hq` is 7.6x faster for the same N50, most of it in polishing. The coverage cap moves
contiguity instead: on `c6i.16xlarge`, capping at 40x cut the N50 from 33.27 to 11.06 Mbp to save
14.6 minutes. The four-run grid that separates the two levers is in `ONTAssembleWithFlye.wdl`'s
header.

## Contigs and NGA50 on full chr20

From the trio run. HG002 and HG004 hold 90% of the assembly in two contigs, one per arm: 33.26 and
25.78 Mbp for HG002, 33.23 and 25.85 Mbp for HG004. That is 98% of the q arm's 33.9 Mbp of
assemblable sequence and 98% of the p arm's 26.3 Mbp, each contig stopping at the centromere.
HG002's remaining 5.3 Mbp is in 60 contigs averaging 89 kb. HG003 needs three contigs for the same
90%: its p arm comes out whole at 25.78 Mbp and its q arm splits into 16.72 and 16.54 Mbp, while its
genome fraction and NGA50 are in line with the other two.

NGA50 is about 2 Mbp in every full-chr20 run: 1.96-2.19 Mbp across the trio, 1.83-2.08 Mbp in the
`--nano-hq` pair of the four-run grid, and 2.1 Mbp in an unpolished run
(`--nano-raw --iterations 0 --asm-coverage 30 --genome-size 64444167`: N50 33.25 Mbp, genome
fraction 94.9%). Polishing is not what limits it. QUAST breaks aligned blocks at every
rearrangement against GRCh38 over 1 kbp, the sample's own structural variants included.

![The assembly against chromosome 20, one contig per arm stopping at the centromere, and the N50-against-NGA50 gap measured on a separate unpolished run](https://raw.githubusercontent.com/anyscale/templates/main/templates/wdl-genomics-on-ray/assets/chr20-contigs.png)

The first panel is the 14h44m `--nano-raw` run; the current flags reproduce its q-arm contig to
within 0.05% and extend its p-arm contig from 23.65 to 25.78 Mbp. The second is the unpolished run.

## Spot, estimated

None of this is measured: the trio run was entirely on demand, and the retry path has been traced
in code but has not met a real reclaim.

| | |
|---|---|
| Spot Advisor discount, `m5.8xlarge`, us-west-2, 2026-09-24 | -69%, taking the trio run from about $10 to about $3.65, with the head still on demand |
| saving, modelled at assumed interruptions of 1.5-15% per node-hour | 58-64% |
| what a resume mechanism would add | $0.05-$0.61 per cohort run, so there is none |
| where restart-from-zero stops paying | roughly `1/rate` hours per assembly: ten hours at 10% per node-hour |

Past that point a reclaim costs more than spot saves, so a whole-genome assembly belongs on demand.

## Demo reads

`tools/stage-demo-data.sh` keeps whole reads rather than the overlapping portion, so coverage tapers
over about one read length at each edge of the region and the reported coverage runs a percent or
two high. It drops secondary and supplementary records (`-F 0x900`), so a read whose primary
alignment falls outside the region is absent even if part of it aligns inside, as across a
structural breakpoint at the boundary.
