#!/usr/bin/env python3
"""Does this multi-model packing fit on one GPU, and what binds when it does not?

    python packing.py                                  # the source engagement's config
    python packing.py --stages measured --vram 22.03    # this template's four models, on an L4
    python packing.py --strict                         # exit 1 if any GPU is over-committed
    python packing.py --json

Two tables. `--stages shipped` is the source engagement's configuration on its own model set,
which is where every lever in the README was measured. `--stages measured` is this template's
sam3 + dinov3 + siglip2, measured on a g6 L4. Their footprints differ: 21.01 GiB shipped,
5.53 GiB measured. Do not quote one for the other.

An L4 is 24 GB, which is 22.35 GiB, of which torch reports 22.03 usable. Pass `--vram 22.03`;
`--vram 24` invents about 2 GiB of headroom.

THE TRAP

`num_gpus=0.02` is an admission-control token and does not cap VRAM. Ray will place 50 actors
of that stage on one GPU because the fractions sum to 1.0, and CUDA then OOMs. The two limits
come from different numbers and the smaller wins:

    by fraction :  floor(1 / num_gpus)                     actors per GPU
    by VRAM     :  floor(vram_per_gpu / vram_per_actor)     actors per GPU

On the shipped configuration the fraction budget allows 20 object-embedder actors on one GPU
and VRAM allows 6. Reading only the fractions provisions three times what fits.

WHETHER PACKING PAYS: MEASURED, >=26.5% ON ONE L4

Not asserted here. On one L4, 96 frames at 640x480, one actor per stage:

    coresident  n=3  1.2619 / 1.2631 / 1.2792 rows/s   (spread 1.4%)
    serial      n=3  0.9882 / 0.9903 / 0.9979 rows/s   (spread 1.0%)
    verdict: SEPARABLE, coresident > serial by >= 26.5%

`serial` gives each stage the whole GPU with `materialize()` between stages, so the models are
never co-resident. The metric is end-to-end wall clock including warm model loads. Do not
compare it to the ~49.7 rows/s steady-state figure below. Three rounds ran: >=27.7%, >=27.9%,
>=26.5%. The first two needed a run discarded; the third put the throwaway inside the harness.

Re-run `measure_packing.py` after any change to the model set, the image, the batch sizes or
the card. It carries the three guards and the failure each one prevents.

WHAT THIS FILE ASSERTS

Feasibility and ordering, never throughput. It needs no cluster, no GPU and no weights.
Throughput for this shape is fleet-specific:

    ~49.7 rows/s    end to end, fixed confirmation run on the source fleet
    ~88 rows/s      active compute only; output prefix carried ~321k duplicate rows
    209.999 rows/s  extraction only, no detector

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Stage:
    """One model stage, as it appears in a Ray Data `map_batches` call.

    `vram_gib` is measured per actor at `batch`/`subbatch` and does not survive a change to
    either: batch 192-256 OOMed at 30+ GiB per actor, and batch 32 landed at 4.95 GiB. Record
    the batch alongside the VRAM figure.
    """

    name: str
    num_gpus: float
    vram_gib: float
    actors: int
    batch: int
    subbatch: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.num_gpus <= 1:
            raise ValueError(f"{self.name}: num_gpus must be in (0, 1], got {self.num_gpus}")
        if self.vram_gib <= 0:
            raise ValueError(f"{self.name}: vram_gib must be positive")
        if self.actors < 1:
            raise ValueError(f"{self.name}: actors must be >= 1")

    @property
    def by_fraction(self) -> int:
        """Actors per GPU that Ray's admission control will allow."""
        return math.floor(1.0 / self.num_gpus)

    def by_vram(self, vram_per_gpu: float) -> int:
        """Actors per GPU that the memory will actually hold."""
        return math.floor(vram_per_gpu / self.vram_gib)

    def binds(self, vram_per_gpu: float) -> str:
        f, v = self.by_fraction, self.by_vram(vram_per_gpu)
        if v < f:
            return "vram"
        if f < v:
            return "fraction"
        return "both"


# The configuration behind the source engagement's best full run (G2 fleet, 2026-06-09):
#
#   detector : 10 actors x num_gpus 0.2, batch 4, bfloat16, torch.compile DEFAULT mode
#   metrics  :  8-24 actors x num_gpus 0.02, batch 32, sub-batch 4
#   embedders:            num_gpus 0.05
#
# Per-actor VRAM measured on the same shape: object embedding 7.65 GiB, image embedding
# 0.96 GiB, metrics ~2.8 GiB at sub-batch 4. The detector figure is the one the source
# never recorded per-actor; 0.2 of a 48 GiB card is 9.6 GiB of fraction budget and the
# stage is the throughput floor at ~13 images/s per actor, so it is entered here as its
# fraction share, flagged UNMEASURED.
SHIPPED = [
    Stage("detector", num_gpus=0.2, vram_gib=9.6, actors=10, batch=4),
    Stage("object-embedder", num_gpus=0.05, vram_gib=7.65, actors=4, batch=32),
    Stage("image-embedder", num_gpus=0.05, vram_gib=0.96, actors=4, batch=32),
    Stage("metrics", num_gpus=0.02, vram_gib=2.8, actors=8, batch=32, subbatch=4),
]

UNMEASURED = {"detector"}  # vram_gib is its fraction share, not a measurement

# THIS TEMPLATE'S OWN MODEL SET, MEASURED. Two GPU probe jobs on a g6 L4, 2026-08-17,
# 640x480 frames, bf16, no torch.compile. `SHIPPED` above is the SOURCE ENGAGEMENT's
# configuration on a different model set, and its 21.01 GiB one-each footprint describes
# their models, not sam3 + dinov3 + siglip2. Quoting it as this set's footprint was the
# mistake these numbers replace.
#
# Peak allocated ABOVE the pre-stage baseline, so weights plus activations for that stage
# alone. `batch` is a field, not a comment: a VRAM figure without its batch does not plan.
#
#   detector          batch 8, 2 labels   4.19 GiB   (weights alone 1.60)
#   object-embedder   batch 4             0.59 GiB   <- see the caveat below
#   image-embedder    batch 4             0.72 GiB
#   metrics           batch 4, sub 4      0.03 GiB
#
# CAVEAT ON THE OBJECT EMBEDDER, and it is the same trap as everywhere else in this file:
# it was measured at ONE crop per frame, so 4 crops total. Its cost scales with DETECTIONS,
# not with frames, and the source engagement's 7.65 GiB was 32 frames' worth of crops. This
# figure is a floor for a nearly-empty batch and must not be read as a per-actor budget for
# a busy one. It is entered here because a measured floor beats an invented number, and
# flagged for the same reason.
MEASURED_L4 = [
    Stage("detector", num_gpus=0.2, vram_gib=4.19, actors=1, batch=8),
    Stage("object-embedder", num_gpus=0.05, vram_gib=0.59, actors=1, batch=4),
    Stage("image-embedder", num_gpus=0.05, vram_gib=0.72, actors=1, batch=4),
    Stage("metrics", num_gpus=0.02, vram_gib=0.03, actors=1, batch=4, subbatch=4),
]

STAGE_SETS = {"shipped": SHIPPED, "measured": MEASURED_L4}

# Which stages carry a fraction share instead of a measurement, PER TABLE. The detector is
# the source engagement's one gap; on the measured table it is the best-measured stage of
# the four, and printing "(UNMEASURED)" beside a real figure would be a lie in the direction
# that costs least to tell.
UNMEASURED_BY_SET = {"shipped": UNMEASURED, "measured": set()}

# A FLOOR for this template's own model set, from the hub file sizes, 2026-08-17. Not a
# measurement. Weights only: no activations, no workspace, no allocator slack.
#
#   facebook/sam3                model.safetensors  3.44 GB fp32  -> ~1.72 GB bf16
#   facebook/dinov3-vitl16       model.safetensors  1.21 GB fp32  -> ~0.61 GB bf16
#   google/siglip2-base          model.safetensors  1.50 GB fp32  -> ~0.75 GB bf16 (both
#                                                    towers; this template uses the vision
#                                                    tower only, so less)
#
# Two things follow, and neither is a per-actor figure. The table above is the SOURCE
# ENGAGEMENT's measurement on a DIFFERENT model set, so these floors do not replace it --
# they bound it. And the object embedder's 7.65 GiB against a 0.61 GB weight floor says
# activations at batch 32 dominate its footprint by an order of magnitude. The GPU measurement that replaces the
# detector's UNMEASURED entry has these numbers to sanity-check itself against: a measured
# per-actor figure BELOW its own weight floor is a measurement error, not a win.

# G2: one workstation-class 48 GB card. The fleet the shipped configuration was tuned on.
DEFAULT_VRAM_GIB = 48.0


@dataclass
class StageVerdict:
    name: str
    requested_actors: int
    per_gpu_by_fraction: int
    per_gpu_by_vram: int
    binds: str
    gpus_needed_for_requested: int
    vram_gib_per_actor: float
    vram_measured: bool


def plan(stages: list[Stage], vram_per_gpu: float = DEFAULT_VRAM_GIB,
         unmeasured: set[str] | None = None) -> list[StageVerdict]:
    out = []
    for s in stages:
        per_gpu = min(s.by_fraction, s.by_vram(vram_per_gpu))
        out.append(
            StageVerdict(
                name=s.name,
                requested_actors=s.actors,
                per_gpu_by_fraction=s.by_fraction,
                per_gpu_by_vram=s.by_vram(vram_per_gpu),
                binds=s.binds(vram_per_gpu),
                gpus_needed_for_requested=(
                    math.ceil(s.actors / per_gpu) if per_gpu else 0
                ),
                vram_gib_per_actor=s.vram_gib,
                vram_measured=s.name not in (UNMEASURED if unmeasured is None else unmeasured),
            )
        )
    return out


def overcommitted(stages: list[Stage], vram_per_gpu: float) -> list[str]:
    """Stages whose requested actors do not fit the GPUs their fractions imply.

    `num_gpus` says the placement is legal and VRAM says it is not. Ray schedules on the
    first.
    """
    bad = []
    for s in stages:
        if s.by_vram(vram_per_gpu) < 1:
            bad.append(
                f"{s.name}: one actor needs {s.vram_gib} GiB, GPU has {vram_per_gpu} GiB "
                f"-- does not fit at all"
            )
            continue
        gpus_by_fraction = math.ceil(s.actors / s.by_fraction)
        fits = gpus_by_fraction * s.by_vram(vram_per_gpu)
        if s.actors > fits:
            bad.append(
                f"{s.name}: {s.actors} actors at num_gpus={s.num_gpus} pack onto "
                f"{gpus_by_fraction} GPU(s), which hold {fits} at "
                f"{s.vram_gib} GiB each -- over-committed by {s.actors - fits}"
            )
    return bad


def coresident_footprint(stages: list[Stage], actors_each: int = 1) -> float:
    """VRAM held when one GPU carries `actors_each` of EVERY stage at once."""
    return sum(s.vram_gib * actors_each for s in stages)


def check_coresidency(stages: list[Stage], vram_per_gpu: float) -> list[str]:
    """Can all four stages actually sit on one card together?

    The per-stage checks ask how many of one stage fit. This asks whether the set fits, which
    is the binding constraint here. Checking stages independently calls an L4 fine for a set
    whose one-actor-each footprint is 21.0 GiB and whose two-detector variant is 30.6 GiB.
    """
    problems = []
    one = coresident_footprint(stages, 1)
    if one > vram_per_gpu:
        problems.append(
            f"one actor of each stage holds {one:.2f} GiB, GPU has {vram_per_gpu:g} GiB "
            f"-- the four models cannot be co-resident at all, which is the whole thesis"
        )
        return problems
    headroom = vram_per_gpu - one
    if headroom < 0.1 * vram_per_gpu:
        problems.append(
            f"one actor of each stage holds {one:.2f} GiB of {vram_per_gpu:g} GiB, leaving "
            f"{headroom:.2f} GiB ({headroom / vram_per_gpu:.0%}). Activation peaks are not "
            f"in these steady-state figures; under 10% headroom is where the source "
            f"engagement's batch-192 OOM lived"
        )
    return problems


def check_ordering(stages: list[Stage]) -> list[str]:
    """The relative-cost ordering the shape depends on.

    A detector that is cheap next to the embedders inverts the shape, and the packing defaults
    stop transferring. Direction only; the magnitudes are this fleet's.
    """
    by_name = {s.name: s for s in stages}
    problems = []
    det, obj, img = (by_name.get(n) for n in ("detector", "object-embedder", "image-embedder"))
    if det and obj and det.num_gpus <= obj.num_gpus:
        problems.append(
            f"detector requests {det.num_gpus} GPU and the object embedder "
            f"{obj.num_gpus}: the detector must be the larger reservation, or the shape "
            f"has inverted and the packing lesson does not transfer"
        )
    if obj and img and obj.vram_gib <= img.vram_gib:
        problems.append(
            f"object embedder is {obj.vram_gib} GiB and the image embedder "
            f"{img.vram_gib} GiB: the object embedder is supposed to be the VRAM hog"
        )
    return problems


def render(verdicts: list[StageVerdict], vram: float) -> str:
    w = max(len(v.name) for v in verdicts)
    lines = [
        f"GPU: {vram:g} GiB",
        "",
        f"{'stage':<{w}}  actors  per-GPU(frac)  per-GPU(vram)  binds     GPUs  GiB/actor",
    ]
    for v in verdicts:
        star = "" if v.vram_measured else "  (UNMEASURED)"
        lines.append(
            f"{v.name:<{w}}  {v.requested_actors:>6}  {v.per_gpu_by_fraction:>13}  "
            f"{v.per_gpu_by_vram:>13}  {v.binds:<8}  {v.gpus_needed_for_requested:>4}  "
            f"{v.vram_gib_per_actor:>9.2f}{star}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vram", type=float, default=DEFAULT_VRAM_GIB,
                    help=f"VRAM per GPU in GiB (default {DEFAULT_VRAM_GIB:g}, the source fleet)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any stage is over-committed or the ordering inverted")
    ap.add_argument("--stages", choices=sorted(STAGE_SETS), default="shipped",
                    help="'shipped' is the source engagement's config on ITS model set; "
                         "'measured' is this template's own four models, measured on an L4")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    stages = STAGE_SETS[args.stages]
    verdicts = plan(stages, args.vram, UNMEASURED_BY_SET[args.stages])
    over = overcommitted(stages, args.vram)
    # The ordering rule ("the object embedder is the VRAM hog") is a property of a loaded
    # batch. The measured table is one crop per frame, so the rule does not apply there.
    order = [] if args.stages == "measured" else check_ordering(stages)
    cores = check_coresidency(stages, args.vram)

    if args.json:
        print(json.dumps(
            {"vram_gib": args.vram,
             "stages": [asdict(v) for v in verdicts],
             "overcommitted": over,
             "coresidency": cores,
             "coresident_gib_one_each": coresident_footprint(stages, 1),
             "ordering": order},
            indent=2))
    else:
        print(render(verdicts, args.vram))
        vram_bound = [v.name for v in verdicts if v.binds == "vram"]
        if vram_bound:
            print(f"\nVRAM binds before the fraction does on: {', '.join(vram_bound)}.")
            print("Reading only num_gpus from the config would over-provision those stages.")
        print(f"\nco-resident, one actor of each stage: "
              f"{coresident_footprint(stages, 1):.2f} GiB of {args.vram:g} GiB")
        for problem in over:
            print(f"\nOVER-COMMITTED  {problem}")
        for problem in cores:
            print(f"\nCO-RESIDENCY    {problem}")
        for problem in order:
            print(f"\nORDERING        {problem}")
        if not over and not cores and not order:
            print("\nfeasible co-resident, and the relative-cost ordering holds")

    return 1 if (args.strict and (over or cores or order)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
