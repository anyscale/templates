#!/usr/bin/env python3
"""Will this multi-model packing actually fit, and what binds when it does not?

    python packing.py                             # the engagement's shipped configuration
    python packing.py --stages measured --vram 22.03   # THIS template's four models, on an L4
    python packing.py --strict                    # exit 1 if any GPU is over-committed
    python packing.py --json

Two tables, and the difference matters. `--stages shipped` is the source engagement's
configuration on the source engagement's MODEL SET, which is where every lever in the
README was measured. `--stages measured` is this template's own sam3 + dinov3 + siglip2,
measured on a g6 L4. Reading the shipped table's 21.01 GiB as this set's footprint was a
real mistake in an earlier version of this file's README: the measured set is 5.53 GiB.

An L4 is 24 **GB**, which is 22.35 GiB, and torch reports 22.03 GiB usable. `--vram 24`
is a unit error worth about 2 GiB of headroom that does not exist.

THE TRAP THIS ANSWERS

`num_gpus=0.02` is an ADMISSION-CONTROL TOKEN, not a memory limit. Ray will happily place
50 actors of that stage on one GPU because the fractions sum to 1.0, and CUDA will then
OOM, because nothing in the fraction says anything about VRAM. The two limits are
computed from different numbers and the smaller one wins:

    by fraction :  floor(1 / num_gpus)                     actors per GPU
    by VRAM     :  floor(vram_per_gpu / vram_per_actor)     actors per GPU

On the source engagement's own shipped configuration the fraction budget allows 20
object-embedder actors on one GPU and VRAM allows 6. Anyone reading only the fractions in
the config would provision three times the actors that fit. That is the arithmetic behind
"pack the GPUs, do not count them", and it is why this file is executable rather than a
paragraph.

DOES PACKING ACTUALLY BUY ANYTHING? MEASURED: YES, >=26.5% ON ONE L4.

Not arithmetic, and not asserted by this file -- recorded here because a tool that computes
feasibility should say whether feasibility was worth pursuing. On one L4, 96 frames at
640x480, one actor per stage:

    coresident  n=3  1.2619 / 1.2631 / 1.2792 rows/s   (spread 1.4%)
    serial      n=3  0.9882 / 0.9903 / 0.9979 rows/s   (spread 1.0%)
    verdict: SEPARABLE, coresident > serial by >= 26.5%

`serial` = each stage alone with the whole GPU, `materialize()` between stages, so the models
are never co-resident. The metric is end-to-end wall clock INCLUDING warm model loads, which
is a different quantity from the ~49.7 rows/s steady-state figure quoted below; do not compare
them. Three rounds agreed (>=27.7%, >=27.9%, >=26.5%); the first two needed a run discarded
and so could not carry the claim, and the third put the throwaway inside the harness instead.
Full scope, and the failure each guard prevents, is in `measure_packing.py`'s docstring.

**Re-run it with `measure_packing.py`** after any change to the model set, the image, the
batch sizes or the card. That script carries the three guards those failed rounds bought --
a warmup on the node that does the work, a throwaway pass per arm, and interleaved arms --
and it reports a bounded verdict rather than a ratio: an arm needs two runs, and two arms
are separable only when the worst run of the better one beats the best run of the worse one.

WHY NO WALL-CLOCK NUMBER IS ASSERTED HERE

This is rung 1: it needs no cluster, no GPU and no weights, so nothing can block it. It
asserts feasibility and ordering, never throughput. The measured throughput for this shape
is fleet-specific and the honest end-to-end figure is ~49.7 rows/s on the source fleet's
fixed confirmation run -- not the 88 rows/s (active compute only, and its output prefix
carried ~321k duplicate rows) and not the 209.999 rows/s (extraction only, no detector).
Those belong in the README with their scope, not in a test.

Stdlib only, on purpose.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Stage:
    """One model stage, as it appears in a Ray Data `map_batches` call.

    `vram_gib` is measured per actor AT `batch`/`subbatch`, and does not survive a change
    to either: the source engagement OOMed at batch 192-256 with 30+ GiB per actor and
    landed at 4.95 GiB by moving to batch 32. A VRAM figure without its batch is not a
    number you can plan with.
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
# fraction share and flagged UNMEASURED rather than invented.
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
# alone. Each figure is meaningless without the batch beside it, which is why `batch` is a
# field rather than a comment.
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

# A FLOOR for this template's own model set, from the hub file sizes rather than a
# measurement (2026-08-17, once the gated repos became readable). Weights only: no
# activations, no workspace, no allocator slack.
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
# activations at batch 32 dominate its footprint by an order of magnitude, which is why a
# VRAM number without its batch is useless. The GPU measurement that replaces the
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

    This is the failure the fraction hides: `num_gpus` says the placement is legal and
    VRAM says it is not, and Ray schedules on the first of those.
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

    The per-stage checks above ask "how many of THIS stage fit", which is the wrong
    question for this template: the thesis is that four DIFFERENT models share a GPU. The
    first version of this file only checked stages independently and would have called an
    L4 fine for a set whose one-actor-each footprint is 21.0 GiB and whose two-detector
    variant is 30.6 GiB. Co-residency is the constraint; per-stage capacity is a side
    condition.
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

    This is the reason a substitute model set cannot be chosen casually: a detector that
    is cheap relative to the embedders inverts the shape and teaches the wrong defaults. Direction only -- no magnitude is asserted,
    because the magnitudes are this fleet's.
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
    # The ordering rule ("the object embedder is the VRAM hog") is a property of a LOADED
    # batch. On the measured table it is one crop per frame, so the rule is not applicable
    # rather than violated, and reporting it would train the reader to ignore the check.
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
