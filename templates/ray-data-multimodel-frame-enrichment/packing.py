#!/usr/bin/env python3
"""Check whether a set of Ray Data stages fits on one GPU, and which limit binds each stage.

    python packing.py                                  # the shipped configuration, 48 GiB card
    python packing.py --stages measured --vram 22.03    # this template's models, on an L4
    python packing.py --strict                         # exit 1 on any problem found
    python packing.py --json

Two tables. `--stages shipped` is pipeline.py's default actor counts and fractions, with the
per-actor VRAM of the production workload they came from, measured on that workload's own
models. `--stages measured` is this template's SAM 3, DINOv3 and SigLIP2, measured on a g6 L4.
One actor of each holds 21.01 GiB shipped and 5.53 GiB measured; do not quote one for the other.

An L4 is 24 GB, which is 22.35 GiB, of which torch reports 22.03 usable. Pass `--vram 22.03`;
`--vram 24` invents about 2 GiB of headroom.

WHY TWO LIMITS

`num_gpus=0.02` is admission control and does not cap VRAM. Ray will place 50 actors of that
stage on one GPU because the fractions sum to 1.0, and CUDA then OOMs. The smaller of two
numbers wins:

    by fraction :  floor(1 / num_gpus)                     actors per GPU
    by VRAM     :  floor(vram_per_gpu / vram_per_actor)     actors per GPU

On the shipped configuration the fraction allows 20 object-embedder actors on one GPU and VRAM
allows 6. Reading only the fractions provisions three times what fits.

This file checks feasibility and ordering, never throughput, and needs no cluster, GPU or
weights. Whether packing pays is measure_packing.py's question. Stdlib only.
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
    either: on the shipped workload's metrics stage, batch 192-256 OOMed at 30+ GiB per actor
    and batch 32 used 4.95 GiB. Record the batch with the VRAM figure.
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


# pipeline.py's defaults, from the production workload's best full run (2026-06-09):
#
#   detector : 10 actors x num_gpus 0.2, batch 4, bfloat16, torch.compile `default` mode
#   metrics  :  8-24 actors x num_gpus 0.02, batch 32, sub-batch 4
#   embedders:            num_gpus 0.05
#
# Per-actor VRAM measured on that workload's models: object embedding 7.65 GiB, image
# embedding 0.96 GiB, metrics ~2.8 GiB at sub-batch 4. Its detector was never measured per
# actor, so the entry is its fraction share of a 48 GiB card (0.2 x 48 = 9.6 GiB), flagged
# UNMEASURED.
SHIPPED = [
    Stage("detector", num_gpus=0.2, vram_gib=9.6, actors=10, batch=4),
    Stage("object-embedder", num_gpus=0.05, vram_gib=7.65, actors=4, batch=32),
    Stage("image-embedder", num_gpus=0.05, vram_gib=0.96, actors=4, batch=32),
    Stage("metrics", num_gpus=0.02, vram_gib=2.8, actors=8, batch=32, subbatch=4),
]

UNMEASURED = {"detector"}  # vram_gib is its fraction share, not a measurement

# This template's own models, measured in two GPU probe jobs on a g6 L4, 2026-08-17: 640x480
# frames, bf16, no torch.compile. Each figure is peak allocated above the pre-stage baseline,
# so weights plus activations for that stage alone. `batch` is a field because a VRAM figure
# without its batch cannot be planned with.
#
#   detector          batch 8, 2 labels   4.19 GiB   (weights alone 1.60)
#   object-embedder   batch 4             0.59 GiB   <- see the caveat below
#   image-embedder    batch 4             0.72 GiB
#   metrics           batch 4, sub 4      0.03 GiB
#
# The object embedder was measured at one crop per frame, 4 crops in all. Its cost scales with
# detections, not frames; the shipped workload's 7.65 GiB was 32 frames' worth of crops. Treat
# 0.59 GiB as a floor for a nearly empty batch, not a per-actor budget for a busy one.
MEASURED_L4 = [
    Stage("detector", num_gpus=0.2, vram_gib=4.19, actors=1, batch=8),
    Stage("object-embedder", num_gpus=0.05, vram_gib=0.59, actors=1, batch=4),
    Stage("image-embedder", num_gpus=0.05, vram_gib=0.72, actors=1, batch=4),
    Stage("metrics", num_gpus=0.02, vram_gib=0.03, actors=1, batch=4, subbatch=4),
]

STAGE_SETS = {"shipped": SHIPPED, "measured": MEASURED_L4}

# Stages whose VRAM figure is a fraction share instead of a measurement, per table. The
# measured table's detector is measured, so nothing is flagged there.
UNMEASURED_BY_SET = {"shipped": UNMEASURED, "measured": set()}

# Weight-only floors for this template's models, from the hub file sizes, 2026-08-17. Not
# measurements: no activations, no workspace, no allocator slack.
#
#   facebook/sam3                model.safetensors  3.44 GB fp32  -> ~1.72 GB bf16
#   facebook/dinov3-vitl16       model.safetensors  1.21 GB fp32  -> ~0.61 GB bf16
#   google/siglip2-base          model.safetensors  1.50 GB fp32  -> ~0.75 GB bf16 (both
#                                                    towers; this template uses the vision
#                                                    tower only, so less)
#
# Use them to sanity-check a measurement: a per-actor figure below its model's weight floor is
# a measurement error.

# The production workload's card: one workstation-class 48 GB GPU, the one the shipped
# configuration was tuned on.
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
    """VRAM held when one GPU carries `actors_each` of every stage at once."""
    return sum(s.vram_gib * actors_each for s in stages)


def check_coresidency(stages: list[Stage], vram_per_gpu: float) -> list[str]:
    """Can one actor of every stage sit on one card together?

    The per-stage checks ask how many of one stage fit; this asks whether the set does, which
    is the binding constraint here. On an L4, a per-stage check passes a set whose
    one-actor-each footprint is 21.0 GiB, or 30.6 GiB with two detectors.
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
            f"in these steady-state figures; under 10% headroom is where the originating "
            f"workload's batch-192 OOM lived"
        )
    return problems


def check_ordering(stages: list[Stage]) -> list[str]:
    """The relative-cost ordering the shape depends on.

    A detector that is cheap next to the embedders inverts the shape, and the packing defaults
    stop transferring. Direction only; the magnitudes are the shipped workload's. main() skips
    this for the measured table, which was taken at one crop per frame.
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
                    help=f"VRAM per GPU in GiB (default {DEFAULT_VRAM_GIB:g}, the card the "
                         "shipped configuration was tuned on)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if a stage is over-committed, one actor of each stage does "
                         "not fit on one GPU, or (shipped table) the cost ordering has inverted")
    ap.add_argument("--stages", choices=sorted(STAGE_SETS), default="shipped",
                    help="'shipped': pipeline.py's defaults with the production workload's "
                         "VRAM figures; 'measured': this template's models, measured on an L4")
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
