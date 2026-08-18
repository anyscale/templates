#!/usr/bin/env python3
"""Tests for packing.py.

    python tests/test_packing.py

Stdlib `unittest`, no GPU, no weights, no cluster. That is the point: this is the rung-1
artifact, so its tests must run anywhere the template is checked out.

Every assertion here is a DIRECTION or a FEASIBILITY, never a magnitude. A test that
pinned "6 actors per GPU" would go stale on the next card and then be loosened until it
passed, which is worse than no test.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # @dataclass resolves through sys.modules
    spec.loader.exec_module(module)
    return module


pk = _load("packing")
# `measure_packing` needs a GPU and the gated weights to RUN, but its module level is
# stdlib only, so the separability rule it applies is testable here -- which is the half
# of that script a wrong answer would come out of.
mp = _load("measure_packing")


class TheTrap(unittest.TestCase):
    """The claim the file exists to make: the fraction is not a memory limit."""

    def test_vram_binds_before_the_fraction_on_the_shipped_config(self):
        # The headline. If this ever stops holding, the template is teaching a trap that
        # no longer exists and the prose has to change with it.
        verdicts = {v.name: v for v in pk.plan(pk.SHIPPED, pk.DEFAULT_VRAM_GIB)}
        obj = verdicts["object-embedder"]
        self.assertEqual(obj.binds, "vram")
        self.assertLess(
            obj.per_gpu_by_vram, obj.per_gpu_by_fraction,
            "the object embedder is the stage where reading num_gpys alone over-provisions",
        )

    def test_at_least_one_stage_is_fraction_bound_so_the_test_is_not_vacuous(self):
        # If every stage were VRAM-bound the comparison above would be trivially true.
        # The image embedder is cheap enough that its fraction binds, which is what makes
        # "the smaller of the two wins" a real rule rather than a restatement.
        verdicts = {v.name: v for v in pk.plan(pk.SHIPPED, pk.DEFAULT_VRAM_GIB)}
        self.assertEqual(verdicts["image-embedder"].binds, "fraction")


class Feasibility(unittest.TestCase):
    def test_the_shipped_config_fits_the_fleet_it_was_tuned_on(self):
        self.assertEqual(pk.overcommitted(pk.SHIPPED, pk.DEFAULT_VRAM_GIB), [])

    def test_the_same_config_is_over_committed_on_a_smaller_card(self):
        # Direction, not magnitude: a 24 GiB card must be reported as over-committed at
        # these actor counts. This is the CPU-side half of "does packing survive CI
        # scale" -- at the shipped counts it does not, and the answer is arithmetic.
        problems = pk.overcommitted(pk.SHIPPED, 24.0)
        self.assertTrue(problems, "a 24 GiB card must not silently accept this config")
        self.assertTrue(any("detector" in p for p in problems))

    def test_a_stage_too_large_for_any_single_gpu_says_so_distinctly(self):
        huge = [pk.Stage("detector", num_gpus=0.5, vram_gib=80.0, actors=1, batch=1)]
        problems = pk.overcommitted(huge, 24.0)
        self.assertTrue(any("does not fit at all" in p for p in problems))

    def test_strict_decides_the_exit_code_and_plain_reporting_does_not(self):
        self.assertEqual(pk.main(["--vram", "24", "--strict"]), 1)
        self.assertEqual(pk.main(["--vram", "24"]), 0)
        self.assertEqual(pk.main(["--strict"]), 0)


class CoResidency(unittest.TestCase):
    """The constraint the per-stage checks cannot see, and the template's actual thesis."""

    def test_all_four_stages_fit_one_card_on_both_fleets(self):
        for vram in (pk.DEFAULT_VRAM_GIB, 24.0):
            self.assertEqual(
                pk.check_coresidency(pk.SHIPPED, vram), [],
                f"one actor of each stage must fit a {vram:g} GiB card",
            )

    def test_the_footprint_is_the_sum_and_not_the_largest_stage(self):
        one = pk.coresident_footprint(pk.SHIPPED, 1)
        largest = max(s.vram_gib for s in pk.SHIPPED)
        self.assertGreater(one, largest, "co-residency has to add the stages up")

    def test_a_card_too_small_for_the_set_is_caught_even_when_each_stage_fits_alone(self):
        # The decisive case. Every stage fits a 10 GiB card on its own; the SET does not.
        # A per-stage check calls this fine, which is why check_coresidency exists.
        small = 10.0
        for s in pk.SHIPPED:
            self.assertGreaterEqual(
                s.by_vram(small), 1, f"{s.name} alone must fit for this test to be about the sum"
            )
        problems = pk.check_coresidency(pk.SHIPPED, small)
        self.assertTrue(problems)
        self.assertIn("cannot be co-resident", problems[0])

    def test_thin_headroom_is_reported_before_it_becomes_an_oom(self):
        # Steady-state VRAM is not peak VRAM. The source engagement's batch-192 OOM lived
        # in the gap between them, so a set that only just fits is a finding.
        one = pk.coresident_footprint(pk.SHIPPED, 1)
        problems = pk.check_coresidency(pk.SHIPPED, one * 1.02)
        self.assertTrue(any("headroom" in p for p in problems))


class Ordering(unittest.TestCase):
    """The constraint on substituting models, which is why this shape is fussy."""

    def test_the_shipped_ordering_holds(self):
        self.assertEqual(pk.check_ordering(pk.SHIPPED), [])

    def test_a_detector_cheaper_than_the_embedders_is_caught(self):
        # The README's warning about substituting models, made executable: swap in a light
        # detector and the shape inverts, so the packing lesson stops transferring.
        inverted = [
            pk.Stage("detector", num_gpus=0.02, vram_gib=1.0, actors=1, batch=4),
            pk.Stage("object-embedder", num_gpus=0.05, vram_gib=7.65, actors=1, batch=32),
            pk.Stage("image-embedder", num_gpus=0.05, vram_gib=0.96, actors=1, batch=32),
        ]
        problems = pk.check_ordering(inverted)
        self.assertTrue(any("inverted" in p for p in problems))

    def test_an_object_embedder_lighter_than_the_image_embedder_is_caught(self):
        swapped = [
            pk.Stage("detector", num_gpus=0.2, vram_gib=9.6, actors=1, batch=4),
            pk.Stage("object-embedder", num_gpus=0.05, vram_gib=0.5, actors=1, batch=32),
            pk.Stage("image-embedder", num_gpus=0.05, vram_gib=0.96, actors=1, batch=32),
        ]
        self.assertTrue(any("VRAM hog" in p for p in pk.check_ordering(swapped)))


class Inputs(unittest.TestCase):
    def test_a_fraction_outside_zero_to_one_is_refused(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                pk.Stage("x", num_gpus=bad, vram_gib=1.0, actors=1, batch=1)

    def test_the_unmeasured_vram_figure_is_flagged_rather_than_passed_off(self):
        # The detector's per-actor VRAM was never recorded by the source engagement. It
        # is entered as its fraction share, and anything reading this plan has to be able
        # to tell that apart from a measurement.
        verdicts = {v.name: v for v in pk.plan(pk.SHIPPED)}
        self.assertFalse(verdicts["detector"].vram_measured)
        self.assertTrue(verdicts["object-embedder"].vram_measured)


class Separability(unittest.TestCase):
    """The rule behind the README's >=26.5%, which is the only number this template quotes
    from its own GPU. A margin is a claim about the runs, so the rule that produced it has
    to be executable rather than cited."""

    CORESIDENT = [1.2619, 1.2631, 1.2792]
    SERIAL = [0.9882, 0.9903, 0.9979]

    def test_the_recorded_run_reproduces_the_readme_figure(self):
        # Not a magnitude assertion about the hardware: an assertion that the arithmetic
        # behind the published sentence is the arithmetic this file ships. If the rule is
        # ever loosened, this fails and the README has to change with it.
        verdict = mp.separable(self.CORESIDENT, self.SERIAL)
        self.assertIn("SEPARABLE", verdict)
        self.assertIn(">= 26.5%", verdict)

    def test_the_margin_is_the_worst_case_gap_not_a_ratio_of_means(self):
        # Means would read ~27.5% and three runs do not support it. The bound has to be
        # the smaller number, or the rule is decorative.
        import statistics

        verdict = mp.separable(self.CORESIDENT, self.SERIAL)
        of_means = (statistics.mean(self.CORESIDENT) / statistics.mean(self.SERIAL) - 1) * 100
        self.assertGreater(of_means, 26.5, "the means must be the more flattering figure "
                                           "for this test to be about the difference")
        self.assertNotIn(f">= {of_means:.1f}%", verdict)

    def test_overlapping_ranges_are_refused_however_far_apart_the_means(self):
        # One slow run in the better arm is enough. This is the whole point of the rule:
        # 1.5 vs 1.0 on the averages, and still not separable.
        verdict = mp.separable([2.0, 2.0, 0.5], [1.0, 1.0, 1.0])
        self.assertIn("OVERLAP", verdict)
        self.assertNotIn("SEPARABLE", verdict)

    def test_a_single_run_per_arm_is_unsupported_not_a_win(self):
        verdict = mp.separable([5.0], [1.0])
        self.assertIn("UNSUPPORTED", verdict)
        self.assertNotIn("SEPARABLE", verdict)

    def test_the_direction_is_read_off_the_data_not_the_argument_order(self):
        # Hand it the arms in the losing order, labels and all. It must still name
        # coresident as the winner rather than crediting whichever argument came first --
        # a harness that reports "the first arm won" tells you about your call site.
        verdict = mp.separable(self.SERIAL, self.CORESIDENT, "serial", "coresident")
        self.assertIn("SEPARABLE", verdict)
        self.assertIn("coresident > serial", verdict)
        self.assertIn(">= 26.5%", verdict)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
