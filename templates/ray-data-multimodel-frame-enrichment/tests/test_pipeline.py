#!/usr/bin/env python3
"""Stage-level tests for pipeline.py. No Ray, no GPU, no weights.

    python tests/test_pipeline.py

The stages run as plain callables, so the shapes are pinned in milliseconds. The notebook runs
the Ray DAG on top: that catches scheduling, these catch answers.

The bug they exist for: the object embedder embeds a whole batch of crops in one forward pass
and its output column is per row. Writing the batch total into every row reports eight
detections for each of eight frames that had one apiece.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

os.environ["STUB"] = "1"  # before the module reads it

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pipeline", HERE.parent / "pipeline.py")
pl = importlib.util.module_from_spec(spec)
sys.modules["pipeline"] = pl
spec.loader.exec_module(pl)

import numpy as np  # noqa: E402  (after the stub env var is set)

W, H = 64, 48


def frame(count_seed: int, base: int) -> np.ndarray:
    """A frame whose detection count and content both vary.

    `base` must differ per frame. Otherwise the stub embedder returns the same vector for every
    crop, and a test asserting "each frame got its own vectors" passes against a scatter that
    assigns the right counts to the wrong rows. Varying only pixel [0, 0] is not enough: every
    crop excludes it.
    """
    f = np.full((H, W, 3), base, dtype=np.uint8)
    f[0, 0] = count_seed  # the stub detector derives its box count from this pixel
    return f


def batch(fills: list[int]) -> dict:
    frames = [frame(v, 40 + 17 * i) for i, v in enumerate(fills)]
    return {
        "frame_id": np.array([f"f{i}" for i in range(len(frames))]),
        "image": [f.tobytes() for f in frames],
        "width": np.array([W] * len(frames), dtype=np.int32),
        "height": np.array([H] * len(frames), dtype=np.int32),
    }


class Detection(unittest.TestCase):
    def test_the_stub_varies_its_count_including_zero(self):
        # A constant count would make every scatter test below vacuous.
        out = pl.Detector()(batch([0, 1, 2, 3]))
        counts = list(out["n_detections"])
        self.assertEqual(counts, [0, 1, 2, 3])

    def test_boxes_stay_inside_the_frame(self):
        out = pl.Detector()(batch([3]))
        for x0, y0, x1, y1 in out["boxes"][0]:
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x1, W)
            self.assertLessEqual(y1, H)


class ObjectEmbedding(unittest.TestCase):
    def setUp(self):
        self.det = pl.Detector()
        self.emb = pl.ObjectEmbedder()

    def run_two(self, fills: list[int]) -> dict:
        return self.emb(self.det(batch(fills)))

    def test_the_count_is_per_frame_not_the_batch_total(self):
        out = self.run_two([1, 2, 3])
        self.assertEqual(list(out["object_embedding_count"]), [1, 2, 3])
        self.assertEqual(list(out["n_detections"]), [1, 2, 3])

    def test_each_frame_gets_its_own_vectors(self):
        out = self.run_two([1, 2, 3])
        shapes = [np.asarray(v).shape for v in out["object_embeddings"]]
        self.assertEqual([s[0] for s in shapes], [1, 2, 3])
        self.assertEqual({s[1] for s in shapes}, {self.emb.dim})

    def test_a_frame_with_no_detections_gets_an_empty_array_not_a_raise(self):
        # The degenerate input.
        out = self.run_two([0, 2])
        first = np.asarray(out["object_embeddings"][0])
        self.assertEqual(first.shape, (0, self.emb.dim))
        self.assertEqual(int(out["object_embedding_count"][0]), 0)

    def test_a_whole_batch_with_no_detections_is_fine(self):
        out = self.run_two([0, 0, 0])
        self.assertEqual(list(out["object_embedding_count"]), [0, 0, 0])

    def test_vectors_are_not_all_identical_across_frames(self):
        # Guards a scatter that assigns the right count to each frame but the wrong rows.
        out = self.run_two([1, 1, 1])
        vals = [float(np.asarray(v)[0][0]) for v in out["object_embeddings"]]
        self.assertGreater(len(set(vals)), 1, "every frame got the same vector")


class Downstream(unittest.TestCase):
    def test_metrics_drops_the_blob_and_keeps_one_score_per_row(self):
        stages = [pl.Detector(), pl.ObjectEmbedder(), pl.ImageEmbedder(), pl.Metrics()]
        out = batch([1, 2, 0, 3])
        for stage in stages:
            out = stage(out)
        self.assertNotIn("image", out, "the blob column must not survive to the output")
        self.assertEqual(len(out["sharpness"]), 4)
        self.assertEqual(list(out["object_embedding_count"]), [1, 2, 0, 3])

    def test_the_metrics_subbatch_does_not_change_the_answer(self):
        # The sub-batch bounds VRAM and must not alter the result. If these diverge, the loop
        # is carrying state across chunks.
        stages = [pl.Detector(), pl.ObjectEmbedder(), pl.ImageEmbedder()]
        prepared = batch([1, 2, 0, 3, 1, 2])
        for stage in stages:
            prepared = stage(prepared)
        original = pl.METRICS_SUBBATCH
        try:
            pl.METRICS_SUBBATCH = 1
            one = pl.Metrics()(dict(prepared))["sharpness"]
            pl.METRICS_SUBBATCH = 4
            four = pl.Metrics()(dict(prepared))["sharpness"]
        finally:
            pl.METRICS_SUBBATCH = original
        np.testing.assert_allclose(one, four, rtol=1e-6)


class FakeTensor:
    """Just enough tensor for the detector's box path: numpy behind a torch surface."""

    def __init__(self, arr, transfers):
        self.arr = np.asarray(arr)
        self._transfers = transfers

    @property
    def shape(self):
        return self.arr.shape

    def to(self, *_a, **_k):
        return self

    def cpu(self):
        self._transfers.append(1)
        return self

    def numpy(self):
        return self.arr


class FakeTorch:
    """The torch surfaces the real branches touch."""

    int32 = "int32"
    float32 = "float32"
    # `ImageEmbedder` does `isinstance(feats, torch.Tensor)` to tell a raw tensor from an
    # output object, so the fake needs a Tensor type its own tensors satisfy.
    Tensor = FakeTensor

    def __init__(self):
        self.transfers: list[int] = []

    def cat(self, ts, dim=0):
        arrays = [t.arr for t in ts]
        if not arrays:
            return FakeTensor(np.zeros((0, 4), np.int32), self.transfers)
        return FakeTensor(np.concatenate(arrays, axis=dim), self.transfers)

    def zeros(self, shape, dtype=None):
        return FakeTensor(np.zeros(shape, np.int32), self.transfers)

    class _Ctx:
        def __enter__(self):
            return None

        def __exit__(self, *_a):
            return False

    def inference_mode(self):
        return self._Ctx()


class FakeEncoding(dict):
    """A dict subclass, because `BatchEncoding` is one and the code does `model(**inputs)`.

    A bare object passes the `**`-unpack without modelling the real class.
    """

    def to(self, _device):
        return self


class FakeProcessor:
    """Records every call so the prompt shape can be asserted."""

    def __init__(self, calls, boxes_for, transfers):
        self.calls = calls
        self.boxes_for = boxes_for
        self.transfers = transfers

    def __call__(self, images=None, text=None, return_tensors=None):
        # The real fast tokenizer refuses `list[list[str]]`. Measured on real weights,
        # 2026-08-17: `TypeError: TextEncodeInput must be Union[...]`, because the
        # pre-tokenized-words form needs `is_split_into_words=True` and the processor never
        # passes it.
        if text is not None and not all(isinstance(t, str) for t in text):
            raise TypeError(
                f"text must be list[str], one prompt per image; got {type(text[0]).__name__} "
                "elements, which Sam3Processor reads as pre-tokenized words"
            )
        self.calls.append({"n_images": len(images), "text": text})
        return FakeEncoding(
            pixel_values=np.zeros((len(images), 3, 8, 8), np.float32),
            original_sizes=np.array([[H, W]] * len(images)),
        )

    def post_process_instance_segmentation(self, _out, threshold=None, target_sizes=None):
        assert target_sizes is not None, "target_sizes must be passed or masks stay resized"
        assert len(target_sizes) == self.calls[-1]["n_images"], "one size per image"
        prompt = self.calls[-1]["text"][0]
        n = self.boxes_for[prompt]
        return [
            {"boxes": FakeTensor(np.tile([1, 2, 3, 4], (n, 1)), self.transfers)}
            for _ in range(self.calls[-1]["n_images"])
        ]


class RealPathPromptShape(unittest.TestCase):
    """The non-stub branch, executed against a recording fake.

    `Sam3Processor` passes `text` straight to its tokenizer, so `list[str]` is one prompt per
    image and `list[list[str]]` is HuggingFace's pre-tokenized-words form, which the tokenizer
    rejects. The fake pins the call shape with no GPU and no weights.
    """

    def build(self, labels, boxes_for):
        det = pl.Detector.__new__(pl.Detector)
        det.stub = False
        torch = FakeTorch()
        det.torch = torch
        det.device = "cpu"
        det.calls = []
        det.processor = FakeProcessor(det.calls, boxes_for, torch.transfers)
        det.model = lambda **_kw: object()  # the outputs blob; only the processor reads it
        det.prompts = list(labels)
        return det, torch

    def test_one_pass_per_label_each_a_flat_list_of_strings(self):
        det, _ = self.build(["car", "person"], {"car": 1, "person": 2})
        det(batch([0, 0, 0]))
        self.assertEqual(len(det.calls), 2, "one processor call per label")
        for call, expected in zip(det.calls, ["car", "person"]):
            self.assertEqual(call["text"], [expected] * 3)
            self.assertTrue(all(isinstance(t, str) for t in call["text"]),
                            "a list of lists is pre-tokenized words, not several concepts")

    def test_boxes_are_the_union_over_labels_per_frame(self):
        det, _ = self.build(["car", "person"], {"car": 1, "person": 2})
        out = det(batch([0, 0, 0]))
        self.assertEqual(list(out["n_detections"]), [3, 3, 3])
        for b in out["boxes"]:
            self.assertEqual(np.asarray(b).shape, (3, 4))

    def test_one_host_transfer_for_the_whole_batch_not_b_times_l(self):
        # Why the boxes are concatenated before `.cpu()`: a transfer inside the loop is
        # B x L of them.
        det, torch = self.build(["car", "person", "sign"], {"car": 1, "person": 1, "sign": 1})
        det(batch([0, 0, 0, 0]))
        self.assertEqual(sum(torch.transfers), 1)

    def test_a_label_that_finds_nothing_contributes_nothing(self):
        det, _ = self.build(["car", "ghost"], {"car": 2, "ghost": 0})
        out = det(batch([0, 0]))
        self.assertEqual(list(out["n_detections"]), [2, 2])

    def test_no_labels_at_all_is_an_empty_result_not_a_raise(self):
        det, _ = self.build([], {})
        out = det(batch([0, 0]))
        self.assertEqual(list(out["n_detections"]), [0, 0])
        for b in out["boxes"]:
            self.assertEqual(np.asarray(b).shape, (0, 4))


class FakePooledOutput:
    """What `get_image_features` actually returns on transformers 5.15.0."""

    def __init__(self, pooled):
        self.pooler_output = pooled


class ImageEmbedderRealPath(unittest.TestCase):
    """The real branch of stage 3, which shipped broken with no test on it.

    `ImageEmbedder.__call__` did `self.model.get_image_features(**inputs).to(torch.float32)`.
    On transformers 5.15.0 that returns a `BaseModelOutputWithPooling`, not a tensor, and the
    first run with real weights died on
    `AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'to'`.

    Earlier tests of this stage ran the stub branch, which never touches the model. A stub
    cannot pin a library call's return type.
    """

    def build(self, returns, dim=4):
        emb = pl.ImageEmbedder.__new__(pl.ImageEmbedder)
        emb.stub = False
        torch = FakeTorch()
        emb.torch = torch
        emb.device = "cpu"
        emb.dim = dim
        emb.processor = FakeProcessor([], {}, torch.transfers)
        emb.model = type("M", (), {"get_image_features": lambda _s, **_kw: returns})()
        return emb, torch

    def vecs(self, n, dim=4):
        return FakeTensor(np.arange(n * dim, dtype=np.float32).reshape(n, dim), [])

    def test_pooled_output_object_is_unwrapped(self):
        emb, _ = self.build(FakePooledOutput(self.vecs(2)))
        out = emb(batch([0, 0]))
        self.assertEqual(len(out["image_embedding"]), 2)
        self.assertEqual(np.asarray(out["image_embedding"][0]).shape, (4,))

    def test_a_bare_tensor_still_passes_through(self):
        # The pin is a floor, and this return type has changed once.
        emb, _ = self.build(self.vecs(2))
        out = emb(batch([0, 0]))
        self.assertEqual(len(out["image_embedding"]), 2)

    def test_an_output_with_no_pooler_refuses_rather_than_guessing(self):
        emb, _ = self.build(type("Weird", (), {})())
        with self.assertRaises(RuntimeError) as ctx:
            emb(batch([0, 0]))
        # The message must name the model: in a four-stage pipeline "no pooler_output" does
        # not say which stage stopped.
        self.assertIn(pl.IMAGE_EMBED_MODEL, str(ctx.exception))

    def test_one_row_per_frame_not_one_per_batch(self):
        # Same shape as the object embedder's per-batch count in a per-row column.
        emb, _ = self.build(FakePooledOutput(self.vecs(3)))
        out = emb(batch([0, 0, 0]))
        self.assertEqual(len(out["image_embedding"]), 3)
        self.assertEqual(len(out["frame_id"]), 3)


class GatedStageSelection(unittest.TestCase):
    """Which stages CI may run. A licensing question, not a tuning one.

    SAM 3 and DINOv3 are gated and their terms are accepted per account, so CI runs the ungated
    half. If `stage_plan` returned a gated stage under `ungated_only`, CI would pull weights it
    has no right to and fail on a 401 that reads like an infrastructure error.
    """

    def test_the_full_plan_is_all_four_in_order(self):
        self.assertEqual(pl.stage_plan(False), ["detector", "obj", "img", "metrics"])

    def test_the_ungated_plan_drops_exactly_the_gated_stages(self):
        gated = [key for key, _cls, is_gated in pl.STAGES if is_gated]
        plan = pl.stage_plan(True)
        for key in gated:
            self.assertNotIn(key, plan, f"{key} is gated and must not be in the CI plan")
        self.assertEqual(plan, ["img", "metrics"], "order must survive the filter")

    def test_the_ungated_plan_is_not_empty_so_ci_still_measures_something(self):
        # A filter that removed everything would make the CI run vacuously green.
        self.assertTrue(pl.stage_plan(True))

    def test_the_gated_flags_match_the_gated_model_repositories(self):
        # Ties the flag to the repo, not to a comment. Both gated stages load a `facebook/`
        # repo; neither ungated stage does.
        model_of = {"detector": pl.DETECTOR_MODEL, "obj": pl.OBJECT_EMBED_MODEL,
                    "img": pl.IMAGE_EMBED_MODEL, "metrics": ""}
        for key, _cls, is_gated in pl.STAGES:
            with self.subTest(stage=key):
                self.assertEqual(is_gated, model_of[key].startswith("facebook/"))

    def test_the_object_embedder_never_outlives_the_detector(self):
        # It embeds the detector's crops, so with no `boxes` column there is nothing to
        # embed. Any plan carrying `obj` must carry `detector` too.
        for ungated_only in (False, True):
            plan = pl.stage_plan(ungated_only)
            if "obj" in plan:
                self.assertIn("detector", plan)
                self.assertLess(plan.index("detector"), plan.index("obj"))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
