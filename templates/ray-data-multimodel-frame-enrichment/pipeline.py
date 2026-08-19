#!/usr/bin/env python3
"""Four models on one GPU: promptable detection, two embedders, quality metrics.

    python pipeline.py --input /mnt/cluster_storage/frames --stub      # CPU, no weights
    HF_TOKEN=hf_... python pipeline.py --input /mnt/cluster_storage/frames

Fractional-GPU packing across heterogeneous stages: four models on one GPU, each holding a
different amount of VRAM and running at a different rate. `packing.py` does the arithmetic
without a GPU.

Every lever below carries its measured effect and what should change it.

RATES FOR THIS SHAPE, WITH THEIR SCOPE. Quote the scope with the rate.

    ~49.7 rows/s    end to end, fixed 131k x 2 confirmation run on the source fleet
    ~88 rows/s      active compute only; a watcher bug inserted 3h15m of idle, and the
                    output prefix carried ~321k duplicate rows from earlier runs
    209.999 rows/s  extraction only, no detector
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

# Keep `import ray` inside `build()`. The stage classes are plain callables, and the tests
# exercise them with no Ray installed.

# --------------------------------------------------------------------------------------
# Model set. All public. The two Meta repos are GATED and their terms are accepted per
# account, so HF_TOKEN comes from the reader's environment. Do not add a shared organisation
# token: a service account cannot agree to SAM 3's licence for someone else. CI runs
# --ungated-only.
# --------------------------------------------------------------------------------------
DETECTOR_MODEL = os.environ.get("DETECTOR_MODEL", "facebook/sam3")
OBJECT_EMBED_MODEL = os.environ.get("OBJECT_EMBED_MODEL", "facebook/dinov3-vitl16-pretrain-lvd1689m")
IMAGE_EMBED_MODEL = os.environ.get("IMAGE_EMBED_MODEL", "google/siglip2-base-patch16-224")

# The prompts the detector grounds against. Built once per actor, never per batch: 32 images
# x 17 labels settled at ~2.2 s per actor once precomputed.
#
# Each label must be a concrete noun phrase. PCS grounds noun phrases; abstractions ground to
# nothing. Measured on an L4 against make_fixture.py's fixture, boxes per frame at the shipped
# threshold of 0.4:
#
#     object          0 0 0 0        rectangle       1 0 0 3
#     shape           0 0 0 0        bright square   1 1 1 2
#     region          0 0 0 0
#
# `object,shape,region` was this file's default until that was run, and it returns NOTHING
# at every threshold down to 0.05. The same model on the same GPU finds 2 cats, 2 remote
# controls and 1 couch in a COCO photograph at the same threshold, which is the control
# that says the model and the call are fine and the words were wrong.
#
# So: if you change the imagery, re-check the labels against it. A label that grounds to
# nothing still costs a full forward pass per batch and returns no detections, which reads
# downstream as an empty frame.
LABELS = [s for s in os.environ.get("LABELS", "bright square,rectangle").split(",") if s]

# --------------------------------------------------------------------------------------
# Levers
# --------------------------------------------------------------------------------------

# Read CPU. A divisor, not a budget. At 0.1 on 96 cores the read stage opened ~950 slots and
# one 64-row block took 430 s. The value is fleet-specific; the direction is what transfers.
DOWNLOAD_NUM_CPUS = float(os.environ.get("DOWNLOAD_NUM_CPUS", "1.0"))

# Detector. The throughput floor at ~13 images/s per actor with all SMs busy. Raising the
# batch does not raise throughput; buy parallelism with actors. Raise the batch only if a
# profile shows the SMs idle.
DETECTOR_ACTORS = int(os.environ.get("DETECTOR_ACTORS", "10"))
DETECTOR_GPU = float(os.environ.get("DETECTOR_GPU", "0.2"))
DETECTOR_BATCH = int(os.environ.get("DETECTOR_BATCH", "4"))

# torch.compile mode. `reduce-overhead` was measured and rejected: its CUDA graphs produce
# overwritten-output failures inside Ray actors. Reproduce that before re-enabling it.
TORCH_COMPILE = os.environ.get("TORCH_COMPILE", "default")

# Embedders. Measured per-actor VRAM on the source shape: object embedding 7.65 GiB, image
# embedding 0.96 GiB. Same GPU fraction, 8x the memory. The fraction is admission control and
# does not cap VRAM; see packing.py.
OBJ_EMB_GPU = float(os.environ.get("OBJ_EMB_GPU", "0.05"))
IMG_EMB_GPU = float(os.environ.get("IMG_EMB_GPU", "0.05"))
EMB_BATCH = int(os.environ.get("EMB_BATCH", "32"))
EMB_ACTORS = int(os.environ.get("EMB_ACTORS", "4"))

# Metrics. Sub-batch is decoupled from the Ray batch. Variable per-image shapes create VRAM
# shape-buckets: a Ray batch of 192-256 OOMed at 30+ GiB per actor, and Ray batch 32 with a
# sub-batch of 4 landed at 4.95 GiB. Raise the sub-batch only on fixed-shape input.
METRICS_ACTORS = int(os.environ.get("METRICS_ACTORS", "8"))
METRICS_GPU = float(os.environ.get("METRICS_GPU", "0.02"))
METRICS_BATCH = int(os.environ.get("METRICS_BATCH", "32"))
METRICS_SUBBATCH = int(os.environ.get("METRICS_SUBBATCH", "4"))

STUB = os.environ.get("STUB", "") == "1"

# The compiled dependency closure that ships beside this file. Resolved from `__file__`, so it
# is found whatever the cwd.
LOCK = Path(__file__).resolve().parent / "python_depset.lock"


def runtime_env(stub: bool = False) -> dict:
    """The runtime env for the stage actors.

    Keep this. The driver's install does not reach the actors: the compute config pins the
    head unschedulable (`resources: {CPU: 0}`), so every `map_batches` actor runs on a GPU
    worker that never ran `uv pip install`, and `--system` installs do not propagate. Without
    this the models are importable only where they are never loaded.

    Hand the actors the same lock the driver installed. A second pin list here would drift.

    `--stub` needs none of it: the stub branches touch numpy, which the base image ships.
    """
    if stub:
        return {}
    if not LOCK.is_file():
        raise SystemExit(
            f"{LOCK} is missing, so the stage actors would run on whatever the base image\n"
            "ships -- which is neither torch nor transformers. Recompile it from the repo\n"
            "root with `./scripts/depsets/update_deps.sh --name "
            "ray_data_multimodel_frame_enrichment_depset_<ray>_<py>`,\n"
            "or run with --stub, which needs no weights and no added dependencies."
        )
    return {"pip": str(LOCK)}


def _frames(batch: dict) -> list[np.ndarray]:
    """Decode the fixed-size blob column into frames. One reshape, no per-row copy."""
    out = []
    for raw, w, h in zip(batch["image"], batch["width"], batch["height"]):
        arr = np.frombuffer(raw, dtype=np.uint8)
        out.append(arr.reshape(int(h), int(w), 3))
    return out


class Detector:
    """Promptable detection. The batch dimension collapses; the label dimension does not.

    Promptable Concept Segmentation takes one noun phrase and returns every instance of it, so
    `L` labels cost `L` forward passes. `Sam3Processor` hands `text` straight to its tokenizer
    with `padding="max_length", max_length=32`, so `list[str]` is one prompt per image.

    Do not write `text=[labels] * len(frames)`. That is `list[list[str]]`, HuggingFace's
    pre-tokenized-words form, which needs `is_split_into_words=True` that the processor never
    passes. Measured on real weights, the fast tokenizer rejects it:

        TypeError: TextEncodeInput must be Union[TextInputSequence,
                   Tuple[InputSequence, InputSequence]]

    So: `L` passes of batch `B`, with one host transfer for the whole batch at the end.
    """

    def __init__(self) -> None:
        self.stub = STUB
        if self.stub:
            return
        import torch
        from transformers import Sam3Model, Sam3Processor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Sam3Processor.from_pretrained(DETECTOR_MODEL)
        model = Sam3Model.from_pretrained(DETECTOR_MODEL, dtype=torch.bfloat16)
        self.model = model.to(self.device).eval()
        if TORCH_COMPILE:
            self.model = torch.compile(self.model, mode=TORCH_COMPILE)
        # One prompt per pass, resolved once per actor.
        self.prompts = list(LABELS)

    def __call__(self, batch: dict) -> dict:
        frames = _frames(batch)
        if self.stub:
            # Deterministic boxes, so the DAG and the downstream shapes run with no weights.
            # Not a source of any measured number.
            #
            # Keep the count varying per frame, including zero. A constant count cannot
            # distinguish a correct per-frame scatter downstream from a wrong one, and zero
            # exercises the degenerate-input path.
            boxes = []
            for f in frames:
                n = int(np.frombuffer(f[:1, :1].tobytes(), dtype=np.uint8)[0]) % 4
                h, w = f.shape[0], f.shape[1]
                boxes.append(
                    np.array(
                        [[4 + 8 * k, 4 + 8 * k, min(4 + 8 * k + 24, w), min(4 + 8 * k + 24, h)]
                         for k in range(n)],
                        dtype=np.int32,
                    ).reshape(n, 4)
                )
        else:
            torch = self.torch
            # One pass per label, each over the whole microbatch. See the class docstring
            # for why the label dimension cannot join the batch dimension here.
            per_frame: list[list] = [[] for _ in frames]
            for prompt in self.prompts:
                inputs = self.processor(
                    images=frames, text=[prompt] * len(frames), return_tensors="pt"
                ).to(self.device)
                with torch.inference_mode():
                    out = self.model(**inputs)
                # `original_sizes` is what the processor recorded before resizing, so it is
                # the mapping back. Deriving (h, w) from the frames by hand duplicates it.
                sizes = inputs.get("original_sizes")
                if hasattr(sizes, "tolist"):
                    sizes = sizes.tolist()
                results = self.processor.post_process_instance_segmentation(
                    out, threshold=0.4, target_sizes=sizes
                )
                for i, r in enumerate(results):
                    per_frame[i].append(r["boxes"].to(torch.int32))

            # One transfer per batch, at the end, across every label. Device syncs in this
            # path were ~2000 per batch; removing them was the largest single contributor to
            # the 6.64 -> 39.65 rows/s move, confounded with Arrow and packing changes that
            # landed the same days. Do not put `.cpu()` inside the loop: that is B x L
            # transfers.
            joined = [torch.cat(bs, dim=0) if bs else torch.zeros((0, 4), dtype=torch.int32)
                      for bs in per_frame]
            counts = [int(t.shape[0]) for t in joined]
            flat = torch.cat(joined, dim=0).cpu().numpy() if joined else np.zeros((0, 4), np.int32)
            boxes, at = [], 0
            for n in counts:
                boxes.append(flat[at:at + n].reshape(n, 4))
                at += n
        return {
            "frame_id": batch["frame_id"],
            "image": batch["image"],
            "width": batch["width"],
            "height": batch["height"],
            "boxes": boxes,
            "n_detections": np.array([len(b) for b in boxes], dtype=np.int32),
        }


class ObjectEmbedder:
    """Embed each detected crop. The VRAM hog: 7.65 GiB per actor on the source shape."""

    def __init__(self) -> None:
        self.stub = STUB
        if self.stub:
            self.dim = 16
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(OBJECT_EMBED_MODEL)
        model = AutoModel.from_pretrained(OBJECT_EMBED_MODEL, dtype=torch.bfloat16)
        # Raise on unloaded weights. A random-initialization warning is logged and ignored,
        # and the model then emits garbage at full speed.
        if getattr(model.config, "_name_or_path", "") == "":
            raise RuntimeError(
                f"{OBJECT_EMBED_MODEL} did not load pretrained weights. Refusing to emit "
                "embeddings from a randomly initialized model."
            )
        self.model = model.to(self.device).eval()
        self.dim = int(model.config.hidden_size)

    def __call__(self, batch: dict) -> dict:
        frames = _frames(batch)
        # Crop extraction is CPU work and stays on the CPU side of the actor.
        #
        # `owner` records which frame each crop came from. The batch is embedded in one
        # forward pass, but the output column is per row: write the per-frame count, not the
        # batch total. Getting that wrong reports eight detections for each of eight frames
        # that had one apiece, and reads like a formatting difference.
        crops: list[np.ndarray] = []
        owner: list[int] = []
        for i, (frame, boxes) in enumerate(zip(frames, batch["boxes"])):
            for x0, y0, x1, y1 in np.asarray(boxes, dtype=np.int32):
                crop = frame[max(0, y0) : max(1, y1), max(0, x0) : max(1, x1)]
                if crop.size:
                    crops.append(crop)
                    owner.append(i)

        if not crops:
            # A batch with no detections must not raise. Building a fixed-shape tensor array
            # from an empty crop array is the degenerate-input failure.
            vecs = np.zeros((0, self.dim), dtype=np.float32)
        elif self.stub:
            vecs = np.stack([
                np.full(self.dim, float(c.mean()), dtype=np.float32) for c in crops
            ])
        else:
            torch = self.torch
            inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.model(**inputs)
            vecs = out.pooler_output.to(torch.float32).cpu().numpy()

        # Scatter back to per-frame. Ragged by construction: a frame with three detections
        # carries three vectors and a frame with none carries an empty (0, dim) array,
        # which is the shape the downstream join has to expect.
        per_frame = [np.zeros((0, self.dim), dtype=np.float32) for _ in frames]
        counts = np.zeros(len(frames), dtype=np.int32)
        for i in range(len(frames)):
            rows = [v for v, o in zip(vecs, owner) if o == i]
            if rows:
                per_frame[i] = np.stack(rows)
                counts[i] = len(rows)

        return {
            "frame_id": batch["frame_id"],
            "image": batch["image"],
            "width": batch["width"],
            "height": batch["height"],
            "n_detections": batch["n_detections"],
            "object_embeddings": per_frame,
            "object_embedding_count": counts,
            "object_embedding_dim": np.array([self.dim] * len(frames), dtype=np.int32),
        }


class ImageEmbedder:
    """Whole-frame embedding. The cheap one: 0.96 GiB per actor, same GPU fraction."""

    def __init__(self) -> None:
        self.stub = STUB
        if self.stub:
            self.dim = 8
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(IMAGE_EMBED_MODEL)
        self.model = AutoModel.from_pretrained(
            IMAGE_EMBED_MODEL, dtype=torch.bfloat16
        ).to(self.device).eval()
        self.dim = int(self.model.config.vision_config.hidden_size)

    def __call__(self, batch: dict) -> dict:
        frames = _frames(batch)
        if self.stub:
            vecs = np.stack([np.full(self.dim, float(f.mean()), dtype=np.float32) for f in frames])
        else:
            torch = self.torch
            inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                feats = self.model.get_image_features(**inputs)
            # On transformers 5.15.0 this returns a `BaseModelOutputWithPooling`, not a
            # tensor; a bare `.to(...)` raises AttributeError. Handle both: the pin is a
            # floor, and this return type has changed once.
            if not isinstance(feats, torch.Tensor):
                pooled = getattr(feats, "pooler_output", None)
                if pooled is None:
                    raise RuntimeError(
                        f"{IMAGE_EMBED_MODEL}: get_image_features returned "
                        f"{type(feats).__name__} with no pooler_output; this pipeline needs "
                        "one vector per image and will not guess which field that is."
                    )
                feats = pooled
            vecs = feats.to(torch.float32).cpu().numpy()
        out = dict(batch)
        out["image_embedding"] = list(vecs)
        return out


class Metrics:
    """No-reference quality metrics, sub-batched away from the Ray batch size."""

    def __init__(self) -> None:
        self.stub = STUB
        if self.stub:
            return
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, batch: dict) -> dict:
        frames = _frames(batch)
        scores = []
        for i in range(0, len(frames), METRICS_SUBBATCH):
            chunk = frames[i : i + METRICS_SUBBATCH]
            if self.stub:
                scores.extend(float(np.var(c) / 255.0) for c in chunk)
                continue
            torch = self.torch
            # One host-to-device transfer per sub-batch, then vectorized. A per-image path
            # with CPU round-trips and scalar reads cost ~2000 device syncs per batch.
            t = torch.from_numpy(np.stack(chunk)).to(self.device, non_blocking=True)
            t = t.to(torch.float32) / 255.0
            gray = t.mean(dim=-1)
            dx = gray[:, :, 1:] - gray[:, :, :-1]
            dy = gray[:, 1:, :] - gray[:, :-1, :]
            sharp = (dx.pow(2).mean(dim=(1, 2)) + dy.pow(2).mean(dim=(1, 2))).sqrt()
            scores.extend(sharp.cpu().numpy().astype(float).tolist())
        out = {k: v for k, v in batch.items() if k != "image"}
        out["sharpness"] = np.array(scores, dtype=np.float32)
        return out


# The stages, in order, as (key, class name, gated). Kept out of `build()` and free of any Ray
# import, so the selection is testable with no cluster.
#
# `gated` means the stage's weights need terms accepted per account. It follows the model
# repository and is not a configuration knob.
STAGES = [
    ("detector", "Detector", True),
    ("obj", "ObjectEmbedder", True),
    ("img", "ImageEmbedder", False),
    ("metrics", "Metrics", False),
]


def stage_plan(ungated_only: bool = False) -> list[str]:
    """The stage keys to build, in order.

    `ungated_only` drops SAM 3 and DINOv3 and leaves SigLIP2 plus the weightless metrics
    stage: two models on one card with real weights, which is not the four-model claim.

    The object embedder goes with the detector. It embeds the detector's crops, so with no
    `boxes` column there is nothing to embed.
    """
    return [key for key, _cls, gated in STAGES if not (ungated_only and gated)]


def build(input_path: str, stub: bool = False, ungated_only: bool = False):
    """The DAG. Four stages by default; two under `ungated_only`.

    `stub` drops the GPU reservations along with the weights. `num_gpus=0.2` on a box with no
    GPU does not fail: the actor pool never admits, the dataset never produces a batch, and
    the run hangs.

    `ungated_only` keeps the GPU and the real weights and drops the gated stages, so CI can
    run without a token. See `stage_plan`.
    """
    gpu = {"detector": DETECTOR_GPU, "obj": OBJ_EMB_GPU,
           "img": IMG_EMB_GPU, "metrics": METRICS_GPU}
    actors = {"detector": DETECTOR_ACTORS, "obj": EMB_ACTORS,
              "img": EMB_ACTORS, "metrics": METRICS_ACTORS}
    batch = {"detector": DETECTOR_BATCH, "obj": EMB_BATCH,
             "img": EMB_BATCH, "metrics": METRICS_BATCH}
    cls = {"detector": Detector, "obj": ObjectEmbedder,
           "img": ImageEmbedder, "metrics": Metrics}
    if stub:
        gpu = dict.fromkeys(gpu, 0.0)
        # One actor per stage: the stub asserts the DAG and the shapes, and a laptop
        # scheduling 30 actors to do that teaches nothing extra.
        actors = dict.fromkeys(actors, 1)

    # Use `compute=ActorPoolStrategy(size=n)`, not `concurrency=(n, n)`. The tuple form is
    # deprecated as of Ray 2.51 and warns once per stage on 2.57. `size=n` is the spelling of
    # min_size == max_size, and this template needs a fixed pool: the autoscaler parks below
    # the GPU count on whole-GPU actors.
    from ray.data import ActorPoolStrategy

    def pool(stage: str) -> dict:
        kw: dict = {"num_cpus": 0, "compute": ActorPoolStrategy(size=actors[stage])}
        if gpu[stage]:
            kw["num_gpus"] = gpu[stage]
        else:
            # num_cpus=0 with no GPU request is also unschedulable, so the stub takes a core.
            kw["num_cpus"] = 1
        return kw

    import ray

    ds = ray.data.read_parquet(
        input_path, ray_remote_args={"num_cpus": DOWNLOAD_NUM_CPUS}
    )
    for stage in stage_plan(ungated_only):
        ds = ds.map_batches(cls[stage], batch_size=batch[stage], **pool(stage))
    return ds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--stub", action="store_true",
                    help="no weights, no GPU: exercises the DAG and every shape")
    ap.add_argument("--ungated-only", action="store_true",
                    help="real GPU and real weights, but only the stages whose models are "
                         "not gated (SigLIP2 + metrics). Needs no HF_TOKEN. This is TWO-model "
                         "co-residency, not the four-model claim")
    # Stats are on by default. Spell the flag as the one that turns them off: an
    # `action="store_true", default=True` flag cannot change anything.
    ap.add_argument("--no-stats", action="store_false", dest="stats",
                    help="skip the per-stage Ray Data timings printed after the run")
    args = ap.parse_args(argv)

    if args.stub:
        os.environ["STUB"] = "1"
        globals()["STUB"] = True

    if not args.stub and not args.ungated_only and not os.environ.get("HF_TOKEN"):
        raise SystemExit(
            "Set HF_TOKEN before running. Two of the three models are GATED, so this needs\n"
            "your own Hugging Face token and your own acceptance of their terms -- gating is\n"
            "per ACCOUNT, not per token, so no token setting substitutes for the clicks:\n"
            "\n"
            f"  1. accept the terms at https://huggingface.co/{DETECTOR_MODEL}\n"
            f"  2. accept the terms at https://huggingface.co/{OBJECT_EMBED_MODEL}\n"
            "  3. create a token (a fine-grained READ token is enough) at\n"
            "     https://huggingface.co/settings/tokens and export it as HF_TOKEN\n"
            "\n"
            "Approval is usually immediate but is not guaranteed to be. Until then, or if you\n"
            "just want to see the pipeline run, `--stub` exercises every stage and every shape\n"
            "with no weights and no GPU."
        )

    import ray

    ray.init(ignore_reinit_error=True, runtime_env=runtime_env(stub=args.stub))
    ds = build(args.input, stub=args.stub, ungated_only=args.ungated_only)
    if args.ungated_only:
        print(f"UNGATED-ONLY: {len(stage_plan(True))} of {len(STAGES)} stages "
              f"({', '.join(stage_plan(True))}). The two gated models are absent, so this "
              f"run says nothing about the four-model claim.")

    t0 = time.perf_counter()
    if args.output:
        ds.write_parquet(args.output)
        rows = ds.count()
    else:
        rows = sum(len(b["frame_id"]) for b in ds.iter_batches(batch_format="numpy"))
    elapsed = time.perf_counter() - t0

    print(f"\n{rows} rows in {elapsed:.1f}s = {rows / elapsed:.2f} rows/s")
    print("State the unit and the scope. rows/s, images/s and detections/s differ here by "
          "the detections per frame, and the published figures for this shape differ by "
          "which one they meant.")
    if args.stats:
        print("\n" + ds.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
