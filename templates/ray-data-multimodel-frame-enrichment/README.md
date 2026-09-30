# Multi-model frame enrichment on one GPU

An enrichment pass over video frames often runs several models per frame. Each holds a
different amount of VRAM and runs at a different rate, so a whole GPU per model leaves most of
each card unused, and packing them by `num_gpus` fractions alone can run a card out of memory.

Ray Data runs each stage as its own actor pool, with its own GPU fraction, batch size and pool
size, and streams batches through all the stages at once. Here four stages share one L4: SAM 3
detects objects from text prompts, DINOv3 embeds each detection, SigLIP2 embeds the whole
frame, and a metric with no model scores sharpness. With 96 frames at 640x480 and one actor per
stage, running all four co-resident on the card had at least 26.5% higher end-to-end throughput
than running them one after another:

| arm | rows/s, 3 timed runs |
|---|---|
| co-resident: four stages sharing the card | 1.2619 to 1.2792 |
| serial: each stage alone with the whole GPU, one after another | 0.9882 to 0.9979 |

The 26.5% is the gap between the slowest co-resident run and the fastest serial run, so it is a
lower bound. Each run's wall clock includes loading the models from a warm cache; production
scale and more actors per stage are not measured.

You end with:

- a Parquet dataset with one row per frame: detection count, a DINOv3 embedding per detection,
  a SigLIP2 frame embedding and a sharpness score
- `packing.py`, which says whether a set of stages fits one GPU and which limit binds each stage
- `measure_packing.py`, which measures whether co-residency pays on your card

SAM 3 and DINOv3 are gated, so the full run needs your own Hugging Face token. Everything before
the install cell runs without a GPU or a token.

## Before you start: your Hugging Face token

Hugging Face grants access to SAM 3 and DINOv3 per account. Nobody can accept the terms for you,
so this template ships no shared token. Do this once:

1. accept the terms at <https://huggingface.co/facebook/sam3>
2. accept the terms at <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>
3. create a read token at <https://huggingface.co/settings/tokens>

Add the token as the environment variable `HF_TOKEN` in the workspace's Dependencies tab.
Saving restarts the workspace, and variables set there reach every node. The gated models load
on the GPU workers, which never see an `export` in a terminal or `os.environ` in this notebook.
Approval is usually immediate; until then, everything before the gated section runs.

## Check the packing, no GPU needed

This defines `run()`, which raises if a script fails, and runs the unit tests; each file ends
with `OK`.


```python
import os
import subprocess
import sys


def run(*args):
    """Run a template script with this kernel's Python, and raise if it exits non-zero.

    Keep this in place of `!python script.py`: IPython's `!` does not raise on a failure.
    """
    subprocess.run([sys.executable, *args], check=True)


run("tests/test_packing.py")
run("tests/test_pipeline.py")
```

This prints the shipped configuration, `pipeline.py`'s defaults with the per-actor VRAM of the
production workload they came from, on its 48 GiB card; expect
`VRAM binds before the fraction does on: object-embedder, metrics.`


```python
run("packing.py")
```

This checks this template's own models on an L4 with `--strict`, which exits 1 unless they fit
together with 10% to spare and no stage is over-committed; expect
`co-resident, one actor of each stage: 5.53 GiB of 22.03 GiB`.


```python
run("packing.py", "--stages", "measured", "--vram", "22.03", "--strict")
```

`num_gpus` is admission control, not a memory cap. A stage fits the smaller of
`floor(1 / num_gpus)` and `floor(vram_per_gpu / vram_per_actor)` actors per GPU: in the shipped
configuration the object embedder's 0.05 admits 20 actors to a card that holds 6 of them at
7.65 GiB each. Each stage can fit alone while the set does not, so `packing.py` also sums one
actor of every stage.

Pass `--vram` the usable VRAM torch reports, 22.03 GiB on a 24 GB L4, not the card's rating.
Treat 5.53 GiB as a floor: the object embedder was measured at one crop per frame, and its
memory grows with detections. `pipeline.py`'s default actor counts over-commit an L4, so the
four-stage run below sets one actor per stage.

## Run the pipeline without weights

This writes 48 synthetic 640x480 frames, bright shapes on a textured background, into four
Parquet files; expect `48 frames, 4 file(s), 0.92 MB per frame`.


```python
# Demo-size defaults. Set these environment variables to change the run without editing here.
FIXTURE = os.environ.get("FIXTURE_DIR", "/mnt/cluster_storage/frames")
OUTPUT = os.environ.get("OUTPUT_DIR", "/mnt/cluster_storage/frames-enriched")
FRAMES = os.environ.get("FRAMES", "48")
FILES = os.environ.get("FILES", "4")
WIDTH = os.environ.get("WIDTH", "640")
HEIGHT = os.environ.get("HEIGHT", "480")

run("make_fixture.py", "--out", FIXTURE, "--frames", FRAMES, "--files", FILES,
    "--width", WIDTH, "--height", HEIGHT)
```

`--stub` runs all four stages on deterministic fake detections and embeddings, with no weights,
token or GPU; expect four `MapBatches` operators, then `48 rows in ...` at a rate that means
nothing here.


```python
run("pipeline.py", "--input", FIXTURE, "--stub")
```

## Install the dependencies

This installs `python_depset.lock`, the compiled closure of `requirements.txt`, on the driver;
`pipeline.py` hands the same lock to the actors on the GPU workers.


```python
# Keep check=True, so a half-finished install fails the cell. Keep the requirements flag
# and the lock filename adjacent in one string: check-dep-delivery cannot see them in a list.
INSTALL = (
    "uv pip install -r python_depset.lock --system --no-deps --no-cache-dir "
    "--index-strategy unsafe-best-match"
)
subprocess.run(INSTALL.split(), check=True)
```

This checks the driver's install; expect
`transformers 5.15.0 torch 2.9.1+cu129 torchvision 0.24.1+cu129 sam3 -> Sam3Config`.


```python
# Touch a class: `from transformers import Sam3Model` alone is lazy and succeeds with no torch.
import torch
import torchvision
import transformers
from transformers import Sam3Model

print("transformers", transformers.__version__, "torch", torch.__version__,
      "torchvision", torchvision.__version__, "sam3 ->", Sam3Model.config_class.__name__)
```

## Run the ungated stages on the GPU

SigLIP2 is not gated and the metrics stage has no weights, so this runs both on the GPU with no
token; expect `UNGATED-ONLY: 2 of 4 stages (img, metrics)`.


```python
UNGATED_OUT = os.environ.get("UNGATED_OUTPUT_DIR", OUTPUT + "-ungated")

run("pipeline.py", "--input", FIXTURE, "--output", UNGATED_OUT, "--ungated-only")
```

This checks the output; expect
`ungated pair: 48 rows, image embedding width {768}, all sharpness finite`.


```python
import ray

ds = ray.data.read_parquet(UNGATED_OUT)
rows = ds.take_all()
assert rows, "the ungated run produced no rows"

import math

widths = {len(r["image_embedding"]) for r in rows}
assert widths and min(widths) >= 256, f"image embedding width {widths} looks like a stub"
assert all(math.isfinite(float(r["sharpness"])) for r in rows), "a sharpness score is not finite"
assert "image" not in rows[0], "the blob column must not survive to the output"

print(f"ungated pair: {len(rows)} rows, image embedding width {widths}, "
      f"all sharpness finite")
```

## Run all four stages with your token

This checks that `HF_TOKEN` authenticates and can read both gated repos, before any download;
expect `gated access ok as '<your username>'`, or a message naming the cause and the fix.


```python
# Fetch a file: `/api/models/<repo>` returns 200 anonymously, so it cannot show access.
import json, os, sys, urllib.error, urllib.request

GATED = ["facebook/sam3", "facebook/dinov3-vitl16-pretrain-lvd1689m"]
token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    sys.exit("HF_TOKEN is empty. Accept the terms on both gated model pages with your own\n"
             "Hugging Face account, create a token (a fine-grained READ token is enough),\n"
             "and export it as HF_TOKEN. Nobody can do this step for you: SAM 3 and DINOv3\n"
             "are licensed per ACCOUNT.")


def get(url):
    """Return ('http', status, body), or ('unreachable', message, None) if no answer came."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return ("http", resp.status, resp.read())
    except urllib.error.HTTPError as exc:
        return ("http", exc.code, None)
    except Exception as exc:  # URLError, SSL, timeout, anything else
        return ("unreachable", f"{type(exc).__name__}: {exc}", None)


kind, val, body = get("https://huggingface.co/api/whoami-v2")
if kind == "unreachable":
    sys.exit(f"cannot reach huggingface.co: {val}. This is a network finding, not a gate one.")

if val != 200:
    sys.exit(
        f"HF_TOKEN did not authenticate: /api/whoami-v2 returned HTTP {val}.\n"
        "This is a token problem, not a gating one: Hugging Face does not know who you are,\n"
        "and a corrected token fixes it. Check that the token is a live read token and that\n"
        "it belongs to the account that accepted the terms."
    )
try:
    who = json.loads(body).get("name", "unknown") if body else "unknown (empty whoami body)"
except ValueError:
    who = "unknown (unparseable whoami response)"

denied, unreachable = [], []
for repo in GATED:
    kind, val, _ = get(f"https://huggingface.co/{repo}/resolve/main/config.json")
    if kind == "unreachable":
        unreachable.append(f"{repo} -> {val}")
    elif val != 200:
        denied.append((repo, val))

if unreachable:
    sys.exit(f"could not check gated access: {'; '.join(unreachable)}")
if denied:
    listing = ", ".join(f"{repo} -> HTTP {code}" for repo, code in denied)
    if {code for _, code in denied} == {403}:
        remedy = (
            "Accept the terms once on each model card with that account. The gate is\n"
            "per-account, not per-token, so no token change fixes it; a fine-grained read\n"
            "token is sufficient once the terms are accepted."
        )
    else:
        remedy = (
            "The token authenticated, so this is not the ordinary gate refusal (403). Read\n"
            "the status above before changing anything: 401 here after a successful whoami\n"
            "means the token lacks read scope on the repo, and anything else is Hugging Face\n"
            "telling you something this check does not model."
        )
    sys.exit(f"Hugging Face account '{who}' cannot read: {listing}.\n{remedy}")
print(f"gated access ok as '{who}'")
```

The next two cells stop if `HF_TOKEN` is unset, then run all four stages at one actor each,
which fits one L4; the first run downloads the SAM 3 and DINOv3 weights.


```python
assert os.environ.get("HF_TOKEN"), (
    "Set HF_TOKEN first: two of the three models are gated, and gating is per ACCOUNT. "
    "Accept the terms on both model pages listed at the top of this notebook, then create "
    "a fine-grained read token. `--stub` above needs none of it."
)
```


```python
# One actor per stage fits one L4; pipeline.py's defaults need more than one GPU. setdefault,
# so an exported value wins.
for key, value in dict(
    DETECTOR_ACTORS="1", EMB_ACTORS="1", METRICS_ACTORS="1",
    DETECTOR_BATCH="2", EMB_BATCH="8", METRICS_BATCH="8", METRICS_SUBBATCH="2",
).items():
    os.environ.setdefault(key, value)

run("pipeline.py", "--input", FIXTURE, "--output", OUTPUT)
```

This checks that each frame's embedding count matches its detections, that the embeddings have
a real DINOv3 width, and that the detector found something; it prints rows, detections and
embedding width.


```python
ds = ray.data.read_parquet(OUTPUT)
rows = ds.take_all()
assert rows, "the real run produced no rows"

mismatched = [r["frame_id"] for r in rows
              if int(r["n_detections"]) != int(r["object_embedding_count"])]
assert not mismatched, f"per-frame embedding count disagrees with detections: {mismatched[:5]}"

dims = {int(r["object_embedding_dim"]) for r in rows}
assert dims and min(dims) >= 256, f"embedding width {dims} looks like a stub, not DINOv3"

detected = sum(int(r["n_detections"]) for r in rows)
assert detected > 0, "the detector found nothing on any frame -- the fixture or the prompts are wrong"

print(f"{len(rows)} rows, {detected} detections, embedding dim {dims}")
```

## Measure co-residency on your card

`measure_packing.py` times both arms on one GPU with the gated weights in about 20 minutes, and
prints `SEPARABLE` with a margin only when their ranges do not overlap.


```python
# It times the fixture at --input as generated, FRAMES frames (the table at the top used
# 96); --frames does not change that. It uses the actor counts set for the four-stage run.
#
# run("measure_packing.py", "--input", FIXTURE, "--frames", "96", "--runs", "3")
```

## Models and licences

| stage | model | licence |
|---|---|---|
| detector | `facebook/sam3` | SAM License, gated |
| object embedder | `facebook/dinov3-vitl16-pretrain-lvd1689m` | DINOv3 License, gated |
| image embedder | `google/siglip2-base-patch16-224` | Apache-2.0 |
| metrics | no model: gradient energy in plain torch | n/a, no weights |

`NOTICE` summarises the licence terms and carries the "Built with DINOv3" acknowledgement the
DINOv3 License asks for.

## Next steps

- Your own frames: write Parquet as `make_fixture.py` does, one row per frame with `frame_id`,
  raw RGB `image` bytes, `width` and `height`, and pass its directory to `pipeline.py --input`.
- Labels: `LABELS` defaults to `bright square,rectangle`, which suit the fixture. Each label
  costs one detector pass per batch, so keep the list short, use concrete noun phrases, and
  re-check them when the imagery changes; `object`, `shape` and `region` find nothing here.
- Settings: each is an environment variable in `pipeline.py`, and the comment above it gives its
  measured effect on the production workload. Put your stages' measured VRAM in `packing.py`'s
  tables and run it before you scale.
- Dependencies: keep the `runtime_env` in `pipeline.py`. The driver's install does not reach
  the GPU workers, and without it every stage fails on `import torch`.
- Models: licence-check a substitute first; `NOTICE` says why YOLO is not used. A detector that
  is cheap next to the embedders inverts the shape, and `packing.py`'s ordering check fails if
  that happens.
- Scale: re-run `measure_packing.py` at your actor counts before counting on the margin.
