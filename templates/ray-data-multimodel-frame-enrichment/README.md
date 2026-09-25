# Multi-model frame enrichment on one GPU

An enrichment pass over video frames often runs several models per frame. Each holds a
different amount of VRAM and runs at a different rate, so a whole GPU per model leaves most of
each card unused, and packing them by `num_gpus` fractions alone can run a card out of memory.

Ray Data runs each stage as its own actor pool, with its own GPU fraction, batch size and pool
size, and streams batches through all the stages at once. Here four stages share one L4: SAM 3
detects objects from text prompts, DINOv3 embeds each detection, SigLIP2 embeds the whole
frame, and a metric with no model scores sharpness. Running all four co-resident, together on
the card, had at least 26.5% higher end-to-end throughput than running them one after another
(96 frames, three timed runs per arm; details under "Co-resident vs serial" below).

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

Approval is usually immediate but can take longer. Until then, everything before the gated
section runs.

## Check the arithmetic and the stages, no GPU needed

The cells before the install need only the base image, which ships numpy, pyarrow and Ray but
no torch, torchvision or transformers (measured on the 2.57.0 image; the 2.58.0 package freeze
lists the same).

The first cell defines `run()`, which runs a template script and raises if it fails, then runs
the unit tests. Each test file ends with `OK`.


```python
import os
import subprocess
import sys


def run(*args):
    """Run a template script with this kernel's Python, and raise if it exits non-zero.

    Keep this in place of `!python script.py`. IPython's `!` does not raise on a non-zero exit
    (measured: the next cell runs and papermill exits 0), so a failed stage would go unnoticed.
    """
    subprocess.run([sys.executable, *args], check=True)


# No GPU, weights or cluster needed. If these fail, stop here.
run("tests/test_packing.py")
run("tests/test_pipeline.py")
```

## Two limits per GPU: the fraction and the memory

`num_gpus=0.02` lets Ray place 50 actors of a stage on one GPU, because 50 fractions sum to 1.0.
It does not cap their memory, so if they do not fit, CUDA runs out. A stage's per-GPU capacity
is the smaller of two numbers:

```
by fraction :  floor(1 / num_gpus)                  actors per GPU
by VRAM     :  floor(vram_per_gpu / vram_per_actor)  actors per GPU
```

`packing.py` computes both. Its default table is the shipped configuration: `pipeline.py`'s
default actor counts and fractions, which come from the production workload this template was
derived from. The per-actor VRAM is that workload's, on its own models and the 48 GiB card it
was tuned on.

| stage | `num_gpus` | GiB/actor | by fraction | by VRAM | binds |
|---|---|---|---|---|---|
| detector | 0.20 | 9.60 · unmeasured | 5 | 5 | both |
| object embedder | 0.05 | 7.65 | 20 | 6 | VRAM |
| image embedder | 0.05 | 0.96 | 20 | 50 | fraction |
| metrics | 0.02 | 2.80 | 50 | 17 | VRAM |

Read only the fractions and you provision three times as many object-embedder actors as fit.
The two embedders take the same fraction and differ 8x in memory. The detector's 9.60 is its
fraction share of the card, unmeasured, and `packing.py` prints `(UNMEASURED)` beside it.

The next cell prints this table and the total for one actor of each stage, 21.01 GiB of 48 GiB.


```python
# The shipped configuration on a 48 GiB card.
run("packing.py")
```

Per-stage capacity does not say whether one actor of every stage fits on one card together.
`packing.py` sums them:

| | one actor of each stage | card | headroom |
|---|---|---|---|
| `--stages shipped` (that workload's models) | 21.01 GiB | 48 GiB | 56% |
| `--stages measured` (SAM 3, DINOv3, SigLIP2) | 5.53 GiB | 22.03 GiB L4 | 75% |

The measured row is this template's models, from two probe jobs on a g6 L4. An L4 has 24 GB,
which is 22.35 GiB, and torch reports 22.03 GiB usable, so pass `--vram 22.03`. On an L4 the
shipped set would leave 5% free. `packing.py` flags anything under 10%, the band where the
shipped workload hit its batch-192 OOM. Run `python packing.py --vram 22.03` to see the shipped
actor counts over-commit an L4.

The object embedder's 0.59 GiB was measured at one crop per frame. Its memory grows with
detections, so treat it as a floor for a nearly empty batch, not a budget for a busy one.

The next cell checks the measured table against an L4 with `--strict`, which exits 1 if the
stages do not fit together or a stage is over-committed. Expect `co-resident, one actor of each
stage: 5.53 GiB of 22.03 GiB`. The last line also mentions the cost ordering, which `packing.py`
checks only on the shipped table.


```python
# This template's models on an L4. Exits 1 if they do not fit.
run("packing.py", "--stages", "measured", "--vram", "22.03", "--strict")
```

## Generate frames and run the DAG without weights

`make_fixture.py` writes synthetic frames with bright shapes on a textured background, so there
is no download and no dataset licence. Environment variables set the frame count and size.
Shrink those to go faster, and keep all four stages.

By default the next cell writes 48 frames of 640x480 into four Parquet files and prints
`48 frames, 4 file(s), 0.92 MB per frame`.


```python
# Defaults are the demo size. Set these environment variables to change the run without
# editing the notebook.
FIXTURE = os.environ.get("FIXTURE_DIR", "/mnt/cluster_storage/frames")
OUTPUT = os.environ.get("OUTPUT_DIR", "/mnt/cluster_storage/frames-enriched")
FRAMES = os.environ.get("FRAMES", "48")
FILES = os.environ.get("FILES", "4")
WIDTH = os.environ.get("WIDTH", "640")
HEIGHT = os.environ.get("HEIGHT", "480")

run("make_fixture.py", "--out", FIXTURE, "--frames", FRAMES, "--files", FILES,
    "--width", WIDTH, "--height", HEIGHT)
```

`--stub` runs every stage on deterministic fake detections and embeddings, with no weights, no
token and no GPU, so it tests the DAG and every output shape. It drops the GPU reservations too:
on a node with no GPU, `num_gpus=0.2` hangs instead of failing.

Expect Ray Data's execution plan to list the four `MapBatches` operators, then a `48 rows in ...`
line and per-operator stats. The rate means nothing here.


```python
run("pipeline.py", "--input", FIXTURE, "--stub")
```

## Install the dependencies

`python_depset.lock` is `requirements.txt` compiled against a freeze of the base image. Install
the lock, not `requirements.txt`, so the driver and the actors get the same resolution.


```python
# This runs uv, not this kernel's Python, so it does not go through run(). Keep check=True: a
# half-finished install must fail the cell.
#
# Keep the requirements flag and the lock filename adjacent in this string. The repo's
# check-dep-delivery hook looks for the flag, whitespace and the filename; an argument list puts
# a quote and a comma between them, and the hook then reports that nothing installs the lock.
INSTALL = (
    "uv pip install -r python_depset.lock --system --no-deps --no-cache-dir "
    "--index-strategy unsafe-best-match"
)
subprocess.run(INSTALL.split(), check=True)
```

That install reaches the driver, on the head node. The stages run as Ray Data actors on GPU
workers, where a `uv pip install --system` does not propagate, so `pipeline.py` hands them the
same lock with `ray.init(runtime_env={"pip": .../python_depset.lock})`. Delete that line and every
stage fails on `import torch` while the driver looks fine.

The next cell checks the driver's install. The transformers package imports lazily, so the
cell touches a SAM 3 class to force a real import. Expect
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

SigLIP2 is not gated and the metrics stage has no weights, so these two stages run on the GPU
with no token. A pass shows that the GPU worker is up, the runtime env reached the actors, and
two stages with fractional reservations share one card. It does not test the four-stage fit,
which rests on the arithmetic above and your own run below.

Expect `UNGATED-ONLY: 2 of 4 stages (img, metrics)` and a rows/s line that, at this size, is
mostly model loading. The check cell after it prints
`ungated pair: 48 rows, image embedding width {768}, all sharpness finite`.


```python
UNGATED_OUT = os.environ.get("UNGATED_OUTPUT_DIR", OUTPUT + "-ungated")

# Real GPU and SigLIP2 weights, no token. Keep --ungated-only an explicit flag; do not infer it
# from a missing HF_TOKEN.
run("pipeline.py", "--input", FIXTURE, "--output", UNGATED_OUT, "--ungated-only")
```


```python
import ray

ds = ray.data.read_parquet(UNGATED_OUT)
rows = ds.take_all()
assert rows, "the ungated run produced no rows"

# Checks for the two ungated stages: a real SigLIP2 width (the stub's is 8), finite sharpness
# scores, and no image blob left in the output.
import math

widths = {len(r["image_embedding"]) for r in rows}
assert widths and min(widths) >= 256, f"image embedding width {widths} looks like a stub"
assert all(math.isfinite(float(r["sharpness"])) for r in rows), "a sharpness score is not finite"
assert "image" not in rows[0], "the blob column must not survive to the output"

print(f"ungated pair: {len(rows)} rows, image embedding width {widths}, "
      f"all sharpness finite")
```

## Run all four stages with your token

Every code cell from here on needs `HF_TOKEN`. This repo's CI has no token and skips them; they
carry the `skip-in-ci` tag.

The first cell checks access before any download. It confirms that the token authenticates,
then fetches one file from each gated repo. Expect `gated access ok as '<your username>'`. On
failure it names the cause. A `401` from whoami means Hugging Face does not recognise the token,
and a new token fixes it. A `403 ... not in the authorized list` on a file means the account
has not accepted the terms, and no new token will help.


```python
# Check gated access before loading anything. Otherwise a missing acceptance shows up as a 403
# traceback several GB into a model download, naming neither the cause nor the remedy.
#
# A file fetch answers the access question and the metadata endpoint does not:
# `/api/models/<repo>` returns 200 anonymously for both repos.
import json, os, sys, urllib.error, urllib.request

GATED = ["facebook/sam3", "facebook/dinov3-vitl16-pretrain-lvd1689m"]
token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    sys.exit("HF_TOKEN is empty. Accept the terms on both gated model pages with your own\n"
             "Hugging Face account, create a token (a fine-grained READ token is enough),\n"
             "and export it as HF_TOKEN. Nobody can do this step for you: SAM 3 and DINOv3\n"
             "are licensed per ACCOUNT.")


def get(url):
    """Return ('http', status, body) or ('unreachable', message, None).

    A network failure gets its own kind, so a proxy or DNS error is not reported as a denial.
    Read the body inside the `with`: the response is closed after it.
    """
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

# whoami first. A failure here means the token did not authenticate, which a corrected token
# fixes. A 403 on a file below means the token works and the account has not accepted the terms,
# which no token change fixes. Measured: with a bogus token every request returns 401.
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

### The four-stage run

The next cell stops if `HF_TOKEN` is unset; `pipeline.py` makes the same check before any
download. The cell after it sets one actor per stage, which fits one L4, and runs all four
stages. The first run downloads the SAM 3 and DINOv3 weights.


```python
assert os.environ.get("HF_TOKEN"), (
    "Set HF_TOKEN first: two of the three models are gated, and gating is per ACCOUNT. "
    "Accept the terms on both model pages listed at the top of this notebook, then create "
    "a fine-grained read token. `--stub` above needs none of it."
)
```


```python
# One actor per stage fits one L4. pipeline.py's defaults (10 detectors at 0.2 GPU each) need
# more than one GPU; see `python packing.py`. setdefault, so an exported value wins.
for key, value in dict(
    DETECTOR_ACTORS="1", EMB_ACTORS="1", METRICS_ACTORS="1",
    DETECTOR_BATCH="2", EMB_BATCH="8", METRICS_BATCH="8", METRICS_SUBBATCH="2",
).items():
    os.environ.setdefault(key, value)

run("pipeline.py", "--input", FIXTURE, "--output", OUTPUT)
```

The next cell checks the answers: each frame's embedding count matches its detections, the
object embeddings have a real DINOv3 width, and the detector fired on at least one frame. A run
that detects nothing still completes and passes the first two checks. The cell prints the row
count, the total detections and the embedding width.


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

## Co-resident vs serial on one L4

Measured with this template's models on one L4: 96 frames at 640x480, one actor per stage,
end-to-end wall clock including warm model loads.

| arm | rows/s, 3 timed runs | spread |
|---|---|---|
| co-resident: four stages sharing the card | 1.2619 / 1.2631 / 1.2792 | 1.4% |
| serial: each stage alone with the whole GPU, materialized between stages | 0.9882 / 0.9903 / 0.9979 | 1.0% |

Verdict: co-resident beats serial by at least 26.5%. Two arms count as separable when the worst
run of the faster arm beats the best run of the slower one (1.2619 > 0.9979), and the margin is
that gap, a lower bound. Ranges that overlap are not separable however far apart the means sit,
and an arm needs at least two runs.

This figure is the third of three rounds, and it excluded nothing. The first two read at least
27.7% and 27.9%, but only after a slow run was dropped post hoc; on their full data neither was
separable. Whether the margin holds at production scale, or with more actors per stage, is not
measured.

### Optional: measure it yourself

`measure_packing.py` needs one GPU, the gated weights and about 20 minutes. It warms the model
cache on the node that does the work, discards one untimed pass of each arm, and interleaves the
timed runs. It prints the verdict by the rule above and writes every run to a JSONL, failures
included. It uses the actor counts set above.


```python
# Commented out: about 20 minutes of GPU. It times the fixture at --input as generated above
# (FRAMES frames); its --frames flag does not change that. The table above used 96 frames.
#
# run("measure_packing.py", "--input", FIXTURE, "--frames", "96", "--runs", "3")
```

## Settings in pipeline.py

Each setting is an environment variable. The notes give the effect measured on the production
workload behind the shipped configuration, and what should change it.

| setting | default | effect |
|---|---|---|
| `DOWNLOAD_NUM_CPUS` | 1.0 | CPUs per read task, so read parallelism is cores divided by this: 0.1 on 96 cores opened ~950 read slots and one 64-row block took 430 s |
| `DETECTOR_BATCH` | 4 | the detector is the slowest stage, at ~13 images/s per actor with all SMs busy, and a larger batch does not raise its throughput; add actors instead |
| `TORCH_COMPILE` | `default` | `reduce-overhead` was measured and rejected: its CUDA graphs produce overwritten-output failures inside Ray actors |
| `METRICS_SUBBATCH` | 4 | separate from the Ray batch: a Ray batch of 192-256 OOMed at 30+ GiB per actor, and Ray batch 32 with sub-batch 4 used 4.95 GiB |
| `EMB_ACTORS` | 4 | a fixed pool, `min_size == max_size`: with an autoscaling pool of whole-GPU actors, Ray Data settled below the GPU count |

## Detector labels

`LABELS` defaults to `bright square,rectangle`, which suit the synthetic fixture. SAM 3's
Promptable Concept Segmentation takes one noun phrase and returns every instance of it, so each
label costs one detector pass per batch. Keep the list short, and use concrete noun phrases.
Measured on an L4 against this template's fixture, boxes per frame at the shipped threshold of
0.4:

| prompt | boxes per frame |
|---|---|
| `object` / `shape` / `region` | 0 0 0 0 |
| `rectangle` | 1 0 0 3 |
| `bright square` | 1 1 1 2 |

`object,shape,region` finds nothing at any threshold down to 0.05, while the same weights, GPU
and threshold find 2 cats, 2 remote controls and 1 couch in a COCO photograph. Re-check the
labels when you change the imagery: a label that finds nothing still costs a full pass, and the
frame reads as empty downstream.

Pass one prompt string per image, `text=[prompt] * len(frames)`. A list of lists is Hugging
Face's pre-tokenized-words form, which needs `is_split_into_words=True`, and `Sam3Processor`
never passes it. On real weights the fast tokenizer rejects it:

```
TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
```

## Throughput on the production workload

These rates come from the production workload behind the shipped configuration, with its own
models and fleet. They are not this template's L4 numbers, and a row there was not a frame
(39.65 rows/s was 9.9 images/s), so they do not compare with this pipeline's rows/s either.

| rate | scope |
|---|---|
| ~49.7 rows/s | end to end, fixed 131k x 2 confirmation run |
| ~88 rows/s | active compute only, excluding idle time a monitoring bug added; the output also held duplicate rows from earlier runs |
| 209.999 rows/s | extraction only, no detector |
| 39.65 rows/s = 9.9 images/s | 99.18K images in 2h47m, GPU 93-99% |

Quote ~49.7 rows/s as the end-to-end figure, with its unit and scope. On that workload, fixed
and repeated A/B runs came out lower than every figure from exploratory sweeps, and reversed two
conclusions, one of them a +3.4% win that had been written up.

## Models and licences

All three models are public. The two Meta models are gated; see the top of this notebook.

| stage | model | licence |
|---|---|---|
| detector | `facebook/sam3` | SAM License, gated |
| object embedder | `facebook/dinov3-vitl16-pretrain-lvd1689m` | DINOv3 License, gated |
| image embedder | `google/siglip2-base-patch16-224` | Apache-2.0 |
| metrics | no model: gradient energy in plain torch | n/a, no weights |

`NOTICE` summarises the licence terms and carries the "Built with DINOv3" acknowledgement the
DINOv3 License asks for.

Swapping a model changes the packing. A detector that is cheap next to the embedders inverts
the shape and the packing defaults stop transferring; `packing.py`'s ordering check fails if that
happens. YOLO is not used: Ultralytics YOLO is AGPL-3.0 and YOLOv9/YOLOR are GPL-3.0.

## Checks worth copying

- Per-frame scatter. The object embedder embeds a whole batch of crops in one forward pass, and
  its output column is per row. Writing the batch total into every row looks like a formatting
  difference and is a wrong answer.
- Degenerate input. A frame with no detections yields an empty `(0, dim)` array instead of
  raising.

`tests/` checks the arithmetic and each stage's output with no GPU or weights. No test asserts a
throughput ratio; that needs replicated runs (`measure_packing.py`).

## Files

```
packing.py           the arithmetic. no GPU, no weights, no cluster
measure_packing.py   whether co-residency pays. one GPU, the gated weights, ~20 min
pipeline.py          four stages, fractional GPUs, --stub for CPU, --ungated-only for SigLIP2 + metrics
make_fixture.py      synthetic frames, so there is no dataset licence to track
requirements.txt     what this template adds to the base image, and why each pin is there
python_depset.lock   the compiled closure of those pins, installed on the driver and handed
                     to the actors by pipeline.py
tests/               test_packing.py, test_pipeline.py; no GPU or weights needed
NOTICE               the two Meta licences and their obligations
```

The compute config, in the `anyscale/templates` repo under
`configs/ray-data-multimodel-frame-enrichment/`, gives a head that runs no tasks and one to two
GPU workers with one L4 each (24 GB, 22.03 GiB usable).
