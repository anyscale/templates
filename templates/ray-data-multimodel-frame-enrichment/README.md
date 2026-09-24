# Multi-model frame enrichment on one GPU

Four stages on one GPU, three of them models: promptable detection, an object embedder, a
whole-frame embedder, and no-reference quality metrics. The work is in fitting all four onto
one card so none of them starves the others.

## Before you run it: your own token

Two of the three models are gated, and gating on Hugging Face is per account. No token setting
substitutes for accepting the terms yourself:

1. accept the terms at <https://huggingface.co/facebook/sam3>
2. accept the terms at <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>
3. create a token at <https://huggingface.co/settings/tokens> — a fine-grained read token is
   enough — and export it as `HF_TOKEN`

Approval on both is usually immediate, but it is an approval and not a download, so it can sit.
`--stub` needs none of it: all four stages and every output shape, with no weights, no token and
no GPU.

There is no shared token for this template, because nobody can accept those terms on your
behalf. CI runs the ungated half on a real GPU and the four-model cells are tagged out of it.

Two failure modes, and they look nothing alike. `401` means Hugging Face does not know who you
are: a missing or malformed token. `403 "you are not in the authorized list"` means your token is
fine and your account has not accepted the terms, and a new token will not fix it. The preflight
cell below reports which one it got.

## Why the install comes after the arithmetic

Everything above the install cell runs on the bare image, so it runs anywhere, including a
laptop with no GPU. Moving a torch-using cell above the install breaks that.

Measured on `anyscale/ray:2.57.0-py312-cu129`: it ships numpy, pyarrow and ray, and no torch,
torchvision or transformers. For `2.58.0-py312-cu129`, which this template runs on, the image's
package freeze lists the same; that is read from the freeze, not measured on a job. `-cu129` is
the CUDA runtime, not PyTorch.


```python
import os
import subprocess
import sys


def run(*args):
    """Run one of this template's scripts, and FAIL LOUDLY if it fails.

    Do not use `!python script.py` here. IPython's shell escape does not raise on a non-zero
    exit: measured, a cell running `!python -c "sys.exit(7)"` completes, the next cell runs,
    and papermill exits 0. Every stage below is invoked from a cell, so a green run would say
    nothing about whether the pipeline finished.
    """
    subprocess.run([sys.executable, *args], check=True)


# Rung 1: no GPU, no weights, no cluster, nothing installed. If these fail, nothing below
# is worth running.
run("tests/test_packing.py")
run("tests/test_pipeline.py")
```

## `num_gpus` is admission control, not a memory limit

`num_gpus=0.02` does not cap an actor's VRAM. It tells Ray it may place 50 of them on one card,
because the fractions sum to 1.0, and CUDA then decides whether that was true. The two limits
come from different numbers:

```
by fraction :  floor(1 / num_gpus)                  actors per GPU
by VRAM     :  floor(vram_per_gpu / vram_per_actor)  actors per GPU
```

`packing.py` computes both. On the source engagement's shipped configuration, on the 48 GiB card
it was tuned on:

| stage | `num_gpus` | GiB/actor | by fraction | by VRAM | binds |
|---|---|---|---|---|---|
| detector | 0.20 | 9.60 · unmeasured | 5 | 5 | both |
| object embedder | 0.05 | 7.65 | 20 | 6 | VRAM |
| image embedder | 0.05 | 0.96 | 20 | 50 | fraction |
| metrics | 0.02 | 2.80 | 50 | 17 | VRAM |

Read only the fractions and you provision three times the object-embedder actors that fit. The
two embedders take the same fraction and 8x the memory apart. The detector's per-actor figure is
its fraction share, and the tool prints `UNMEASURED` beside it.


```python
# The source engagement's shipped configuration on the 48 GiB card it was tuned on.
run("packing.py")
```

Per-stage capacity does not answer whether the set fits together. `packing.py` carries two
tables, because the numbers above are the source engagement's on its own model set:

| | one actor of each stage | card | headroom |
|---|---|---|---|
| `--stages shipped` (their models) | 21.01 GiB | 48 GiB G2 | 56% |
| `--stages measured` (sam3 + dinov3 + siglip2) | 5.53 GiB | 22.03 GiB L4 | 75% |

The measured row is two GPU probe jobs on a g6 L4. Do not read the shipped table's 21.01 GiB as
this set's footprint: different models. An L4 is 24 GB, which is 22.35 GiB, of which torch
reports 22.03 usable, so pass `--vram 22.03`. At 22.03 the shipped set leaves 5%, which
`packing.py` flags as the band the batch-192 OOM lived in.

The shipped actor counts still over-commit an L4, which is the other half of what `packing.py`
prints.

The object embedder's measured 0.59 GiB was taken at one crop per frame. Its cost scales with
detections, not frames, so treat it as a floor for a nearly-empty batch and not a per-actor
budget for a busy one.


```python
# This template's own four models, on the card in configs/, against the 22.03 GiB an L4
# reports rather than a nominal 24.
#
# `--strict` exits 1 if the four models cannot be co-resident, if a stage is over-committed,
# or if the relative-cost ordering has inverted. Worth knowing before the GPU spend.
run("packing.py", "--stages", "measured", "--vram", "22.03", "--strict")
```

### Whether packing pays: at least 26.5% over running the stages serially

Measured on one L4, 96 frames at 640x480, one actor per stage:

| arm | rows/s (n=3) | spread |
|---|---|---|
| co-resident — four stages sharing the card | 1.2619 / 1.2631 / 1.2792 | 1.4% |
| serial — each stage alone with the whole GPU, materialized between | 0.9882 / 0.9903 / 0.9979 | 1.0% |

Verdict: SEPARABLE, co-resident > serial by >= 26.5%. Three rounds agreed.

The rule behind that verdict: an arm needs at least two runs, and two arms are separable only
when the worst run of the better arm beats the best run of the worse one. Here 1.2619 > 0.9979,
and the margin quoted is that gap, not a ratio of means. Overlapping ranges are not separable at
that sample size, however far apart their averages sit. `measure_packing.py` applies this rule
to the runs it collects.

Quote the scope with the number. This is end-to-end wall clock including warm model loads on a
96-frame fixture. It is a different quantity from the ~49.7 rows/s steady-state figure below.
Whether the margin holds at production scale, or with the shipped actor counts on a bigger card,
is not measured.

Re-derive it with `measure_packing.py`: one GPU, the gated weights, about 20 minutes. It writes
a JSONL of every run and prints the verdict. Its docstring carries the three guards and the
failure each one prevents. The short version: a cost paid once by whichever arm runs first will
make your better arm look worse.

## Each label costs a forward pass

`LABELS` defaults to two concepts, which is two detector passes per microbatch. SAM 3's
Promptable Concept Segmentation takes one noun phrase and returns every instance of it, so the
label dimension does not batch the way the frame dimension does.

Each label must be a concrete noun phrase. Measured on an L4 against this template's own
fixture, boxes per frame at the shipped threshold of 0.4:

| prompt | boxes per frame |
|---|---|
| `object` / `shape` / `region` | 0 0 0 0 |
| `rectangle` | 1 0 0 3 |
| `bright square` | 1 1 1 2 |

`object,shape,region` returns nothing at any threshold down to 0.05. The same weights, same GPU
and same threshold find 2 cats, 2 remote controls and 1 couch in a COCO photograph, so the words
are what changed. PCS grounds noun phrases; abstractions ground to nothing.

Re-check the labels if you change the imagery. A label that grounds to nothing still costs a
full forward pass, and reads downstream as an empty frame.

Do not write `text=[labels] * len(frames)`. `Sam3Processor` hands `text` straight to its
tokenizer, so `list[str]` is one prompt per image; `list[list[str]]` is HuggingFace's
pre-tokenized-words form, which needs `is_split_into_words=True` that the processor never
passes. On real weights the fast tokenizer rejects it:

```
TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
```

Budget the detector at `L` passes and keep `L` small.

## The fixture, and the whole DAG with no weights

`make_fixture.py` generates frames, so there is no download, no attribution and no licence to
track.

Frame count and frame geometry may shrink; neither carries the lesson. Do not trim the
resident-model count to fit a budget.


```python
# Read from the environment so a run can be shrunk without editing the notebook. These
# defaults are the demo; the CI test exports smaller ones. The frame COUNT and the frame
# GEOMETRY are the two things you may shrink -- the number of resident models is not,
# because that is the whole claim.
FIXTURE = os.environ.get("FIXTURE_DIR", "/mnt/cluster_storage/frames")
OUTPUT = os.environ.get("OUTPUT_DIR", "/mnt/cluster_storage/frames-enriched")
FRAMES = os.environ.get("FRAMES", "48")
FILES = os.environ.get("FILES", "4")
WIDTH = os.environ.get("WIDTH", "640")
HEIGHT = os.environ.get("HEIGHT", "480")

run("make_fixture.py", "--out", FIXTURE, "--frames", FRAMES, "--files", FILES,
    "--width", WIDTH, "--height", HEIGHT)
```

`--stub` runs the whole DAG with no weights and no GPU, which is how the shapes get tested
anywhere. It drops the GPU reservations too: `num_gpus=0.2` on a box without one does not fail,
it hangs.


```python
run("pipeline.py", "--input", FIXTURE, "--stub")
```

## Dependencies

`python_depset.lock` is the compiled closure of `requirements.txt` against a freeze of this
template's base image. Install the lock, not `requirements.txt`, or the driver and the actors
run different resolutions of the same pins.


```python
# `uv`, not this kernel's python, so this one is spelled out instead of going through run().
# Keep check=True: a half-finished install must not read as a working environment.
#
# Keep the requirements flag and the lock filename adjacent in the source text.
# scripts/hooks/check-dep-delivery.py proves a template installs its own lock by matching the
# flag followed by whitespace and the filename. A hand-written argument list puts a comma and
# a quote between them, the match fails, and the template reads as shipping a lock nothing
# installs.
INSTALL = (
    "uv pip install -r python_depset.lock --system --no-deps --no-cache-dir "
    "--index-strategy unsafe-best-match"
)
subprocess.run(INSTALL.split(), check=True)
```

That install reaches the driver only, and the models do not run there. Every stage is a Ray Data
actor on a GPU worker, and a `uv pip install --system` does not propagate. `pipeline.py` hands
the actors the same lock with `ray.init(runtime_env={"pip": .../python_depset.lock})`. Delete
that line and every stage fails on `import torch` while the driver looks fine.


```python
# TOUCH a class, do not just import one. transformers is a lazy module: `from transformers
# import Sam3Model` binds a name and loads nothing, and it passed on an image with no torch
# installed at all.
import torch
import torchvision
import transformers
from transformers import Sam3Model

print("transformers", transformers.__version__, "torch", torch.__version__,
      "torchvision", torchvision.__version__, "sam3 ->", Sam3Model.config_class.__name__)
```

## First, the half that needs no token

SigLIP2 is not gated and the metrics stage carries no weights, so those two stages run on a real
GPU with real weights and no token. Worth running before the gated download: it shows the GPU is
there, the runtime env reached the actors, the fractional reservations schedule, and two models
share one card.

Two models, not four. The four-model figure rests on `packing.py`'s arithmetic above plus your
own run below. `pipeline.py` prints which pair it ran.

This is what CI runs, for the licensing reason in the next section.


```python
UNGATED_OUT = os.environ.get("UNGATED_OUTPUT_DIR", OUTPUT + "-ungated")

# Real GPU, real SigLIP2 weights, no token. `--ungated-only` is a flag and never reads a
# credential. Do not make it infer from a missing token: a run that silently drops stages
# becomes an untested branch that is the only branch anyone runs.
run("pipeline.py", "--input", FIXTURE, "--output", UNGATED_OUT, "--ungated-only")
```


```python
import ray

ds = ray.data.read_parquet(UNGATED_OUT)
rows = ds.take_all()
assert rows, "the ungated run produced no rows"

# What two stages on real weights can be held to. The image embedder is SigLIP2's vision
# tower, so the width is a real width and not a stub's; sharpness is finite because the
# metrics kernel ran on the card.
import math

widths = {len(r["image_embedding"]) for r in rows}
assert widths and min(widths) >= 256, f"image embedding width {widths} looks like a stub"
assert all(math.isfinite(float(r["sharpness"])) for r in rows), "a sharpness score is not finite"
assert "image" not in rows[0], "the blob column must not survive to the output"

print(f"ungated pair: {len(rows)} rows, image embedding width {widths}, "
      f"all sharpness finite")
```

## Now the gated half, which only you can authorise

SAM 3 and DINOv3 are gated on Hugging Face and their terms are accepted per account by whoever
runs this. So there is no shared token for this template: CI does not hold one, this repo does
not fetch one, and the cells below are stripped from the CI run.

`templates/vla-fine-tuning` reads a shared organisation token from Secrets Manager for its gated
model. Do not copy that here. SAM 3's licence is accepted per account, and a shared credential
would stand in for a person's agreement.

Accept the terms on both model pages, create a token, export it as `HF_TOKEN`, and run the rest.
The cells from here down are tagged `skip-in-ci`.


```python
# Preflight the gate before loading anything. Without this the failure is a 403 traceback
# several GB into a model download, naming neither the cause nor the remedy.
#
# A FILE fetch is the only request that answers this. `/api/models/<repo>` returns 200
# anonymously for both of these, so the metadata endpoint cannot tell you whether you can
# pull.
import json, os, sys, urllib.error, urllib.request

GATED = ["facebook/sam3", "facebook/dinov3-vitl16-pretrain-lvd1689m"]
token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    sys.exit("HF_TOKEN is empty. Accept the terms on both gated model pages with your own\n"
             "Hugging Face account, create a token (a fine-grained READ token is enough),\n"
             "and export it as HF_TOKEN. Nobody can do this step for you: SAM 3 and DINOv3\n"
             "are licensed per ACCOUNT.")


def get(url):
    """Returns ('http', status, body) or ('unreachable', message, None).

    DENIED and UNREACHABLE are different claims and must not share a branch. Catching only
    HTTPError would let a proxy or DNS failure raise straight through this preflight as a
    traceback, which is the failure mode the preflight exists to replace.

    The body is read INSIDE the `with`. Returning the response object instead closed it on
    the way out and the success path died on an empty read -- found by exercising the
    granted branch, which the two failure branches had looked fine without.
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

# WHOAMI FIRST, AND ITS FAILURE IS A DIFFERENT FAILURE. 401 means the token itself did not
# authenticate, and a corrected token DOES fix that; 403 on a file means the token is fine and
# the account is not on the gated list, where no token change helps. An earlier version of
# this check printed the per-account remedy for both -- so a developer with a typo in their
# token was told to go and accept terms they had already accepted. Measured: with a bogus
# token every request 401s, and that branch is the one that used to give the wrong advice.
if val != 200:
    sys.exit(
        f"HF_TOKEN did not authenticate: /api/whoami-v2 returned HTTP {val}.\n"
        "This is a CREDENTIAL failure, not a gate one -- Hugging Face does not know who you\n"
        "are, so a corrected token DOES fix it. Check the token is a live read token, and\n"
        "that it belongs to the account that accepted the terms."
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

### Your own four-model run

`pipeline.py` refuses with the two-step remedy if `HF_TOKEN` is missing, before starting a
multi-GB download.


```python
assert os.environ.get("HF_TOKEN"), (
    "Set HF_TOKEN first: two of the three models are gated, and gating is per ACCOUNT. "
    "Accept the terms on both model pages listed at the top of this notebook, then create "
    "a fine-grained read token. `--stub` above needs none of it."
)
```


```python
# One actor per stage, which fits the L4 in configs/. The shipped counts (10 detectors) need
# the 48 GiB class; see packing.py. `setdefault`, so anything already exported wins.
for key, value in dict(
    DETECTOR_ACTORS="1", EMB_ACTORS="1", METRICS_ACTORS="1",
    DETECTOR_BATCH="2", EMB_BATCH="8", METRICS_BATCH="8", METRICS_SUBBATCH="2",
).items():
    os.environ.setdefault(key, value)

run("pipeline.py", "--input", FIXTURE, "--output", OUTPUT)
```

Every assertion below is about the answer, not the speed: a per-frame count matching its own
detections, a real embedding width, and at least one frame the detector fired on. A pipeline
returning zero detections everywhere completes and measures nothing.


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

### Optional: re-derive the packing number

One GPU, the gated weights, about 20 minutes. It interleaves the arms, warms the node that does
the work, and throws away one untimed pass of each arm. It prints the verdict against the rule
above and writes every run to a JSONL, failures included.


```python
# Left commented: 20 minutes of GPU, and a measurement rather than a pipeline step.
# Uncomment to re-derive the number.
#
# run("measure_packing.py", "--input", FIXTURE, "--frames", "96", "--runs", "3")
```

## The levers

`pipeline.py` is the control panel. Each lever carries its measured effect and what should
change it.

| lever | default | why |
|---|---|---|
| `DOWNLOAD_NUM_CPUS` | 1.0 | a divisor, not a budget: 0.1 on 96 cores opened ~950 read slots and one 64-row block took 430 s |
| `DETECTOR_BATCH` | 4 | the detector is the throughput floor at ~13 images/s per actor with all SMs busy, and raising the batch does not raise throughput |
| `TORCH_COMPILE` | `default` | `reduce-overhead` was measured and rejected: its CUDA graphs produce overwritten-output failures inside Ray actors |
| `METRICS_SUBBATCH` | 4 | decoupled from the Ray batch. Batch 192-256 OOMed at 30+ GiB/actor; Ray batch 32 with sub-batch 4 landed at 4.95 GiB |
| `EMB_ACTORS` | 4 | fixed pool, `min_size == max_size`: the autoscaler parks below the GPU count on whole-GPU actors |

## Rates for this shape, and their scope

| rate | scope |
|---|---|
| ~49.7 rows/s | end to end, fixed 131k x 2 confirmation run |
| ~88 rows/s | active compute only; a watcher bug inserted 3h15m of idle, and the output prefix held ~321k duplicate rows from earlier runs |
| 209.999 rows/s | extraction only, no detector |
| 39.65 rows/s = 9.9 images/s | 99.18K images in 2h47m, GPU 93-99% |

~49.7 rows/s is the end-to-end figure. The larger two are not this pipeline end to end.

State the unit as well as the scope. rows/s, images/s and detections/s differ here by the
detections per frame, and published figures for this shape differ by which one they meant.

On the source engagement the numbers from fixed, repeated, anchored A/B runs came out lower than
everything quoted from exploratory sweeps. Two conclusions reversed that way, one of them after
being written up as a +3.4% win.

## Models and licences

All three models are public. Two are gated on Hugging Face and need their terms accepted once
by the account behind your `HF_TOKEN`.

| stage | model | licence |
|---|---|---|
| detector | `facebook/sam3` | SAM License, gated |
| object embedder | `facebook/dinov3-vitl16-pretrain-lvd1689m` | DINOv3 License, gated |
| image embedder | `google/siglip2-base-patch16-224` | Apache-2.0 |
| metrics | no model: gradient energy in plain torch | n/a, no weights |

`templates/vla-fine-tuning` supplies its gated model's token to CI from an organisation secret.
Do not copy that here: SAM 3's terms are accepted per account, so a service account would be
standing in for a person's agreement. CI runs `pipeline.py --ungated-only` and the gated cells
are tagged `skip-in-ci`.

See `NOTICE`. DINOv3 asks for a "Built with DINOv3" acknowledgement, which is the last line
in it.

Substituting a model is not a free swap. A detector that is cheap next to the embedders inverts
the shape and the packing defaults stop transferring, so `packing.py` checks the relative-cost
ordering and fails if it flips. YOLO variants are absent: Ultralytics is AGPL-3.0 and
YOLOv9/YOLOR are GPL-3.0.

## Guards worth copying

- **Raise on unloaded weights.** A random-initialization warning is logged and ignored, and the
  model then emits garbage at full speed.
- **Per-frame scatter.** The object embedder embeds a whole batch of crops in one forward pass
  and its output column is per row. Writing the batch total into every row reads like a
  formatting difference and is a wrong answer.
- **Degenerate input.** A frame with no detections yields an empty `(0, dim)` array, not an
  exception.

`tests/` runs the arithmetic and the stage answers with no GPU and no weights. The CI test runs
those, the four-stage stub DAG, and a two-model co-residency run on one card with the ungated
pair; its header says so. Everything asserted is feasibility or correctness, never a throughput
ratio: one run per arm is not a measurement.

## Layout

```
packing.py           the arithmetic. no GPU, no weights, no cluster
measure_packing.py   whether packing pays. one GPU, the gated weights, ~20 min
pipeline.py          four stages, fractional GPUs, --stub for CPU, --ungated-only for CI
make_fixture.py      synthetic frames, so there is no dataset licence to track
requirements.txt     what this template adds to the base image, and why each pin is there
python_depset.lock   the compiled closure of those pins, installed on the driver and handed
                     to the actors by pipeline.py
tests/               test_packing.py, test_pipeline.py, both runnable anywhere
NOTICE               the two Meta licences and their obligations
```

The compute configs and the CI test live in the `anyscale/templates` repo, at
`configs/ray-data-multimodel-frame-enrichment/{aws,gce}.yaml` (one L4, 24 GB = 22.35 GiB) and
`tests/ray-data-multimodel-frame-enrichment/tests.sh`.
