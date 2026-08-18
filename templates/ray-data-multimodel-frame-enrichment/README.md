# Multi-model frame enrichment on one GPU

Four stages on the same GPU, three of them models: promptable detection, an object
embedder, a whole-frame embedder, and no-reference quality metrics. The interesting part is
not any one model, it is fitting all four onto one card so none of them starves the others.

## Before you run it: your own token, and two forms

Two of the three models are **gated**, and gating on Hugging Face is per **account**, not per
token — so no token setting substitutes for accepting the terms yourself:

1. accept the terms at <https://huggingface.co/facebook/sam3>
2. accept the terms at <https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m>
3. create a token at <https://huggingface.co/settings/tokens> — a **fine-grained read** token
   is enough — and export it as `HF_TOKEN`

Approval on both is usually immediate, but it is an approval and not a download, so it can
sit. `--stub` needs none of this: it exercises all four stages and every output shape with no
weights, no token and no GPU, which is also what CI runs before it touches the real thing.

The failure mode if you skip step 1 or 2 is worth recognising, because the two look nothing
alike. **401** means Hugging Face does not know who you are — a missing or malformed token.
**403 "you are not in the authorized list"** means your token is fine and your *account* has
not accepted the terms; a new token will not fix it. The CI test for this template
preflights both and names which one it got.

## The order of this notebook is not cosmetic

The dependency install sits **below** the arithmetic and the stub run, and moving it up is
free while moving a torch-using cell up is not. Measured on the base image
`anyscale/ray:2.57.0-py312-cu129`: it ships numpy, pyarrow and ray, and **no torch,
torchvision or transformers at all** — `-cu129` is the CUDA runtime, not PyTorch. So
everything before the install cell runs on the bare image, which is what makes those cells
runnable anywhere, including on a laptop with no GPU.

`tests/` is the same idea in test form: it runs the arithmetic and the stage answers with no
GPU and no weights.


```python
# Rung 1: no GPU, no weights, no cluster, nothing installed. If these fail, nothing below
# is worth running.
!python tests/test_packing.py
!python tests/test_pipeline.py
```

## `num_gpus` is admission control, not a memory limit

This is the trap the template exists for. `num_gpus=0.02` does not cap an actor's VRAM. It
tells Ray it may place fifty of them on one card, because the fractions sum to 1.0, and
CUDA then decides whether that was true. The two limits come from different numbers:

```
by fraction :  floor(1 / num_gpus)                  actors per GPU
by VRAM     :  floor(vram_per_gpu / vram_per_actor)  actors per GPU
```

`packing.py` computes both. On the source engagement's own shipped configuration, on the
48 GiB card it was tuned on:

| stage | `num_gpus` | GiB/actor | by fraction | by VRAM | binds |
|---|---|---|---|---|---|
| detector | 0.20 | 9.60 · unmeasured | 5 | 5 | both |
| object embedder | 0.05 | 7.65 | 20 | 6 | **VRAM** |
| image embedder | 0.05 | 0.96 | 20 | 50 | fraction |
| metrics | 0.02 | 2.80 | 50 | 17 | **VRAM** |

Read only the fractions and you would provision three times the object-embedder actors
that fit. Note the two embedders: **same fraction, eight times the memory.** The
detector's per-actor figure is its fraction share rather than a measurement, and the tool
prints it as `UNMEASURED` so it cannot be quoted as one.


```python
# The source engagement's shipped configuration on the 48 GiB card it was tuned on.
!python packing.py
```

Co-residency is the constraint the per-stage view misses, and `packing.py` carries **two
tables** because the numbers above are the source engagement's on the source engagement's
model set:

| | one actor of each stage | card | headroom |
|---|---|---|---|
| `--stages shipped` (their models) | 21.01 GiB | 48 GiB G2 | 56% |
| `--stages measured` (sam3 + dinov3 + siglip2) | **5.53 GiB** | 22.03 GiB L4 | **75%** |

The measured row is two GPU probe jobs on a g6 L4, not arithmetic. An earlier version of
this section quoted the 21.01 GiB figure as *this* set's footprint and concluded it fit a
24 GiB L4 with 12.5% headroom. Two errors: the models are not the same models, and **an L4
is 24 GB, which is 22.35 GiB, of which torch reports 22.03 usable** — so `--vram 24` was a
unit error worth about 2 GiB of headroom that does not exist. On the real capacity the
source table leaves 5%, which `packing.py`'s own rule flags as the band the batch-192 OOM
lived in.

The shipped *actor counts* still over-commit the L4, which is the other half of what
`packing.py` prints.

One caveat travels with the measured table and is not a formality: the object embedder's
0.59 GiB was measured at **one crop per frame**. Its cost scales with detections, not
frames, so that figure is a floor for a nearly-empty batch and not a per-actor budget for a
busy one.


```python
# This template's own four models, on the card in configs/. This is the assertion the CI
# test makes -- co-residency of one actor per stage, against the measured 22.03 GiB an L4
# actually reports rather than a nominal 24.
!python packing.py --stages measured --vram 22.03
```

### And it is worth doing: at least 26.5% faster than running the stages serially

Feasibility is arithmetic; whether packing *pays* is not. Measured on one L4, 96 frames at
640×480, one actor per stage:

| arm | rows/s (n=3) | spread |
|---|---|---|
| **co-resident** — four stages sharing the card | **1.2619 / 1.2631 / 1.2792** | 1.4% |
| serial — each stage alone with the whole GPU, materialized between | 0.9882 / 0.9903 / 0.9979 | 1.0% |

Verdict: **SEPARABLE, co-resident > serial by ≥ 26.5%.** Three rounds agreed.

The verdict is one blunt rule and it is worth stating rather than citing: an arm needs at
least two runs, and two arms are separable only when the *worst* run of the better arm beats
the *best* run of the worse one. Here 1.2619 > 0.9979, and the margin quoted is that gap, not
a ratio of means. Ranges that overlap are not separable at that sample size, however far apart
their averages sit. `measure_packing.py` applies exactly this rule to the runs it collects.

**Quote the scope with it.** This is end-to-end wall clock *including warm model loads* on a
96-frame fixture — a CI-scale answer to a CI-scale question. It is a different quantity from
the ~49.7 rows/s steady-state figure below, and the two must not be compared. Whether the
margin holds at production scale, or with the shipped actor counts on a bigger card, is not
measured.

Re-derive it with **`measure_packing.py`** (one GPU, the gated weights, ~20 min). It writes a
JSONL of every run and prints the verdict against the rule above, so nothing about the number
lives outside the template. Its docstring carries the three guards the failed rounds bought
and names the failure each one prevents — the short version is that a cost paid once by
whichever arm runs first will make your better arm look worse, and no amount of interleaving
fixes it.

## Each label is a forward pass, and that is the model's rule not a knob

`LABELS` defaults to two concepts, which is **two detector passes per microbatch**. SAM 3's
Promptable Concept Segmentation takes one noun phrase and returns every instance of it, so
the label dimension cannot be batched away the way the frame dimension can.

**The labels have to be concrete, and that is measured rather than stylistic.** This
template shipped `object,shape,region` as its default, and on an L4 against its own fixture
that returns **nothing at any threshold down to 0.05**:

| prompt | boxes per frame at the shipped threshold 0.4 |
|---|---|
| `object` / `shape` / `region` | 0 0 0 0 |
| `rectangle` | 1 0 0 3 |
| `bright square` | 1 1 1 2 |

The control that makes this a statement about the words rather than about the model: the
same weights, same GPU, same threshold, on a COCO photograph find 2 cats, 2 remote controls
and 1 couch. PCS grounds noun phrases; abstractions ground to nothing. If you change the
imagery, re-check the labels against it — a label that grounds to nothing still costs a
full forward pass and reads downstream as "there was nothing there".

It matters because the obvious code is wrong. `Sam3Processor` hands `text` straight to its
tokenizer, so `list[str]` is one prompt *per image*, and `text=[labels] * len(frames)` is
`list[list[str]]` — HuggingFace's pre-tokenized-*words* form, which needs
`is_split_into_words=True` that the processor never passes. On real weights the fast
tokenizer rejects it:

```
TypeError: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
```

This template did exactly that until `facebook/sam3` became readable. An earlier version of
this section said the wrong call returned plausible boxes for the phrase
`"object shape region"`; it does not, and that was written from reading the library rather
than running it. Budget the detector at `L` passes and keep `L` small.

## The fixture, and the whole DAG with no weights

`make_fixture.py` generates frames, so there is no download, no attribution and no licence
to track. The frame **count** and the frame **geometry** may both shrink — neither carries
the lesson — but the number of resident models may not. Dropping to two stages to fit a
budget would make the template demonstrate something it does not claim.


```python
import os

# Override to point somewhere else. The CI test uses 24 frames at 640x480; the geometry and
# the count are the two things you are allowed to shrink.
FIXTURE = os.environ.get("FIXTURE_DIR", "/mnt/cluster_storage/frames")
OUTPUT = os.environ.get("OUTPUT_DIR", "/mnt/cluster_storage/frames-enriched")

!python make_fixture.py --out {FIXTURE} --frames 24 --files 4 --width 640 --height 480
```

`--stub` runs the whole DAG with no weights and no GPU, which is how the shapes get tested
anywhere. It drops the GPU reservations too — asking for `num_gpus=0.2` on a box without
one does not fail, it hangs.


```python
!python pipeline.py --input {FIXTURE} --stub
```

## Dependencies: the driver install does not reach the models

`python_depset.lock` is the compiled closure of `requirements.txt` against a freeze of this
template's base image. Install the lock, not `requirements.txt`, or the driver and the actors
run different resolutions of the same pins.


```python
!uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
```

**That install reaches the driver only, and the models do not run there.** Every stage of
this pipeline is a Ray Data actor on a GPU worker, and neither the compute config's head
node nor a `uv pip install --system` puts torch there. `pipeline.py` hands the same lock to
the actors with `ray.init(runtime_env={"pip": .../python_depset.lock})` — that is the line
that makes the four models importable where they actually load. Delete it and every stage
fails on `import torch` while the driver looks fine.


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

## The real run

Your own `HF_TOKEN`, from the account that accepted both sets of terms. `pipeline.py`
refuses with the two-step remedy rather than dying several GB into a download if it is
missing.

The CI test does more than this cell: it resolves the token from AWS Secrets Manager when
the environment has none, and preflights both gated repos with a **file** fetch before
loading anything — `/api/models/<repo>` answers 200 anonymously for both of these, so the
metadata endpoint cannot tell you whether you can pull. That orchestration lives in
`tests/tests.sh` rather than here on purpose, so the notebook stays clean of secret
fetching.


```python
assert os.environ.get("HF_TOKEN"), (
    "Set HF_TOKEN first: two of the three models are gated, and gating is per ACCOUNT. "
    "Accept the terms on both model pages listed at the top of this notebook, then create "
    "a fine-grained read token. `--stub` above needs none of it."
)
```


```python
# The CI-scale configuration: one actor per stage, which is what fits an L4. The shipped
# counts (10 detectors) need the 48 GiB class -- see packing.py.
os.environ.update(
    DETECTOR_ACTORS="1", EMB_ACTORS="1", METRICS_ACTORS="1",
    DETECTOR_BATCH="2", EMB_BATCH="8", METRICS_BATCH="8", METRICS_SUBBATCH="2",
)

!python pipeline.py --input {FIXTURE} --output {OUTPUT}
```

Every assertion below is about the **answer**, not the speed: a per-frame count that matches
its own detections, a real embedding width, and at least one frame the detector actually
fired on. A pipeline that returns zero detections everywhere completes happily and measures
nothing.


```python
import ray

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

### Optional: re-derive the packing claim yourself

One GPU, the gated weights, about 20 minutes. It interleaves the arms, warms the node that
does the work, and throws away one pass of each arm before timing anything — three guards,
each of which exists because a round without it measured the wrong thing. It prints the
verdict against the rule above and writes every run to a JSONL, including the failures.


```python
# !python measure_packing.py --input {FIXTURE} --frames 96 --runs 3
```

## Running it: the levers

`pipeline.py` is the control panel. Every lever is one the source engagement measured, each
carries its effect inline, and each says what should flip it.

| lever | default | why |
|---|---|---|
| `DOWNLOAD_NUM_CPUS` | 1.0 | a divisor, not a budget: 0.1 on 96 cores opened ~950 read slots and one 64-row block took 430 s |
| `DETECTOR_BATCH` | 4 | the detector is the throughput floor at ~13 images/s per actor with all SMs busy, and raising the batch does not raise throughput |
| `TORCH_COMPILE` | `default` | `reduce-overhead` was expected to win and was **rejected**: its CUDA graphs produce overwritten-output failures inside Ray actors |
| `METRICS_SUBBATCH` | 4 | decoupled from the Ray batch. Batch 192–256 OOMed at 30+ GiB/actor; Ray batch 32 with sub-batch 4 landed at 4.95 GiB |
| `EMB_ACTORS` | 4 | fixed pool, `min_size == max_size`: the autoscaler parks below the GPU count on whole-GPU actors |

## What the numbers mean, and which ones not to quote

The honest end-to-end figure for this shape on the source fleet is **~49.7 rows/s**, from a
fixed 131k×2 confirmation run. Two larger numbers circulate and neither is this pipeline
end to end:

| rate | scope |
|---|---|
| **~49.7 rows/s** | **full end to end, fixed confirmation run** |
| ~88 rows/s | active compute only; a watcher bug inserted 3h15m of idle, and the output prefix held ~321k duplicate rows from earlier runs |
| 209.999 rows/s | extraction only, no detector |
| 39.65 rows/s = 9.9 images/s | 99.18K images in 2h47m, GPU 93–99% |

The general lesson underneath them is the method: a broad sweep proposes and a fixed,
repeated, anchored A/B disposes. On the source engagement the honest end-to-end numbers came
from the second kind and were **lower** than everything quoted from the first. Two
conclusions reversed that way, including one that had already been written up as a +3.4%
win.

State the unit as well as the scope. rows/s, images/s and detections/s differ here by the
detections per frame, and the published figures for this shape differ by which one they
meant.

## Models and licences

All four are public. Two are gated on Hugging Face and need their terms accepted once by
the account behind your `HF_TOKEN`; `templates/vla-fine-tuning` is the repo's existing
pattern for that, including how CI supplies the token.

| stage | model | licence |
|---|---|---|
| detector | `facebook/sam3` | SAM License, gated |
| object embedder | `facebook/dinov3-vitl16-pretrain-lvd1689m` | DINOv3 License, gated |
| image embedder | `google/siglip2-base-patch16-224` | Apache-2.0 |
| metrics | no model: gradient energy in plain torch | n/a, no weights |

See `NOTICE`. DINOv3 asks for a "Built with DINOv3" acknowledgement, which is why that line
is in it. **Substituting a model is not a free swap**: a detector that is cheap relative to
the embedders inverts the shape and teaches the wrong defaults, so `packing.py` checks the
relative-cost ordering and fails if it flips. YOLO variants are deliberately absent —
Ultralytics is AGPL-3.0 and YOLOv9/YOLOR are GPL-3.0.

## Guards worth copying

Three of these exist because the source engagement shipped without them:

- **`require_pretrained`.** One embedding model logged a random-initialization warning and
  carried on, emitting confident garbage at full speed. A warning nobody reads is not a
  guard, so this raises.
- **Per-frame scatter.** The object embedder embeds a whole batch of crops in one forward
  pass and its output is per row. Getting that wrong writes the batch total into every
  row's count, which reads like a formatting difference and is a wrong answer.
- **Degenerate input.** A frame with no detections yields an empty `(0, dim)` array, not an
  exception. The source engagement found that one in production, from the customer.

`tests/` runs the arithmetic and the stage answers with no GPU and no weights; the repo's CI
test then runs the DAG and the real four-model run on top of them. Both assert feasibility and
correctness and never a throughput ratio: one run per arm is not a measurement.

## Layout

```
packing.py           the arithmetic. no GPU, no weights, no cluster
measure_packing.py   whether packing PAYS. one GPU, the gated weights, ~20 min
pipeline.py          four stages, fractional GPUs, --stub for CPU
make_fixture.py      synthetic frames, so there is no dataset licence to track
requirements.txt     what this template adds to the base image, and why each pin exists
python_depset.lock   the compiled closure of those pins. installed on the driver AND
                     handed to the actors by pipeline.py -- see the note above
tests/               test_packing.py, test_pipeline.py -- both runnable anywhere
NOTICE               the two Meta licences and their obligations
```

The compute configs and the CI test live in the `anyscale/templates` repo rather than in the
template, at `configs/ray-data-multimodel-frame-enrichment/{aws,gce}.yaml` (one L4 — 24 GB =
22.35 GiB) and `tests/ray-data-multimodel-frame-enrichment/tests.sh`.
