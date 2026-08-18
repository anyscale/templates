#!/usr/bin/env bash
# CI test for ray-data-multimodel-frame-enrichment.
#
# The thesis of this template is CO-RESIDENCY: four models sharing one GPU. So the tests
# assert feasibility and correctness, and deliberately do NOT assert a throughput ratio.
# One run per arm is not a measurement -- the source engagement reversed two of its own
# conclusions when a fixed confirmation run replaced an exploratory sweep -- and a CI job
# that quoted a packing speedup off single runs would be reproducing the exact mistake this
# workload's history is a record of. The packing measurement lives in a separate GPU run
# with replicates -- measure_packing.py -- which scores itself against a stated rule.
#
# That measurement now EXISTS -- co-residency beats serial by >=26.5% on one L4, three rounds,
# n=3 per arm (see the README). It still does not belong here. CI runs each
# arm once, and a single run cannot separate 26% from a session-warmup artifact: two of the
# three rounds were NOT SEPARABLE on their own full data for exactly that reason. Asserting the
# ratio here would pass for the wrong reason on a good day and fail for the wrong reason on a
# busy one. Assert answers in CI; measure ratios where you can replicate.
set -euo pipefail

FIXTURE="${FIXTURE_DIR:-/mnt/cluster_storage/frames-fixture}"

# ---------------------------------------------------------------------------------
# 1. Rung 1: the arithmetic. No GPU, no weights, no cluster.
# ---------------------------------------------------------------------------------
# Steps 1 and 2 run on the BARE image, before any install. That is load-bearing rather
# than incidental: the base image ships numpy, pyarrow and ray and no torch at all
# (measured), so everything up to step 3 needs nothing added. If you move the install
# earlier, nothing breaks; if you move a torch-using step earlier, everything does.
python tests/test_packing.py
python tests/test_pipeline.py

# The set has to be co-resident on THIS fleet's card, whatever it is. The L4 in configs/ is
# a 24 **GB** card, which is 22.35 GiB, and torch reports 22.03 GiB usable after the driver
# takes its share -- MEASURED on a g6 L4, 2026-08-17. Asserting against 24 GiB was a unit
# error worth about 2 GiB of imaginary headroom, so this asserts against the measured
# figure. Over-commitment of the SHIPPED actor counts on a small card is expected and is not
# what this asserts -- co-residency of one actor per stage is.
python - <<'PY'
import json
import subprocess
# `--stages measured` is THIS template's four models, measured on an L4. The default table
# is the source engagement's configuration on a DIFFERENT model set, and asserting CI
# feasibility against it was measuring someone else's models.
out = json.loads(subprocess.run(
    ["python", "packing.py", "--stages", "measured", "--vram", "22.03", "--json"],
    capture_output=True, text=True, check=True).stdout)
assert out["coresidency"] == [], f"four models do not co-reside on an L4: {out['coresidency']}"
print(f"co-resident footprint {out['coresident_gib_one_each']:.2f} GiB of 22.03 GiB -- ok")
PY

# ---------------------------------------------------------------------------------
# 2. The DAG, with no weights. Catches scheduling; the stage tests catch answers.
# ---------------------------------------------------------------------------------
# Geometry and frame count may shrink for CI. The NUMBER OF RESIDENT MODELS may not:
# dropping to two stages to fit a budget would make the template demonstrate something it
# does not claim.
python make_fixture.py --out "$FIXTURE" --frames 24 --files 4 --width 640 --height 480

python pipeline.py --input "$FIXTURE" --stub | tee /tmp/stub.txt
grep -q "rows/s" /tmp/stub.txt

# ---------------------------------------------------------------------------------
# 3. The real thing, four models on one GPU.
# ---------------------------------------------------------------------------------
# The base image ships NEITHER torch nor transformers. `-cu129` is the CUDA runtime, not
# PyTorch -- an earlier version of this comment said torch was provided, and a probe job on
# the image is what corrected it. requirements.txt carries the measured baseline and why
# each pin is there. SAM 3 needs transformers 5.x, the one hard version constraint.
#
# It also carries what the image DOES ship, which changed under this template when the repo
# moved from Ray 2.56/cu128 to 2.57/cu129: numpy went 1.26.4 -> 2.2.6 and pyarrow 19 -> 23,
# which turned two "no-op" pins into a downgrade of Ray's own stack. They are gone now.
# Re-measure on an image bump; do not assume the pins carry.
#
# INSTALL THE LOCK, NOT requirements.txt. `python_depset.lock` is the compiled closure of
# those pins against a freeze of this exact image, and it is what pipeline.py hands the
# actors through `runtime_env`. Resolving requirements.txt here instead would test a
# different resolution from the one the workers get. `uv pip`, never a bare `pip install`:
# a workspace tracks a bare install and appends it UNPINNED to every actor's pip list,
# and one unhashed entry puts pip in hash-checking mode against a hashed lock and fails
# every runtime env for this template. scripts/hooks/check-dep-delivery.py enforces both.
uv pip install -q -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match

# TOUCH a class, do not just import one. transformers is a lazy module: `from transformers
# import Sam3Model` binds a name and loads nothing, and it passed on an image with no torch
# installed at all. That vacuous check is how the wrong comment above survived a round.
python -c 'import transformers, torch, torchvision; from transformers import Sam3Model; print("transformers", transformers.__version__, "torch", torch.__version__, "torchvision", torchvision.__version__, "sam3 ->", Sam3Model.config_class.__name__)'

# Both Meta repos are gated. AN HF_TOKEN ALREADY IN THE ENVIRONMENT WINS -- someone running
# this by hand brings their own, and their account is the one that accepted the terms. Only CI
# falls through to Secrets Manager, the same pattern as tests/vla-fine-tuning/tests.sh.
#
# The order matters and is not a convenience: reading the org secret unconditionally overwrote
# a working personal token with one whose account is on neither gated list, so a developer with
# access got the CI account's 403.
set +x  # do not echo the resolved secret under xtrace
if [ -n "${HF_TOKEN:-}" ]; then
    echo "using HF_TOKEN from the environment"
else
    export HF_TOKEN=$(aws --region=us-west-2 secretsmanager get-secret-value \
        --secret-id anyscale_hf_token --query SecretString --output text)
    echo "using HF_TOKEN from Secrets Manager (anyscale_hf_token)"
fi
set -x

# PREFLIGHT THE GATE, because the failure without it is a 403 traceback several GB into a
# model download and it names neither the account nor the remedy. Measured 2026-08-17: the
# secret resolves fine and `svc-huggingface` is on neither gated list, so this is the check
# that was missing rather than a hypothetical.
#
# A FILE fetch is the only request that answers this. `/api/models/<repo>` returns 200
# anonymously for both of these, so the metadata endpoint cannot tell you whether you can
# pull.
python - <<'PY'
import json, os, sys, urllib.error, urllib.request

GATED = ["facebook/sam3", "facebook/dinov3-vitl16-pretrain-lvd1689m"]
token = os.environ.get("HF_TOKEN", "").strip()
if not token:
    sys.exit("HF_TOKEN is empty. Export your own (a fine-grained read token is enough) "
             "after accepting the terms on both gated model pages, or -- in CI -- check "
             "that Secrets Manager returned something for anyscale_hf_token.")


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
try:
    who = json.loads(body).get("name", "unknown") if body else f"unknown (HTTP {val})"
except ValueError:
    who = f"unknown (unparseable whoami response, HTTP {val})"

denied, unreachable = [], []
for repo in GATED:
    kind, val, _ = get(f"https://huggingface.co/{repo}/resolve/main/config.json")
    if kind == "unreachable":
        unreachable.append(f"{repo} -> {val}")
    elif val != 200:
        denied.append(f"{repo} -> HTTP {val}")

if unreachable:
    sys.exit(f"could not check gated access: {'; '.join(unreachable)}")
if denied:
    sys.exit(
        f"Hugging Face account '{who}' cannot read: {', '.join(denied)}.\n"
        "Accept the terms once on each model card with that account. The gate is\n"
        "per-account, not per-token, so no token change fixes it; a fine-grained read\n"
        "token is sufficient once the terms are accepted."
    )
print(f"gated access ok as '{who}'")
PY

# The CI-scale configuration: one actor per stage, which is what fits an L4. The shipped
# counts (10 detectors) need the 48 GiB class -- see packing.py.
export DETECTOR_ACTORS=1 EMB_ACTORS=1 METRICS_ACTORS=1
export DETECTOR_BATCH=2 EMB_BATCH=8 METRICS_BATCH=8 METRICS_SUBBATCH=2

python pipeline.py --input "$FIXTURE" --output /mnt/cluster_storage/frames-enriched \
    | tee /tmp/real.txt

python - <<'PY'
import ray

ds = ray.data.read_parquet("/mnt/cluster_storage/frames-enriched")
rows = ds.take_all()
assert rows, "the real run produced no rows"

# Correctness, not speed. Every assertion here is about the ANSWER: a per-frame count that
# matches its own detections, a real embedding width, and at least one frame the detector
# actually fired on. A pipeline that returns zero detections everywhere completes happily
# and measures nothing.
mismatched = [r["frame_id"] for r in rows
              if int(r["n_detections"]) != int(r["object_embedding_count"])]
assert not mismatched, f"per-frame embedding count disagrees with detections: {mismatched[:5]}"

dims = {int(r["object_embedding_dim"]) for r in rows}
assert dims and min(dims) >= 256, f"embedding width {dims} looks like a stub, not DINOv3"

detected = sum(int(r["n_detections"]) for r in rows)
assert detected > 0, "the detector found nothing on any frame -- the fixture or the prompts are wrong"

print(f"{len(rows)} rows, {detected} detections, embedding dim {dims}")
PY
