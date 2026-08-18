#!/usr/bin/env bash
# CI test for ray-data-multimodel-frame-enrichment.
#
# Runs README.ipynb top to bottom with papermill, which is what a user does. The notebook is
# the template; this script is the orchestration around it that a user does not need --
# shrinking the run, resolving the CI secret, and preflighting the gated models.
#
# The thesis of this template is CO-RESIDENCY: four models sharing one GPU. So the notebook
# asserts feasibility and correctness and deliberately does NOT assert a throughput ratio.
# One run per arm is not a measurement -- the source engagement reversed two of its own
# conclusions when a fixed confirmation run replaced an exploratory sweep -- and a CI job
# that quoted a packing speedup off single runs would be reproducing the exact mistake this
# workload's history is a record of.
#
# That measurement now EXISTS -- co-residency beats serial by >=26.5% on one L4, three
# rounds, n=3 per arm (see the README). It still does not belong here. CI runs each arm once,
# and a single run cannot separate 26% from a session-warmup artifact: two of the three
# rounds were NOT SEPARABLE on their own full data for exactly that reason. Asserting the
# ratio here would pass for the wrong reason on a good day and fail for the wrong reason on
# a busy one. Assert answers in CI; measure ratios where you can replicate.
#
# NO `skip-in-ci` TAG, and that is deliberate rather than an oversight. Every cell runs here.
# The mechanism is real and was checked before being ruled out: 16 tests.sh in this repo run
# `jupyter nbconvert --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor
# .remove_cell_tags=...` (13 spell the tag as a JSON list, 3 as a bare string; both were
# verified on nbconvert 7.16.6 and 7.17.1 to remove exactly the tagged cell, with a
# cell tagged `s` left in place as the control against character-wise coercion). But
# testing-template.md says to add the strip step only when needed, and nothing here needs
# it: the one cell that cannot run in CI -- measure_packing.py, 20 minutes of GPU -- ships
# commented out.
set -euxo pipefail

# CI shrink. The notebook's defaults are the demo: 48 frames, and the shipped actor counts
# from packing.py. These are read by the notebook from the environment, so nothing in it has
# to be edited to run smaller.
#
# The frame COUNT and the frame GEOMETRY may shrink. The NUMBER OF RESIDENT MODELS may not:
# dropping to two stages to fit a budget would make the template demonstrate something it
# does not claim.
export FIXTURE_DIR="${FIXTURE_DIR:-/mnt/cluster_storage/frames-fixture}"
export OUTPUT_DIR="${OUTPUT_DIR:-/mnt/cluster_storage/frames-enriched}"
export FRAMES=24 FILES=4 WIDTH=640 HEIGHT=480

# The CI-scale configuration: one actor per stage, which is what fits an L4. The shipped
# counts (10 detectors) need the 48 GiB class -- see packing.py. The notebook applies these
# with setdefault, so what is exported here wins.
export DETECTOR_ACTORS=1 EMB_ACTORS=1 METRICS_ACTORS=1
export DETECTOR_BATCH=2 EMB_BATCH=8 METRICS_BATCH=8 METRICS_SUBBATCH=2

# The ONLY install this script does, and it deliberately adds no torch. The notebook installs
# the template's own python_depset.lock partway through, at the point a user would -- and the
# cells above that point run on the BARE IMAGE, which is load-bearing rather than incidental:
# the base image ships numpy, pyarrow and ray and no torch, torchvision or transformers at
# all (measured; `-cu129` is the CUDA runtime, not PyTorch). Installing the lock here instead
# would destroy that property and test something the user never runs.
#
# `uv pip`, never a bare `pip install`: a workspace tracks a bare install and appends it
# UNPINNED to every actor's pip list, and one unhashed entry puts pip in hash-checking mode
# against a hashed lock, which fails every runtime env for this template.
# scripts/hooks/check-dep-delivery.py bare-pip enforces it.
uv pip install -q --system papermill ipykernel

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
# This stays in tests.sh rather than moving into the notebook: testing-template.md puts
# local-only orchestration -- hard gates, secret fetching -- here so the notebook stays clean.
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
        "are, so a corrected token DOES fix it. Check the token is a live read token; in CI,\n"
        "check what Secrets Manager returned for anyscale_hf_token."
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
PY

# Mimic the user: the whole notebook, top to bottom. `--cwd .` runs it from the template dir
# so its relative paths (packing.py, python_depset.lock, tests/) resolve; `--log-output`
# streams to the CI log and the .out.ipynb is what you read afterwards.
#
# Check the log's `Executing Cell N` and `Ending Cell N` counts MATCH before believing a
# pass: rayapp has reported success for a run whose SSH session dropped mid-notebook.
papermill README.ipynb /tmp/ray-data-multimodel-frame-enrichment.out.ipynb \
    --log-output --kernel python3 --cwd .
