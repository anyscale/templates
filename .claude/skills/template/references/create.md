# Create a new template

`<name>` below is the template's identifier (used in `BUILD.yaml`, `configs/`, `tests/`). Must match `^[a-z0-9_-]+$`.

## Authoring (defer)

Authoring lives in **anyscale/anyscale-template-agent** (https://github.com/anyscale/anyscale-template-agent) — a 4-phase pipeline: Author → Debug → Diagram → Finalize. Two starting points:

- **Bootstrap from scratch** (prompt-only) — describe what the template should demonstrate; the agent generates it.
- **Wrap existing work** — Jupyter notebook, Python files, markdown, URL, or GitHub repo.

Wait for the deliverable (typically a `.ipynb` + supporting files). Move it into `templates/<dir>/` and continue below.

## Step 1: BUILD.yaml entry

Append a list item per `build-yaml-schema.yaml`. Pick the image case per SKILL.md "Image URI cases":

- **Stock Anyscale** — `cluster_env.image_uri: anyscale/ray:<tag>` (or `anyscale/ray-llm:<tag>`).
- **Anyscale custom on GCP** — publish via `.claude/skills/template/scripts/publish-custom-image.sh <dockerfile-dir> <name> <ray-version>`, then set `cluster_env.byod.docker_image` to the printed URI and `byod.ray_version` to the matching Ray version.
- **Third-party** — set `cluster_env.byod.docker_image` to the upstream tag; infer `byod.ray_version` from the image.

Cross-field rules (validator-enforced) live at the bottom of `build-yaml-schema.yaml`. The common ones: `configs/<name>/` and `tests/<name>/` directory basenames must equal `<name>`.

## Step 2: Compute configs

Legacy schema, per `compute-config-schema.yaml`. NOT the new ComputeConfig API.

**Preferred — translate from a tested workspace.** Ask the user for the Anyscale console URL of the workspace they validated the template on. Pull the workspace's compute config (head node type, worker node types, autoscaler bounds, spot, flags) and translate into `configs/<name>/aws.yaml` + `configs/<name>/gce.yaml`.

**Fallback — guided Q&A.** Ask:
- GPU or CPU? Head + worker instance type?
- `min_workers` / `max_workers`?
- Spot or on-demand (`use_spot`)?
- Cross-zone autoscaling (`flags.allow-cross-zone-autoscaling: true`)?

Cross-check shape against `configs/distributing-pytorch/{aws,gce}.yaml` (GPU example) or `configs/basic-single-node/{aws,gce}.yaml` (CPU single-node — shared across intro and serve templates). AWS uses `m5.*` / `g5.*` / `g6.*` / `p4d.*`; GCP uses `n2-standard-*` / `g2-standard-*-nvidia-l4-*` / `a2-highgpu-*-nvidia-a100-*`. Pair AWS and GCP at the same GPU class.

## Step 3: Test script

Write `tests/<name>/tests.sh`. Three shapes — pick by template type:

- **Notebook via nb2py** (most common) — convert `README.ipynb` to Python and execute. Copy `tests/distributing-pytorch/tests.sh` + `tests/distributing-pytorch/nb2py.py` as the starting point.
- **Notebook via papermill** — for notebooks that don't translate cleanly via nb2py. See `tests/audio-dataset-curation-llm-judge/tests.sh`.
- **Custom script** — for templates shipping a `.py` entrypoint or service. See `tests/asynchronous_inference/tests.sh`.

## Step 4: Validate

```
python3 ci/validate_build_yaml.py --no-network
```

Fix anything it reports — schema errors, missing paths, name mismatches.

## Handoff

Now run the publish flow per `publish.md`.
