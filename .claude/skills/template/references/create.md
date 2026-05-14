# Create a new template

`<name>` below is the template's identifier — used in `BUILD.yaml`, `configs/<name>/`, `tests/<name>/`.

## Source the content

Move the template's content (notebook / Python files / partial draft) into `templates/<dir>/`. Most users arrive with something partial-to-done already — just integrate what they have.

If nothing exists yet, or the user wants existing work agent-polished, defer to **anyscale/anyscale-template-agent** (https://github.com/anyscale/anyscale-template-agent) first:

- **Bootstrap from scratch** (prompt-only) — when there's no starting content.
- **Wrap or improve existing work** — for adding diagrams, debugging, or finalizing.

Wait for its deliverable, drop it in `templates/<dir>/`, then continue.

## Step 1: BUILD.yaml entry

Append a list item per `build-yaml-schema.yaml`. Pick the image case per SKILL.md "Image URI cases" — set `cluster_env.image_uri` (stock) or `cluster_env.byod.{docker_image,ray_version}` (custom or third-party). For custom GCP, publish the image first (script in SKILL.md) and use the printed URI.

Cross-field rules (validator-enforced) live at the bottom of `build-yaml-schema.yaml`.

## Step 2: Compute configs

Legacy schema, per `compute-config-schema.yaml`. NOT the new ComputeConfig API.

**Preferred — translate from a tested workspace.** Ask the user for the Anyscale console URL of the workspace they validated the template on. Extract the workspace ID (`expwrk_*` from `/workspaces/<id>`) and fetch its config:

```
anyscale workspace_v2 get --id expwrk_<id> --json | jq '.config.compute_config'
```

The CLI returns the new ComputeConfig API shape; translate fields into the legacy schema per `compute-config-schema.yaml`. Pick `configs/<name>/aws.yaml` or `configs/<name>/gce.yaml` by instance family. For dual-cloud support, mirror the same GPU class across both files. `idle_termination_minutes`, `region`, and cloud metadata don't go in the template config — they're operational.

**Fallback — guided Q&A.** When no tested workspace exists, walk the user through the schema fields (`compute-config-schema.yaml`) — head/worker instance types, autoscaler bounds, spot, cross-zone autoscaling.

## Step 3: Test script

Write `tests/<name>/tests.sh`. Two shapes — pick by template type:

- **Notebook via papermill** (most common) — for `.ipynb` templates. See `tests/audio-dataset-curation-llm-judge/tests.sh`.
- **Custom script** — for templates shipping a `.py` entrypoint or service. See `tests/asynchronous_inference/tests.sh`.

## Step 4: Validate

```
python3 ci/validate_build_yaml.py --no-network
```

Fix anything it reports.

## Handoff

Now run the publish flow per `publish.md`.
