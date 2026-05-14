# Create a new template

Read `format.md` first — it owns repo conventions, schemas, and the legacy-API warning. This file is the generation workflow.

`<name>` below is the template's identifier — used in `BUILD.yaml`, `templates/<name>/`, `configs/<name>/`, `tests/<name>/`.

## Source the content

Move the template's content (notebook / Python files / partial draft) into `templates/<name>/`. Most users arrive with something partial-to-done already — just integrate what they have.

If nothing exists yet, or the user wants existing work agent-polished, defer to **anyscale/anyscale-template-agent** (https://github.com/anyscale/anyscale-template-agent) first:

- **Bootstrap from scratch** (prompt-only) — when there's no starting content.
- **Wrap or improve existing work** — for adding diagrams, debugging, or finalizing.

Wait for its deliverable, drop it in `templates/<name>/`, then continue.

## Step 1: BUILD.yaml entry

Append a list item per `build-yaml-schema.yaml`. Pick the image case per SKILL.md "Image URI cases" — set `cluster_env.image_uri` (stock) or `cluster_env.byod.{docker_image,ray_version}` (custom or third-party). For custom GCP, publish the image first (script in SKILL.md) and use the printed URI.

## Step 2: Compute configs

**Preferred — translate from a tested workspace.** Ask the user for the Anyscale console URL of the workspace they validated the template on. Extract the workspace ID (`expwrk_*` from `/workspaces/<id>`) and fetch its config:

```
anyscale workspace_v2 get --id expwrk_<id> --json | jq '.config.compute_config'
```

The CLI returns the new ComputeConfig API shape; translate fields into the legacy schema per `compute-config-schema.yaml`. Pick `configs/<name>/aws.yaml` or `configs/<name>/gce.yaml` by instance family.

**Fallback — guided Q&A.** When no tested workspace exists, walk the user through the schema fields (`compute-config-schema.yaml`) — head/worker instance types, autoscaler bounds, spot, cross-zone autoscaling.

## Step 3: Test script

Write `tests/<name>/tests.sh`. Two shapes — pick by template type:

- **Notebook via papermill** (most common) — for `.ipynb` templates. See `tests/audio-dataset-curation-llm-judge/tests.sh`.
- **Custom script** — for templates shipping a `.py` entrypoint or service. See `tests/asynchronous_inference/tests.sh`.

Design the test to mirror the user flow, with minimal-impact shortcuts to keep CI fast:

- **Skip redundant paths.** If two code paths exercise the same logic (e.g., a local Ray Serve deployment and the same app deployed as an Anyscale Service), test only one.
- **Prefer cheap GPUs.** A10 (`g5.*`) or L4 (`g6.*` / `g2-*-nvidia-l4-*`) over A100/H100 — faster to provision, cheaper to run.
- **Shrink work via env vars.** Lower dataset size, epochs, or warmup through environment variables read in the notebook. Keep the user-facing code path clean (e.g., `epochs = int(os.getenv("EPOCHS", 100))` reads as real config, not test scaffolding).

## Step 4: Validate

Apply `format.md` to the new template.

## Handoff

1. **Open a PR** with the new template.
2. **Run CI** — have the user comment `/test-template` on the PR.
3. **After merge, publish to dev / staging / prod** via the Buildkite tmpl-publish pipeline. Offer to run it for the user (see `publish.md`), or point them at it to run themselves.
4. **Reference the template in `anyscale/product`** so it surfaces in the Anyscale console frontend.
