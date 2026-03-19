---
name: migrate-template
description: Migrate templates from old GitHub-branch workflow to new ascommon:// publishing workflow. Covers both BUILD.yaml (anyscale/templates) and workspace-templates.yaml (anyscale/product).
---

# Template Migration Workflow

Migrate existing templates from the **old workflow** (GitHub branch-based `base_url`) to the **new workflow** (`ascommon://` S3-based `base_url`).

Reference: https://www.notion.so/anyscale-hq/Anyscale-Template-publishing-guide-V3-317027c809cb801bbf09fed1cb9bf71d

---

## Context

### Old workflow (to migrate away from)
Templates hosted via GitHub branch URLs in `workspace-templates.yaml`:
```yaml
base_url: https://github.com/anyscale/templates/tree/<some-branch>/
path: templates/<template-dir>
```
Content is pulled directly from a frozen GitHub branch. Cluster env and compute config live in `workspace-templates.yaml`.

### New workflow (target)
Templates published via Buildkite to S3, referenced by `ascommon://` URL:
```yaml
base_url: ascommon:///templates/<template-name>/latest
```
Content, cluster env, compute config, and tests are defined in `BUILD.yaml` in the `anyscale/templates` repo. The `workspace-templates.yaml` in `anyscale/product` only holds display/gallery metadata.

---

## Phase 1: Add entries to BUILD.yaml

**Goal:** For each old-workflow template hosted in `anyscale/templates`, add an entry to `BUILD.yaml`.

### 1.1 Identify candidates

Search `workspace-templates.yaml` for entries with:
```
base_url: https://github.com/anyscale/templates/tree/...
```

These are the templates to migrate. Templates hosted in `ray-project/ray` (e.g., `base_url: https://github.com/ray-project/ray/tree/master/`) are a **separate migration** and out of scope here.

### 1.2 Check what already exists in BUILD.yaml

Cross-reference the candidates against existing `BUILD.yaml` entries. Skip any template that already has an entry (already migrated).

### 1.3 Add BUILD.yaml entries

For each candidate not yet in `BUILD.yaml`, add an entry following this exact format:

```yaml
# owner: @owner-name
- name: <template-name>
  dir: templates/<template-dir>
  cluster_env:
    # Option A: Anyscale public image
    image_uri: anyscale/ray:<ray-version>-<image-type>-<python-version>-<cuda-version>
    # Option B: External/custom image (byod)
    #byod:
    #  docker_image: <full-image-uri>
    #  ray_version: <ray-version>
  compute_config:
    GCP: configs/<config-path>/gce.yaml
    AWS: configs/<config-path>/aws.yaml
  test:
    tests_path: tests/<template-name>/
    command: bash tests.sh
    timeout_in_sec: 900
```

#### Where to get the values

| Field | Source |
|---|---|
| `name` | The YAML key from `workspace-templates.yaml` (e.g., `text-embeddings`) |
| `dir` | The `path` field from `workspace-templates.yaml` (e.g., `templates/text-embeddings`) |
| `cluster_env` | Copy from the `cluster_env` block in `workspace-templates.yaml` |
| `compute_config` | Copy from the `compute_config` block in `workspace-templates.yaml` |
| `owner` | The `# owner:` comment from `workspace-templates.yaml` |
| `test` | Check if `tests/<template-name>/` exists in the repo. If yes, reference it. If no test dir exists, you'll need to flag this. |

#### Converting `build_id` to `image_uri`

Old entries may use `build_id` instead of `image_uri`. Convert using string manipulation:

```
build_id: anyscaleray2501-py311        -> image_uri: anyscale/ray:2.50.1-py311
build_id: anyscaleray2530-slim-py312-cu129 -> image_uri: anyscale/ray:2.53.0-slim-py312-cu129
build_id: anyscaleray-ml2490-py39-gpu  -> image_uri: anyscale/ray-ml:2.49.0-py39-gpu
build_id: anyscaleray-llm2540-py311-cu128 -> image_uri: anyscale/ray-llm:2.54.0-py311-cu128
```

Pattern:
- `anyscaleray` prefix -> `anyscale/ray:`
- `anyscaleray-ml` prefix -> `anyscale/ray-ml:`
- `anyscaleray-llm` prefix -> `anyscale/ray-llm:`
- Version digits (e.g., `2501`) -> dotted version (e.g., `2.50.1`)
- Remaining components (`slim`, `py312`, `cu129`, `gpu`) are kept as-is, joined by `-`
- If a component is missing (no image-type, no cuda), just omit it

Reference for valid images: https://docs.anyscale.com/reference/base-images

#### Decision gates

Before adding an entry, verify:
1. The template directory (`templates/<dir>`) exists in the `anyscale/templates` repo on main (or will be merged)
2. Compute configs exist at the referenced paths
3. A test directory exists (create a placeholder `tests.sh` if needed)
4. The `image_uri` or `byod` image is valid and accessible

---

## Phase 2: Update workspace-templates.yaml

**Goal:** Update each migrated template's entry in `workspace-templates.yaml` to use the new `base_url` format, and remove fields that are now managed by `BUILD.yaml`.

### 2.1 Update `base_url`

Replace:
```yaml
base_url: https://github.com/anyscale/templates/tree/<branch>/
path: templates/<template-dir>
```

With:
```yaml
base_url: ascommon:///templates/<template-name>/latest
```

Where `<template-name>` matches the `name` field in `BUILD.yaml` (Step 1.3).

### 2.2 Remove deprecated/moved fields

Remove these fields from the `workspace-templates.yaml` entry (they now live in `BUILD.yaml` or are deprecated):

| Field to remove | Reason |
|---|---|
| `path` | No longer needed; `base_url` is self-contained with `ascommon://` |
| `cluster_env` | Now defined in `BUILD.yaml` |
| `compute_config` | Now defined in `BUILD.yaml` |
| `maximum_uptime_minutes` | Deprecated field |

### 2.3 Keep these fields in workspace-templates.yaml

These fields stay because they control console display and gallery behavior:

```yaml
<template-name>:
  emoji: ...
  title: ...
  description: ...
  base_url: ascommon:///templates/<template-name>/latest
  ld_flag: false
  mins_to_complete: ...
  complexity: ...
  icon_type: ...
  icon_bg_color: "..."
  for_gallery: true
  labels:
    - ...
  oa_group_name: ...         # if present
  logo_ids:                  # if present
    - ...
  for_fine_tuning: ...       # if present
  for_private_endpoints_home_page: ...  # if present
```

### 2.4 Verify consistency

After editing, confirm:
- The `<template-name>` key in `workspace-templates.yaml` matches the `name` in `BUILD.yaml`
- The `base_url` path segment matches the `name` in `BUILD.yaml`
- No `cluster_env`, `compute_config`, or `path` fields remain on migrated entries

---

## Files involved

| File | Repo | Purpose |
|---|---|---|
| `BUILD.yaml` | `anyscale/templates` | Template build definitions (content, env, compute, tests) |
| `backend/workspace-templates.yaml` | `anyscale/product` | Console gallery metadata and `base_url` routing |

---

## Out of scope

- Templates hosted in `ray-project/ray` repo (different migration path)
- Templates already using `ascommon:///` base_url (already migrated)
- Publishing to Buildkite / S3 (separate step, done after PRs merge)
- Creating PRs (done manually with the user)
