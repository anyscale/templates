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

## Important: Schema strictness

Every entry in both `BUILD.yaml` and `workspace-templates.yaml` **must exactly match** the schemas defined in the [publishing guide](https://www.notion.so/anyscale-hq/Anyscale-Template-publishing-guide-V3-317027c809cb801bbf09fed1cb9bf71d). All fields in the schema are required.

- **Do not guess values.** If a field's value is ambiguous or unknown, add a `# TODO: <explain what's needed>` comment instead of inventing a value.
- **`cluster_env` must be one of two forms** — never mix them, never leave it empty:
  - `image_uri: anyscale/ray:...` (Anyscale public image)
  - `byod:` block with `docker_image` + `ray_version` (external/custom image)
- The same applies to `workspace-templates.yaml`: every field in the schema (title, description, base_url, emoji, complexity, icon_type, icon_bg_color, labels, for_gallery, ld_flag, etc.) must be present. If you don't know the right value, use a `# TODO:` comment.

---

## Phase 1: Add entries to BUILD.yaml

**Goal:** For each old-workflow template hosted in `anyscale/templates`, add an entry to `BUILD.yaml`.

### 1.1 Identify candidates

Search [`backend/workspace-templates.yaml`](https://github.com/anyscale/product/blob/master/backend/workspace-templates.yaml) for entries with:
```
base_url: https://github.com/anyscale/templates/tree/...
```

These are the templates to migrate. Templates hosted in `ray-project/ray` (e.g., `base_url: https://github.com/ray-project/ray/tree/master/`) are a **separate migration** and out of scope here.

### 1.2 Check what already exists in BUILD.yaml

Cross-reference the candidates against existing entries in [`BUILD.yaml`](https://github.com/anyscale/templates/blob/main/BUILD.yaml). Skip any template that already has an entry (already migrated).

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

#### Compute config reuse

Before referencing a template-specific compute config, check whether it is identical to one of the shared configs:
- `configs/basic-single-node/` — head-only, `auto_select_worker_config: true` (GCP: `n2-standard-8`)
- `configs/basic-serverless-config/` — head-only, `auto_select_worker_config: true` (GCP: `n1-standard-8`)

If the template's compute config content matches a shared config, reference the shared one instead. Only keep template-specific configs when they define custom worker node types, GPU instances, or other non-default settings.

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
| `maximum_uptime_minutes` | Dead in template flow: parsed but never surfaced to the API or frontend (frontend hardcodes 1440 min). Safe to remove. |
| `for_fine_tuning` | Dead: full plumbing exists but no caller ever passes `for_fine_tuning=true`. See Deprecated fields section. |
| `for_private_endpoints_home_page` | Dead: full plumbing exists but no caller ever passes it as `true`. See Deprecated fields section. |
| `logo_ids` | Dead: serialized into API response but no frontend component reads or renders it. Template card icons use `icon_type`/`icon_bg_color` instead. See Deprecated fields section. |
| `emoji` | Dead: serialized into API response but no frontend component reads or renders it. Cards use `icon_type`/`icon_bg_color` instead. See Deprecated fields section. |

### 2.3 Keep these fields in workspace-templates.yaml

These fields stay because they control console display and gallery behavior.

**CRITICAL: Field order and completeness.** Every migrated entry MUST:
1. Include **all** fields listed below — even for `ld_flag: true` templates. If a value is unknown, add a `# TODO:` comment but still keep the field's position in the order.
2. Follow **exactly this field order** — no exceptions, no reordering:

```yaml
<template-name>:
  title: ...
  description: ...
  base_url: ascommon:///templates/<template-name>/latest
  mins_to_complete: ...
  complexity: ...                # one of: Beginner, Intermediate, Advanced
  icon_type: ...                 # use # TODO: if unknown
  icon_bg_color: "..."           # use # TODO: if unknown
  for_gallery: true              # false only if template is hidden (ld_flag: true)
  labels:                        # MUST be sorted alphabetically (A-Z)
    - ...                        # Only these values have functional effect:
                                 #   Filter pills: Parallel processing, Distributed training,
                                 #     Batch inference, Data processing, Online inference, Fine-tuning,
                                 #     Text, Image, Audio, Video, Tabular,
                                 #     Scikit-learn, XGBoost, Pytorch, Tensorflow,
                                 #     Ray Core, Ray Data, Ray Train, Ray Serve
                                 #   Recommendation scoring adds: LLMs, Stable Diffusion, Parallel Processing
                                 #   Card chips: only Ray Core, Ray Data, Ray Train, Ray Serve render visually
                                 #   Any other value only matches free-text search
  oa_group_name: intro           # ONLY include if value is "intro" — omit entirely otherwise
  ld_flag: false                 # ALWAYS the last field
```

### 2.4 Verify labels against template content

Before finalizing labels, **read the actual template content** (README.md, notebooks) for each template to verify:

1. **`mins_to_complete`** — check if the template mentions a time estimate. Use that value, or estimate from content length.
2. **`complexity`** — check if prerequisites or difficulty are mentioned. Map to Beginner/Intermediate/Advanced.
3. **Labels accuracy** — verify that labels match the Ray libraries and frameworks actually used in the code:
   - If the template imports `ray.data`, add "Ray Data"
   - If the template imports `ray.train`, add "Ray Train"
   - If the template imports `ray.serve`, add "Ray Serve"
   - If the template uses `ray.remote`/`ray.actor`, add "Ray Core"
   - If the template uses PyTorch, add "Pytorch"
   - If the template processes images, add "Image"; text → "Text"; etc.
   - Remove labels that don't match the actual content
4. **Labels sort order** — labels MUST be sorted alphabetically (A-Z). Double-check after any additions or removals.

### 2.5 Verify consistency

After editing, confirm:
- The `<template-name>` key in `workspace-templates.yaml` matches the `name` in `BUILD.yaml`
- The `base_url` path segment matches the `name` in `BUILD.yaml`
- No `cluster_env`, `compute_config`, or `path` fields remain on migrated entries
- All fields appear in the exact order specified in 2.3
- `ld_flag` is the last field in every entry
- Labels are sorted alphabetically
- No deprecated fields remain (`emoji`, `logo_ids`, `oa_group_name` unless "intro", `maximum_uptime_minutes`, `for_fine_tuning`, `for_private_endpoints_home_page`)

---

## File locations

| File | Link | Repo | Purpose |
|---|---|---|---|
| `BUILD.yaml` | [anyscale/templates/BUILD.yaml](https://github.com/anyscale/templates/blob/main/BUILD.yaml) | `anyscale/templates` | Template build definitions (content, env, compute, tests) |
| `workspace-templates.yaml` | [anyscale/product/backend/workspace-templates.yaml](https://github.com/anyscale/product/blob/master/backend/workspace-templates.yaml) | `anyscale/product` | Console gallery metadata and `base_url` routing |

---

## Deprecated fields in workspace-templates.yaml

### `maximum_uptime_minutes`

**Dead field.** The YAML value is parsed but never surfaced to the API — `filter_and_format_templates_standalone()` in `workspaces_service.py` omits it when constructing the `GlobalWorkspaceTemplate` response. The frontend never reads it from the template and instead hardcodes `1440` (24h) in `utils.ts:13`, which is what gets sent to the backend on every Launch. The enforcement mechanism itself is real (Go cluster operator force-terminates clusters via `max_uptime_threshold_seconds` in the `AnyscaleClusterSpec` proto), but the YAML value never reaches it. Safe to remove.

### `for_fine_tuning`

**Dead field.** The full plumbing exists — YAML value, `is_for_fine_tuning()` method in `workspaces_service.py:268`, filter logic in `filter_and_format_templates_standalone()`, API query parameter on the `GET /templates` endpoint, and a `forFineTuning` param in the frontend hook `useListWorkspaceTemplates`. However, **no caller anywhere passes `for_fine_tuning=true`** — neither the frontend pages (`TemplatesPage`, `HomePage`) nor any CLI/Go code ever triggers the filter. The `GlobalWorkspaceTemplate` response model also does not include the field. Only two templates declare it (`skyrl`, `finetune-stable-diffusion`). Safe to remove.

### `for_private_endpoints_home_page`

**Dead field.** Three templates declare it (`langchain-agent-ray-serve`, `unstructured_data_ingestion`, `asynchronous_inference`). The backend has `is_for_private_endpoints_home_page()` in `workspaces_service.py:271`, a filter branch in `filter_and_format_templates_standalone()`, and a `for_private_endpoints_homepage` query parameter on the `GET /templates` router. However, **no frontend page, CLI command, or Go caller ever passes `for_private_endpoints_homepage=true`**. The `GlobalWorkspaceTemplate` response model does not include the field. Appears to be a vestige of an older "private endpoints home page" UI that no longer exists. Safe to remove.

### `emoji`

**Dead field.** Parsed by `workspaces_service.py:759`, included in the `GlobalWorkspaceTemplate` model (`experimental_workspaces.py:187`), and serialized into every API response. However, **no frontend component ever reads `.emoji`** from the template object. `TemplateCard`, `TemplateCardGrid`, and all page components have zero references to it. Template cards use `icon_type`/`icon_bg_color` for their visual icon instead. Safe to remove.

### `logo_ids`

**Dead field.** Defined in 32 templates with values like `ray`, `meta`, `hugging-face`, etc. Parsed by `workspaces_service.py:762`, included in the `GlobalWorkspaceTemplate` model (`experimental_workspaces.py:195`), and serialized into every API response. However, **no frontend component ever reads `logoIds` from the template object**. `TemplateCardGrid` does not pass it to `TemplateCard`, and `TemplateCard` has no `logoIds` prop. Template card icons are driven entirely by `icon_type` and `icon_bg_color`, which appear to have replaced `logo_ids`. Safe to remove.

### `oa_group_name`

**Partially alive.** Has exactly one real caller: `HomePage.tsx:27` queries `oaGroupNames: ["intro"]` to fetch the 5 intro templates for the home page welcome section. This flows through the API query parameter (`experimental_workspaces_router.py:293`) to `filter_and_format_templates_standalone()` in `workspaces_service.py:740` which filters templates server-side by their `oa_group_name` YAML value. However, only the `"intro"` group is ever queried — the other 4 groups (`gen_ai`, `more`, `data`, `ray_summit`) are never requested by any caller. The field is also included in the `GlobalWorkspaceTemplate` API response (`experimental_workspaces.py:199`) but no frontend component ever reads it from the response. **Keep the field for templates with `oa_group_name: intro`; the rest are dead weight but harmless.**

---

## Out of scope

- Templates hosted in `ray-project/ray` repo (different migration path)
- Templates already using `ascommon:///` base_url (already migrated)
- Publishing to Buildkite / S3 (separate step, done after PRs merge)
- Creating PRs (done manually with the user)
