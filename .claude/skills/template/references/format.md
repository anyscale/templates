# Format a template to repo conventions

Validate the specified template against the conventions below. If the template doesn't follow them, format it so it does.

## Rules

- **Only validate and format existing files.** If required files are missing (compute configs, test scripts, etc.), report what's missing and ask the user to add them — do NOT generate them yourself.
- **Do not run tests.** This guide only checks formatting, not functionality.
- **Minimal changes.** Only fix what doesn't follow conventions. Don't refactor or "improve" code.

---

## Repository Structure

```
templates/
├── BUILD.yaml              # Template definitions
├── templates/<dir>/        # Template content (code, notebooks, Dockerfiles)
├── tests/<name>/           # Test scripts
├── configs/<name>/         # Compute configs (AWS/GCP)
└── .claude/skills/template/references/  # Schema docs
```

---

## What to validate

- **BUILD.yaml entry**: matches `build-yaml-schema.yaml`. A template must use either `image_uri` OR `byod` — never both. For image bumps, see SKILL.md ("Image URI cases").
- **Compute configs**: present at `configs/<name>/aws.yaml` and `configs/<name>/gce.yaml`. Schema in `compute-config-schema.yaml`.
- **Tests**: `tests/<name>/tests.sh` exists.

⚠️ **Compute configs use the OLD API format**, NOT the new ComputeConfig API. ALWAYS use existing entries under `configs/` as reference. Do NOT refer to the live anyscale docs — they only document the new schema. Legacy API references:
- ComputeTemplateConfig: https://docs.anyscale.com/ref/0.26.64/compute-config-api#computetemplateconfig-legacy
- ComputeNodeType:       https://docs.anyscale.com/ref/0.26.64/compute-config-api#computenodetype-legacy
- WorkerNodeType:        https://docs.anyscale.com/ref/0.26.64/other#workernodetype-legacy
- Resources:             https://docs.anyscale.com/ref/0.26.64/other#resources-legacy

**If anything is missing**: report to the user, do NOT generate from scratch unless explicitly requested.
