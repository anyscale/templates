# Anyscale console templates

Reference templates that ship in the Anyscale console — dozens of working examples covering Ray Train, Ray Data, Ray Serve, RayLLM, fine-tuning, RAG, and more. Each one is a runnable notebook or script that a customer can launch end-to-end from the Anyscale UI.

## Repository layout

```
BUILD.yaml                # Manifest of every template (entry point, image, compute, test)
templates/<name>/         # Template content — notebook or .py + Dockerfile (if custom image)
tests/<name>/tests.sh     # Per-template smoke test, executed by CI
configs/<name>/           # Compute configs: aws.yaml + gce.yaml (or reuse basic-single-node/)
ci/                       # Schema validators + the README auto-generator
.claude/skills/template/  # Maintenance procedures (Ray bumps, formatting, publishing)
```

## Contributing a template

> For **Anyscale-platform-specific** content. If your template is generic Ray or a Ray library, please contribute upstream to [ray-project/ray](https://github.com/ray-project/ray/tree/master/doc/source/templates) instead.

1. **Template content** at `templates/<name>/`:
   - A `README.ipynb` (preferred) or `.py` walkthrough
   - A `Dockerfile` if you need a custom image — otherwise use a stock `anyscale/ray:...` image
   - **Don't hand-edit `README.md`** — it's auto-generated from the notebook by a pre-commit hook
2. **Test** at `tests/<name>/tests.sh` — runs in CI to confirm the template still works end-to-end
3. **Compute config:** most templates reuse `configs/basic-single-node/`. If you need custom hardware, add `configs/<name>/aws.yaml` + `configs/<name>/gce.yaml`
4. **`BUILD.yaml` entry** — schema in [`.claude/skills/template/references/build-yaml-schema.yaml`](.claude/skills/template/references/build-yaml-schema.yaml), strictly validated by `ci/validate_build_yaml.py` (also runs as a pre-commit hook)
5. **Custom image** (only if you set `cluster_env.byod`): build and push with [`.claude/skills/template/scripts/publish-custom-image.sh`](.claude/skills/template/scripts/publish-custom-image.sh)`<dockerfile-dir> <name> <ray-version>`

## Local development

```bash
pip install pre-commit==3.8.0 nbconvert==7.17.1 pyyaml==6.0.3 pydantic==2.13.3
pre-commit install                                # auto-fire hooks on git commit
pre-commit run --all-files                        # lint + schema + auto-README
python3 ci/validate_build_yaml.py --no-network    # offline BUILD.yaml validation
bash ./update_deps.sh --check                     # dependency lockfile up-to-date check
```

For `rayapp` (the local test runner), GCP/anyscale auth, and the full dev environment, see [`.cursor/install.sh`](.cursor/install.sh) — it's the source of truth.

## CI

Static checks (schema, paths, README generation, build) run on every push. To run a template's actual tests, comment **`/test-template <name>`** on the PR — accepts up to three names, fanned out in parallel.

## Updating an existing template

To bump Ray, format to repo conventions, or republish a custom image, use the `/template` skill at [`.claude/skills/template/`](.claude/skills/template/). Procedures and schema references live there.

## Running this repo with an AI agent

If you're using Cursor Cloud or Claude Code, see [`AGENTS.md`](AGENTS.md) for setup, required secrets, and the operational cheatsheet.
