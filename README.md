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

## Recommended workflow: delegate to the `/template` skill

For most tasks — adding a template, bumping Ray, formatting to conventions, publishing a custom image — the fastest path is to ask Claude Code or Cursor to use the [`/template` skill](.claude/skills/template/). It's the canonical entry point for every procedure in this repo, and it knows the schema, image conventions, CI workflow, and validator rules.

Example prompts:

- *"Use `/template` to bump `deepspeed_finetune` to Ray 2.55.0."*
- *"Use `/template` to format my new `my-rag-pipeline` template against repo conventions."*
- *"Use `/template` to validate the `BUILD.yaml` entry I just added for `foo-bar`."*
- *"Use `/template` to publish a new custom image for `entity-recognition-with-llms` at Ray 2.54.1."*

The sections below describe what the skill (or you, manually) does under the hood — useful when you want to understand a step or work without an agent.

## Contributing a template

> For **Anyscale-platform-specific** content. If your template is generic Ray or a Ray library, please contribute upstream to [ray-project/ray](https://github.com/ray-project/ray/tree/master/doc/source/templates) instead.

1. **Template content** at `templates/<name>/`:
   - **Preferred:** a `README.ipynb` notebook — `README.md` is auto-generated from it on every commit via `nbconvert` (don't hand-edit the `.md`)
   - **Minimum:** a hand-written `README.md` (this is what's rendered in the Anyscale console preview)
   - A `Dockerfile` only if you need a custom image — otherwise reference a stock `anyscale/ray:...` image
2. **Test** at `tests/<name>/tests.sh` — runs in CI to confirm the template still works end-to-end
3. **Compute config:** most templates reuse `configs/basic-single-node/`. If you need custom compute, add `configs/<name>/aws.yaml` + `configs/<name>/gce.yaml`
4. **`BUILD.yaml` entry** — schema in [`.claude/skills/template/references/build-yaml-schema.yaml`](.claude/skills/template/references/build-yaml-schema.yaml), strictly validated by `ci/validate_build_yaml.py` (also runs as a pre-commit hook)
5. **Custom image** (only if you set `cluster_env.byod`): build and push with [`.claude/skills/template/scripts/publish-custom-image.sh`](.claude/skills/template/scripts/publish-custom-image.sh)`<dockerfile-dir> <name> <ray-version>`. You need permissions to push to Anyscale's public GCP Artifact registry.

## Local development

```bash
pip install -r requirements-dev.txt               # pinned dev deps (single source of truth)
pre-commit install                                # auto-fire hooks on git commit
pre-commit run --all-files                        # lint + schema + auto-README
python3 ci/validate_build_yaml.py --no-network    # offline BUILD.yaml validation
bash ./update_deps.sh --check                     # dependency lockfile up-to-date check
```

For `rayapp` (the local test runner), GCP/anyscale auth, and the full dev environment, see [`.cursor/install.sh`](.cursor/install.sh) — it's the source of truth.

## CI

Static checks (schema, paths, README generation, build) run on every push. To run a template's actual tests, comment **`/test-template <name>`** on the PR — accepts up to three names, fanned out in parallel.

## Running this repo with an AI agent

If you're using Cursor Cloud or Claude Code, see [`AGENTS.md`](AGENTS.md) for setup, required secrets, and the operational cheatsheet.
