---
name: template
description: Create, maintain, format, and publish Anyscale console templates. Use for a new template, a Ray-version bump, BUILD.yaml edits, repo conventions, or publishing to dev/staging/prod.
---

# Template skill

A console template = a `BUILD.yaml` entry + `templates/<name>/` content + `configs/<name>/` compute configs + a `tests/<name>/` test. Every template under `templates/` is tested (the validator enforces it); `archive/` holds **test-exempt** content — retired/past-event templates, plus fast untested iteration that can still be published — via `workflows/archive-template.md`.

**How to use:** create/update are **interactive** — interview the user, explain at the point of need, and ask for any missing input. **ray-bump is non-interactive** (the `template-updater` Cursor cloud agent; signals: `.cursor/` setup, a `cursor/...` branch, an automation trigger): never prompt, use defaults; on missing input or preflight failure, post to the PR and stop. Track multi-step runs with your task tool.

## Pick a workflow

Invoked bare (just `/template`) by a human — i.e. **not** the ray-bump automation (no `.cursor/` / `cursor/...` / trigger signals)? Ask first: are you **creating** a new template or **updating** an existing one? Then follow that file — don't default to one. (Create's first step interviews you for everything else.)

- **Create** a new template → `workflows/create-template.md`
- **Update** content/config, no Ray bump → `workflows/update-template.md`
- **Ray-version bump** (non-interactive) → `workflows/bump-ray-version.md`
- **Upgrade locked dependencies** (recompile depsets for a new Ray version) → `workflows/upgrade-dependencies.md`
- **Archive** a retired or past-event template → `workflows/archive-template.md`

Supporting: `references/conventions.md` · `references/dependencies.md` · `references/testing-template.md` · `references/publish-to-backend.md` · `references/run-tests-locally-with-rayapp.md` · `schemas/build-yaml-schema.yaml` · `schemas/compute-config-schema.yaml` · `.claude/skills/template/scripts/push-custom-image-to-gcp.sh`.

## Image URI cases

A template's image is one of three types:

- **Anyscale base** — `cluster_env.image_uri: anyscale/ray:...`. Tags: `https://docs.anyscale.com/reference/base-images/ray-<vXXX>/<pyYY>` (Ray 2.55.1 + py3.11 → `ray-2551/py311`). Same tag space for `anyscale/ray-llm:...` and Dockerfile `FROM` lines.
- **Anyscale custom on GCP** — `cluster_env.byod.docker_image: us-docker.pkg.dev/anyscale-workspace-templates/workspace-templates/<name>:<ray-version>`. Built from the Dockerfile, pushed to our GCP Artifact Registry.
- **Third-party** — `cluster_env.byod.docker_image` from another registry; not ours.

Bump procedure per case → `workflows/bump-ray-version.md`.
