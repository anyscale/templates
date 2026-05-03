---
name: template
description: Work with Anyscale console templates. Covers maintenance (Ray version bumps), formatting (repo conventions), and publishing (S3 via Buildkite). Use for any BUILD.yaml edit, image bump, or template lifecycle task.
---

# Template skill

Read the reference matching your task:

- **How to update** a template to a specific Ray release → `references/update.md`
- **How to format** a template to repo conventions → `references/format.md`
- **How to publish** a template to dev/staging/production (S3 via Buildkite) → `references/publish.md`
- **Publish a custom image** to our GCP artifact registry → run `.claude/skills/template/scripts/publish-custom-image.sh <dockerfile-dir> <image-name> <ray-version>`
- **How to reproduce CI test locally** (rayapp setup) → `references/local-testing.md`
- **BUILD.yaml schema guidance** lookup → `references/build-yaml-schema.yaml`
- **Compute config schema guidance** lookup → `references/compute-config-schema.yaml`

## Image URI cases

A template's image is one of three types, identified by its `BUILD.yaml` entry:

- **Anyscale base** — `image_uri: anyscale/ray:...`. Available tags: `https://docs.anyscale.com/reference/base-images/ray-<vXXX>/<pyYY>` — substitute dots-removed Ray version + python variant (e.g. Ray 2.55.1 + py3.11 → `ray-2551/py311`). The same tag space applies to `anyscale/ray-llm:...` and to Dockerfile `FROM` lines in custom images.
- **Anyscale custom on GCP** — `byod.docker_image: us-docker.pkg.dev/anyscale-workspace-templates/workspace-templates/<name>:<ray-version>`. Image is built from the template's Dockerfile and published to the Anyscale GCP Artifact Registry.
- **Third-party** — `byod.docker_image` from any other registry (e.g. `novaskyai/skyrl-train-ray-2.48.0-py3.12-cu12.8`). Not maintained by us.

For per-case bump procedures, see `references/update.md` Step 1.
