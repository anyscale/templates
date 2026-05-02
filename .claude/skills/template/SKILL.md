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

Infer the case based on the current `BUILD.yaml` image URI:

- **Anyscale base** (`image_uri: anyscale/ray:...`): bump `image_uri` to the new Ray version.
- **Anyscale custom on GCP** (`byod.docker_image: us-docker.pkg.dev/...`): Bump the Dockerfile `FROM` to the new Ray version → run `.claude/skills/template/scripts/publish-custom-image.sh <dockerfile-dir> <image-name> <ray-version>` (use the entry's `name` field as `<image-name>` — the validator requires `<registry>/<name>:<ray-version>`) → update `byod.docker_image` and `ray_version`. If CI later fails, `/fix` will iterate (via `anyscale image build` for fast dev) and we'll republish to GCP before the next CI run.
- **Third-party** (e.g. `novaskyai/skyrl-train-ray-2.48.0-py3.12-cu12.8`): same repo, pick the latest available tag with the highest Ray version, update `byod.docker_image` and `ray_version`. Don't swap to `anyscale/ray`.

For valid `anyscale/ray*` (including `ray-llm`) image tags (used in `image_uri` or as a Dockerfile `FROM`), see `https://docs.anyscale.com/reference/base-images/ray-<vXXX>/<pyYY>` — substitute Ray version with dots removed and python variant (e.g. Ray 2.55.1 + py3.11 → `ray-2551/py311`).
