# Image URI cases

Infer the case based on the current `BUILD.yaml` image URI:

- **Anyscale base** (`image_uri: anyscale/ray:...`): bump `image_uri` to the new Ray version.
- **Anyscale custom on GCP** (`byod.docker_image: us-docker.pkg.dev/...`): Bump the Dockerfile `FROM` to the new Ray version → `docker build + push` per `custom-image-publishing.md` → update `byod.docker_image` and `ray_version`. If CI later fails, `/fix` will iterate (via `anyscale image build` for fast dev) and we'll republish to GCP before the next CI run.
- **Third-party** (e.g. `novaskyai/skyrl-train-ray-2.48.0-py3.12-cu12.8`): same repo, pick the latest available tag with the highest Ray version, update `byod.docker_image` and `ray_version`. Don't swap to `anyscale/ray`.
