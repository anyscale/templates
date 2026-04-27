# Publishing Custom Images to Anyscale Google Artifact Registry

## Setup

```bash
export IMAGE_NAME=<image-name>        # e.g., template_deployment-serve-llm
export RAY_VERSION=<X.Y.Z>            # e.g., 2.54.1
export REGISTRY=us-docker.pkg.dev/anyscale-workspace-templates/workspace-templates
```

## Build and push

```bash
# Always use --platform linux/amd64 to match Anyscale VM architecture
docker build --platform linux/amd64 -t $REGISTRY/$IMAGE_NAME:$RAY_VERSION .

# Push (assumes docker is already authenticated with the registry)
docker push $REGISTRY/$IMAGE_NAME:$RAY_VERSION
```

If the push fails with an authentication error, ask the user to authenticate:
```bash
# Interactive (dev machine):
gcloud auth configure-docker us-docker.pkg.dev

# Non-interactive (CI / service account):
gcloud auth activate-service-account --key-file=<key.json>
gcloud auth configure-docker us-docker.pkg.dev --quiet
```

Do NOT attempt to handle credentials yourself. If auth fails, stop and tell the user.
