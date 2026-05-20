# Run tests locally

Install rayapp and run a template's tests on your dev machine. Useful for validating changes before opening a PR or reproducing a CI failure.

## Install rayapp

The rayapp version used in CI is pinned in `download_rayapp.sh` at the repo root (source of truth). Use the same version locally:

```bash
# Linux: just run the repo's CI script
bash download_rayapp.sh && sudo mv rayapp /usr/local/bin/

# macOS / other: pull the version from download_rayapp.sh, swap the platform
RAYAPP_VERSION=$(grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' download_rayapp.sh)
PLATFORM=$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
curl -sL "https://github.com/ray-project/rayci/releases/download/${RAYAPP_VERSION}/rayapp-${PLATFORM}" -o ~/.local/bin/rayapp && chmod +x ~/.local/bin/rayapp
```

## Set credentials

```bash
export ANYSCALE_CLI_TOKEN="$$(aws --region=us-west-2 secretsmanager get-secret-value --secret-id $$ANYSCALE_CLI_TOKEN_SECRET_NAME | jq -r .SecretString)"
export ANYSCALE_HOST="https://console.anyscale-staging.com"
pip install anyscale==0.26.87
```

If auth issues, stop here and ask user to provide valid authentication for the staging console.

## Run a template's tests

```bash
rayapp test <template-name>
```

## How rayapp test works

1. Creates a workspace using the BUILD.yaml image
2. Copies `templates/<dir>/` and `tests/<name>/` into the workspace
3. Runs `tests.sh` from the workspace root

## Custom images (byod)

Pre-build with `anyscale image build` and update BUILD.yaml's `byod.docker_image` before testing. When finished iterating, run `.claude/skills/template/scripts/publish-custom-image.sh <dockerfile-dir> <image-name> <ray-version>` and update BUILD.yaml again with the GCP URI.
