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
export ANYSCALE_HOST="https://console.anyscale-staging.com"  # match CI: staging by default; prod (console.anyscale.com) only if CI fell back
export ANYSCALE_CLI_TOKEN="<token issued by that console>"   # must match the env in ANYSCALE_HOST
pip install anyscale==0.26.87
```

**Auth errors?** First, match what CI's rayapp did: in CI it runs against **staging** by default and falls back to **prod** (`https://console.anyscale.com`) only when staging has temporary failures — rare, but it happens. Reproduce against the *same* environment the CI run used — point `ANYSCALE_HOST` there and use an `ANYSCALE_CLI_TOKEN` issued for it. Prod and staging are separate consoles with separate tokens, so a mismatch fails auth; if you don't have a token for that env, ask the user.

## Run a template's tests

```bash
rayapp test <template-name>
```

## How rayapp test works

1. Creates a workspace using the BUILD.yaml image
2. Copies `templates/<dir>/` and `tests/<name>/` into the workspace
3. Runs `tests.sh` from the workspace root

## Custom images (byod)

Pre-build with `anyscale image build` and update BUILD.yaml's `byod.docker_image` before testing. When finished iterating, run `.claude/skills/template/scripts/push-custom-image-to-gcp.sh <dockerfile-dir> <image-name> <ray-version>` and update BUILD.yaml again with the GCP URI.
