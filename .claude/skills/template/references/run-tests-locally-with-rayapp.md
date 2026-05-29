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
export ANYSCALE_HOST="https://console.anyscale-staging.com"  # always staging — local tests + fixes never run against prod
export ANYSCALE_CLI_TOKEN="<staging token>"                  # from the staging console's API tokens
pip install anyscale==0.26.87
```

**Always staging.** Local rayapp runs against **staging** (`https://console.anyscale-staging.com`) only — never prod, even if the CI run used prod. Use a staging `ANYSCALE_CLI_TOKEN` (ask the user if you don't have one). If staging itself fails — auth or the test — treat it as an **infra issue and ignore it**; don't chase the failure on prod. (Prod is read-only-exceptional — see `testing-template.md` Recovery.)

## Run a template's tests

```bash
rayapp test <template-name>
```

## How rayapp test works

1. Creates a **staging** workspace using the BUILD.yaml image.
2. **Flattens the template and test content into one directory** — the *contents* of `templates/<dir>/` and `tests/<name>/` land side by side at the workspace root (not as subfolders).
3. Runs `bash tests.sh` from that root, so `tests.sh` finds the template files (e.g. `README.ipynb`) right beside it via bare relative paths.

Example — everything ends up co-located, then `tests.sh` runs:

```
Repo                              Workspace root (flattened — same level)
templates/my-template/              README.ipynb     ← from templates/my-template/
  README.ipynb                      util.py
  util.py                           Dockerfile
  Dockerfile                        tests.sh         ← from tests/my-template/
tests/my-template/                  $ bash tests.sh  →  papermill README.ipynb ...
  tests.sh
```

Replicate this co-location if you ever run `tests.sh` by hand outside rayapp.

## Custom images (byod)

Pre-build with `anyscale image build` and update BUILD.yaml's `byod.docker_image` before testing. When finished iterating, run `.claude/skills/template/scripts/push-custom-image-to-gcp.sh <dockerfile-dir> <image-name> <ray-version>` and update BUILD.yaml again with the GCP URI.
