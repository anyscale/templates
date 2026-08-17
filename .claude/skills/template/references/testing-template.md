# Tests

Two parts: **write** the test (`tests/<name>/tests.sh`, wired into the BUILD.yaml `test` block), then **validate** it by running CI.

## Writing tests

Canon — apply each:

- **Mimic the user.** Users run the whole notebook top to bottom, so the test does too.
- **`papermill --log-output --cwd .` is the canonical runner** — `--log-output` streams to the CI log and saves an `.out.ipynb` for post-hoc debugging; `--cwd .` runs from the template dir so relative paths resolve. Canonical form:
  ```bash
  papermill README.ipynb /tmp/<name>.out.ipynb --log-output --kernel python3 --cwd .
  ```
  (`tests/audio-dataset-curation-llm-judge/tests.sh` is a minimal 0-tag *structure* example; it omits `--cwd .` only because it uses no relative paths.)
- **Strip CI-only cells with a tag — only when needed.** If some cells can't run in CI (SSH keys, multi-GPU/H100), tag them `skip-in-ci` and remove them before papermill:
  ```bash
  jupyter nbconvert --to notebook README.ipynb \
    --TagRemovePreprocessor.enabled=True \
    --TagRemovePreprocessor.remove_cell_tags=skip-in-ci \
    --output /tmp/<name>.ci.ipynb
  papermill /tmp/<name>.ci.ipynb /tmp/<name>.out.ipynb --log-output --kernel python3 --cwd .
  ```
  Templates with no such cells use plain papermill — don't add a strip step you don't need.
- **No redundant service/job tests** when a local path exists. If the template demos both a local run and a Service/Job deployment of the same logic, test only the local path.
- **Shrink runtime with env vars** — epochs, model size, dataset read from env (e.g. `epochs = int(os.getenv("EPOCHS", 100))` — reads as real config, not test scaffolding). **Target < 30 min per test.** Prefer cheap GPUs (A10 `g5.*`, L4 `g6.*`) over A100/H100 — and on multi-GPU shapes, pair that with `enable_cross_zone_scaling` (`../workflows/create-template.md` step 4), or the cheap instance you picked is the one the zone has run out of.
- **Expand big archives on the node, not on `/mnt/cluster_storage`.** Shared storage charges a network round trip per file: fine for a few large sequential reads, ruinous for many small ones. Unpacking Boltz's 45k-file CCD there measured 1033s against 16s on `/mnt/local_storage`, where every node builds its own copy in parallel. Download once to shared storage, expand per node.
- **`timeout_in_sec` bounds the test, not the job** — cluster start and image pull sit outside it. It also sets the Buildkite job timeout, at `max(75, ceil(timeout_in_sec/60) + 30)` minutes, so inflating it to hide a slow test widens the CI budget with it.
- **`tests.sh` holds local-only orchestration** — serve run + readiness poll + shutdown, redis spin-up, hard gates, secret fetching — so the notebook stays clean. Serve + poll + `trap` shutdown example: `tests/deployment-serve-llm/tests.sh`.

## Validate — default (human / interactive)

Zero local setup. Comment `/test-template <name>` on the PR (up to 3 templates in parallel — AGENTS.md). This **only dispatches** the Buildkite `template-test` pipeline (workspace creation + the real test run), so:

- **Monitor via the Buildkite MCP** — the workspace, image pull, and test logs live there. `gh pr checks` shows only the dispatch step, not the test result.
- Green → done — but **confirm the notebook actually finished**. `rayapp test` has reported `Success: true` for a run whose SSH session dropped mid-notebook: papermill logged `Executing Cell 23` and no `Ending Cell 23`, the training cell never got a GPU, and the last four cells never ran. The log's `Executing Cell N` and `Ending Cell N` counts must match.
- Failure → **Recovery**.

The green path needs **no** local rayapp. (Recovery may: `/anyscale-platform-fix` iterates against `rayapp test` — see `run-tests-locally-with-rayapp.md` for setup.)

## Validate — advanced (Cursor cloud / local iteration)

`rayapp test <name>` runs the template's test on a **staging** workspace — setup in `run-tests-locally-with-rayapp.md`. This is what the fix-loop iterates against before re-pushing.

## The scheduled probe is not a source of truth

The daily `template-probe` pipeline is **not** a second opinion on your branch. It creates a workspace from
the **published** template artifact, then pushes in only `tests/<name>/tests.sh` from a clone of `main`. So
it pairs whatever production is serving with main's tests, and every merged template PR widens that gap
until the template is republished.

A red probe therefore has two very different meanings, and the common one is the boring one:

- the **published** template is genuinely broken, or
- the publish is simply **behind `main`** — the code is fine and the fix is to publish, not to edit.

Check the second before touching code. `/test-template` pushes your branch's actual files, so it is the only
signal that reflects what you wrote. Never chase a probe failure with a code change you can't reproduce
under `/test-template`.

## Recovery

Read the Buildkite logs (via MCP) and classify:

- **Agent-fixable (default — most failures)** — anything rooted in the template's own files: code/notebook, Dockerfile, config, or BUILD.yaml — **including a `test.timeout_in_sec` that's too low, or a slow/oversized dataset download the test does** (raise the timeout, or trim/cache the download). These are yours to fix, not to bail on. Delegate to **`/anyscale-platform-fix`**, which iterates against `rayapp test <name>` on **staging** until green. (Interactive/human path only: if the skill is missing, `anyscale skills install -p claude-code -y -f` — needs `anyscale login`. In Cursor, preflight guarantees it.)
- **Infra (external only — not fixable from the template's files)** — workspace-creation *platform* failures, Anyscale API/SSO errors, Buildkite / GitHub-Actions runner errors, **or staging itself failing**. A test that *times out because the template downloads or does too much* is **not** infra — that's a `timeout_in_sec`/download fix (agent-fixable, above). **Don't retry blindly, and never switch to prod.** If `rayapp test <name>` passes locally on staging, trust that, summarize the infra failure on the PR, and hand off to a human.

**Stay on staging.** rayapp and `/anyscale-platform-fix` always target **staging** (`console.anyscale-staging.com`) — a staging auth or test failure (401/403, SSO, rejected token, flaky workspace) is **infra: ignore it, don't chase it on prod**. Prod is read-only-exceptional — use a prod token only to *collect logs/info* from a prod CI run, never to test or fix.
