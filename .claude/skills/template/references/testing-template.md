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
- **Shrink runtime with env vars** — epochs, model size, dataset read from env (e.g. `epochs = int(os.getenv("EPOCHS", 100))` — reads as real config, not test scaffolding). **Target < 30 min per test.** Prefer cheap GPUs (A10 `g5.*`, L4 `g6.*`) over A100/H100.
- **`tests.sh` holds local-only orchestration** — serve run + readiness poll + shutdown, redis spin-up, hard gates, secret fetching — so the notebook stays clean. Serve + poll + `trap` shutdown example: `tests/deployment-serve-llm/tests.sh`.

## Validate — default (human / interactive)

Zero local setup. Comment `/test-template <name>` on the PR (up to 3 templates in parallel — AGENTS.md). This **only dispatches** the Buildkite `template-test` pipeline (workspace creation + the real test run), so:

- **Monitor via the Buildkite MCP** — the workspace, image pull, and test logs live there. `gh pr checks` shows only the dispatch step, not the test result.
- Green → done. Failure → **Recovery**.

The green path needs **no** local rayapp. (Recovery may: `/anyscale-platform-fix` iterates against `rayapp test` — see `run-tests-locally-with-rayapp.md` for setup.)

## Validate — advanced (Cursor cloud / local iteration)

`rayapp test <name>` runs the template's test on a **staging** workspace — setup in `run-tests-locally-with-rayapp.md`. This is what the fix-loop iterates against before re-pushing.

## Recovery

Read the Buildkite logs (via MCP) and classify:

- **Agent-fixable** — template code/notebook, Dockerfile, config, or BUILD.yaml-schema bug. Delegate to **`/anyscale-platform-fix`**, which iterates against `rayapp test <name>` on **staging** until green. (Interactive/human path only: if the skill is missing, `anyscale skills install -p claude-code -y -f` — needs `anyscale login`. In Cursor, preflight guarantees it.)
- **Infra** — workspace-creation timeout, Anyscale API/SSO errors, Buildkite / GitHub-Actions runner errors, **or staging itself failing**. **Don't retry blindly, and never switch to prod.** If `rayapp test <name>` passes locally on staging, trust that, summarize the infra failure on the PR, and hand off to a human.

**Stay on staging.** rayapp and `/anyscale-platform-fix` always target **staging** (`console.anyscale-staging.com`) — a staging auth or test failure (401/403, SSO, rejected token, flaky workspace) is **infra: ignore it, don't chase it on prod**. Prod is read-only-exceptional — use a prod token only to *collect logs/info* from a prod CI run, never to test or fix.
