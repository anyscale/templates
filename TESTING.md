# Template Testing Guide

This guide explains how to test templates in this repository using the `rayapp test` CLI.

## Prerequisites

- `rayapp` binary in your `PATH` (built from the [rayci](https://github.com/ray-project/rayci) repository)
- `anyscale` CLI installed and in your `PATH`
- Environment variables set:
  - `ANYSCALE_HOST` — Anyscale API host
  - `ANYSCALE_CLI_TOKEN` — Anyscale authentication token

## Running Tests

```bash
# Test a specific template
rayapp test <template-name>

# Test all templates that have a test configuration
rayapp test all

# Use a custom BUILD file
rayapp test <template-name> --build path/to/BUILD.yaml
```

Templates without a `test` block in `BUILD.yaml` are skipped automatically.

## Adding a Test Configuration

Add a `test` block to your template entry in `BUILD.yaml`:

```yaml
- name: my-template
  emoji: 🚀
  title: My Template
  description: A sample template
  dir: templates/my-template
  cluster_env:
    build_id: anyscaleray2501-py311
  compute_config:
    AWS: configs/my-template/aws.yaml
  test:
    command: pip install pytest && pytest . -v
```

### Test Configuration Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `command` | Yes | — | Shell command to run in the workspace |
| `timeout_in_sec` | No | 3600 (1 hour) | Max execution time in seconds |
| `tests_path` | No | — | Relative path to a separate test directory to push to the workspace |

### Examples

**Notebook-based template** — test notebooks directly with `nbmake`:

```yaml
test:
  command: pip install nbmake==1.5.5 pytest==9.0.2 && pytest --nbmake . -s -vv
```

**Custom timeout** — for long-running tests:

```yaml
test:
  command: pip install pytest && pytest . -v
  timeout_in_sec: 7200  # 2 hours
```

**Separate test directory** — when tests live outside the template folder:

```yaml
test:
  command: pip install -r tests/requirements.txt && pytest tests/ -v
  tests_path: tests/my-template/ci/
  timeout_in_sec: 1800
```

> **Note on `tests_path`:** Only the *contents* of the folder at `tests_path` are copied into the workspace root — the folder itself is not preserved. For example, if `tests_path: tests/my-template/ci/` contains `test_foo.py`, it will appear as `test_foo.py` in the workspace root, not `tests/my-template/ci/test_foo.py`.

## What Happens During a Test

When you run `rayapp test my-template`, the following steps execute:

1. **Parse BUILD.yaml** — reads and validates all template definitions.
2. **Create compute config** — registers the AWS compute config with Anyscale (only AWS is supported for workspace testing).
3. **Create workspace** — spins up an empty Anyscale workspace with the specified image and compute config.
4. **Push template** — zips and uploads the template directory (`dir`) to the workspace, then unzips it.
5. **Push tests** *(if `tests_path` is set)* — zips and uploads the test directory contents, then unzips them into the workspace root.
6. **Run test command** — executes the `command` wrapped in `timeout <timeout_in_sec> bash -c '<command>'`.
7. **Cleanup** — terminates and deletes the workspace regardless of test outcome.

## Troubleshooting

- **"no templates with test configuration to run"** — the template exists in `BUILD.yaml` but has no `test` block. Add one.
- **"no templates to test"** — the template name doesn't match any entry in `BUILD.yaml`. Check spelling and the `--build` flag.
- **Timeout failures** — increase `timeout_in_sec` or optimize the test command. The default is 1 hour.
- **Test command failures** — the command runs via `bash -c`, so standard shell syntax works. Ensure dependencies are installed as part of the command (e.g., `pip install pytest && pytest .`).
