# Verified command sequence + gotchas

All verified against **anyscale 0.26** on **2026-07-10**, running the full scale-to-zero enforce
end-to-end on a real workspace (`gpu-worker-workspace`, GPU worker `g4dn.xlarge` min 1→0).

## Detect (read-only)

```bash
anyscale workspace_v2 list --no-interactive --json --cloud <cloud>                 # all workspaces
anyscale workspace_v2 list --no-interactive --json --state RUNNING --cloud <cloud> # running only
anyscale workspace_v2 get --id <id> --json --verbose                               # config.compute_config
```

## Enforce (terminate → update → start)

```bash
# 0. read live config + image (record image_uri for step 3)
anyscale workspace_v2 get --id <id> --json --verbose      # -> config.compute_config, config.image_uri

# 1. build autozero.json = live compute_config with min_nodes:0 on GPU worker group(s) only,
#    then GATE on it:
python3 validate_config_diff.py <id> autozero.json         # must print PASS (exit 0)

# 2. terminate (a RUNNING workspace CANNOT be updated) and wait
anyscale workspace_v2 terminate --id <id>
anyscale workspace_v2 wait --id <id> --state TERMINATED --timeout-s 480

# 3. register the config and attach it — MUST also pass --image-uri
anyscale compute-config create -n <name>-autozero -f autozero.json
anyscale workspace_v2 update <id> --compute-config <name>-autozero --image-uri <image_uri>

# 4. start and wait
anyscale workspace_v2 start --id <id>
anyscale workspace_v2 wait --id <id> --state RUNNING --timeout-s 600

# 5. verify
anyscale compute-config get <name>-autozero               # GPU group min_nodes: 0, max unchanged
```

## Gotchas (each cost a debugging round)

| Symptom | Cause | Fix |
|---|---|---|
| `Error: No such command 'list'` | CLI older than 0.26 | `pip install -U anyscale` |
| `json.decoder.JSONDecodeError: Extra data` / `EOF when reading a line` | `workspace_v2 list` pager concatenates per-page JSON, or blocks on "Press Enter" under a non-tty | pass `--no-interactive` |
| `Error: Workspace must be in the TERMINATED state to be updated` | can't change compute config on a RUNNING workspace | terminate → update → start (not a live edit) |
| `Error: If launching from App Build and Compute Template, both the build ID and compute template ID must be provided` | `update --compute-config` alone omits the image/build | also pass `--image-uri <current image>` |
| `workspace_v2 get` returns `compute_config` as a **string** (e.g. `name-autozero:3`) instead of the inline dict | after attaching a *named* config, get returns the name, not the inline body | read the body with `anyscale compute-config get <name>` |

## Notes

- `terminate` = stop the target's own cluster; the workspace persists and `/mnt` survives. It is
  one step of the sequence, NOT deleting the workspace.
- The one-time terminate→start is a real ~3–5 min restart. The payoff is ongoing: the GPU worker
  now autoscales from 0 instead of sitting pinned.
- `terminate`, `wait`, `get`, `start`, `status` take `--id` (or `--name`); `update` takes the
  workspace id positionally (`workspace_v2 update <id> ...`).
