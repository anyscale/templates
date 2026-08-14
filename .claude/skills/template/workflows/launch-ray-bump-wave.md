# Launch a Ray-bump wave

Firing the `template-updater` Cursor agents that open the per-template bump PRs. This is the
step **before** `publish-ray-bump-fanout.md` — it creates the PRs that one lands and publishes.

Three stages, each gated by a human:

| Stage | Fires | Produces |
|---|---|---|
| Prep | `ray-version-prep` (daily cron) | a `[ray-prep] Ray <v>` PR — `build_arg_sets` + an image freeze per tracked image |
| Wave | pushing the `ray-fanout-<v>` tag | one Cursor agent per behind template → one draft PR each, labeled `ray-update` |
| Publish | you, per `publish-ray-bump-fanout.md` | the bumps merged and published to prod |

## 1. Merge the prep PR

Nothing can fan out until the target version has a freeze for **every** image in
`dependencies/images/tracked-images.txt` — `latest-depset-version.py --require <v>` fails closed,
and the fanout runs that same gate. The prep PR says up front whether it passes; if it opens with
**"⚠️ Not ready to fan out"**, merge it anyway (the freezes it does carry are correct) and wait for
the next scheduled run to add the rest once Anyscale publishes those images.

Merging is the approval gate. Nothing is automatic past this point.

Prep failed with **"needs human"** instead of opening a PR? The tracked-image matrix moved — a
tracked image has no tag for the new version. That is `upgrade-dependencies.md`, not this doc.

## 2. Preview the wave

Always look at the delta before firing. Run `ray-bump-fanout` via **workflow_dispatch** — it
previews by default (`dry_run: true`), resolves the version, and writes the list of templates
behind it to the run summary without POSTing anything.

Locally, the same list, no credentials needed:

```bash
uv run python3 scripts/ray-bump/trigger-cursor-bump.py --all -v <v> --list
```

The delta is version-aware against `BUILD.yaml`, so templates already at `<v>` drop out and
`maintained: false` entries never appear. Expect roughly the whole fleet on a fresh minor.

## 3. Fire it

```bash
git tag ray-fanout-<v> && git push origin ray-fanout-<v>
```

The tag *is* the trigger and it encodes the version, so it cannot double-fire — a second push of
the same tag is rejected. One POST per template, one agent, one draft PR.

`workflow_dispatch` with `dry_run: false` does the same thing without a tag. Prefer the tag: it
leaves a record of which version was fanned out and when.

Needs `CURSOR_TEMPLATE_UPDATER_WEBHOOK` and `CURSOR_TEMPLATE_UPDATER_AUTH_TOKEN`. The agent's
prompt is **not** in this repo — it lives on the Cursor automation dashboard, which is its single
source of truth; the payload is only `{template_name, ray_version}`. Each agent then runs
`bump-ray-version.md` non-interactively.

## 4. Re-firing after a partial wave

Safe. The delta recomputes from `BUILD.yaml`, so anything already landed at `<v>` is skipped and
only the stragglers get new agents. Templates whose PR is still open but unmerged **will** get a
second agent — close the stale PR first, or name the templates explicitly instead of `--all`.

To retry a handful by hand:

```bash
scripts/ray-bump/trigger-cursor-bump.py -v <v> <name> <name>            # preview
scripts/ray-bump/trigger-cursor-bump.py -v <v> <name> <name> --execute  # fire
```

Then land and publish them per `publish-ray-bump-fanout.md`.
