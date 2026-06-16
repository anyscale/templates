# Archive a template

Retire a template that's no longer maintained but worth keeping — past **event/workshop** material (Ray Summit, bootcamps, labs) or a superseded-but-instructive example. It moves to `archive/`: kept in-repo, but not built, tested, or published, and not in the console gallery. The `archive/` layout (events vs reference, the `configs/` + `content/` split, year-prefixing) lives in **`archive/README.md`**.

- **Archive vs delete:** archive content worth keeping; just delete dead duplicates, cruft, or anything preserved elsewhere (git history keeps it either way).
- **Publishing** a template *without* the test gate (urgent event fix) is a different intent — see "Publish without the test gate" in `../references/publish-to-backend.md`.

## Steps — one PR per template

1. **Remove the `BUILD.yaml` entry** — CI stops building / testing / publishing it.
2. **Move the content** with `git mv` (history follows):
   - `templates/<name>/` → `archive/events/<year>-<event>/<name>/content/` (event) or `archive/reference/<name>/content/` (non-event).
   - `configs/<name>/` — the compute config, if any → the sibling `archive/.../<name>/configs/`.
   - The template's own in-dir configs (training / deploy / job YAML) stay inside `content/`.
3. **Sweep stragglers:** `grep -rn "<name>"` for cross-links and CI references; clean any that point at the moved template.
4. **Verify:** `pre-commit run --all-files` and premerge stay green.
