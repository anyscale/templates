# Archived templates

Good template content we don't want to lose, but **no longer published, tested, or maintained** — not built by CI and not surfaced in the console gallery.

Layout:
- `events/<year>-<event>/<template>/` — past event / workshop material (Ray Summit, hands-on labs, bootcamps), year-prefixed so it sorts chronologically.
- `reference/<template>/` — superseded-but-useful examples not tied to an event.

Each archived template keeps two subfolders:
- `configs/` — its compute / cluster config (the launch/CI bit), if it had one.
- `content/` — the actual template: README, notebooks, code, and the authors' own configs.

Nothing here is covered by `BUILD.yaml` or the template test/publish pipelines.
