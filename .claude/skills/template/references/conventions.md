# Format a template to repo conventions

Validate the template against the conventions below; if it doesn't follow them, format it so it does.

## Rules of engagement

- **Only validate and format existing files.** If required files are missing (compute configs, test scripts), refer to `../workflows/create-template.md` to learn how to generate them.
- **Do not run tests.** This guide checks formatting, not functionality — see `testing-template.md` for tests.
- **Minimal changes.** Fix only what breaks a convention; don't refactor or "improve" code.

## Repository structure

```
templates/
├── BUILD.yaml                            # Template definitions
├── templates/<name>/                     # Content (code, notebooks, Dockerfiles)
├── tests/<name>/                         # Test scripts
├── configs/<name>/                       # Compute configs (aws.yaml / gce.yaml)
└── .claude/skills/template/schemas/      # BUILD.yaml + compute-config schemas
```

## Conventions

Run the `check-build-yaml` hook first (see **Validate locally**) — it authoritatively covers BUILD.yaml schema, referenced paths, naming, and the legacy compute-config check. Fix what it reports, then verify:

- **BUILD.yaml entry** matches `../schemas/build-yaml-schema.yaml`. Under `cluster_env:`, use either `cluster_env.image_uri` OR `cluster_env.byod`, never both. Image taxonomy: SKILL.md "Image URI cases".
- **Compute configs** present at `configs/<name>/aws.yaml` and `configs/<name>/gce.yaml`. Schema: `../schemas/compute-config-schema.yaml`.
- **Test** present at `tests/<name>/tests.sh`.
- **Dependencies pinned.** Declare template deps in `templates/<name>/requirements.txt`, the BYOD `Dockerfile` (`RUN pip install "pkg==x.y.z"`), or notebook `!pip install` — with exact versions. (The repo-root `dependencies/` directory is Ray base-image constraint management, not per-template deps.)
- **README** — author content in `README.ipynb`; `README.md` is its `jupyter nbconvert --to markdown` render. Never hand-edit `README.md`; regenerate it (`check-readme` enforces a byte-exact match).
- **URLs absolute.** The console can't resolve relative paths. `check-image-urls` enforces this for image refs; apply the same rule to prose/markdown links.
- **URLs alive.** Every link in `README.ipynb` / `README.md` must resolve. Manual check (not in precommit) — especially important on updates, where links rot.
- **Ray-doc links → canonical URL.** For templates that also ship in the Ray repo/docs, link Ray documentation as `https://docs.ray.io/en/latest/...` — never a relative path or a `github.com/ray-project/ray/...` blob link. Exception: link GitHub directly only when the point is to show the source code itself.
- **Ray Train templates use the V2 API, never V1.**

⚠️ **Compute configs use the OLD (legacy) API**, NOT the new ComputeConfig API. ALWAYS mirror an existing `configs/<name>/` entry. Do NOT consult the live anyscale docs — they document only the new schema. Legacy references:
- ComputeTemplateConfig: https://docs.anyscale.com/ref/0.26.64/compute-config-api#computetemplateconfig-legacy
- ComputeNodeType:       https://docs.anyscale.com/ref/0.26.64/compute-config-api#computenodetype-legacy
- WorkerNodeType:        https://docs.anyscale.com/ref/0.26.64/other#workernodetype-legacy
- Resources:             https://docs.anyscale.com/ref/0.26.64/other#resources-legacy

## Validate locally

Run before pushing (CI runs the same hooks). Order matches `.pre-commit-config.yaml` — cheap notebook-hygiene first so the slow nbconvert-based `check-readme` runs last. Per hook: the scoped `pre-commit` call, and the direct script call for iterating one check in isolation.

| Hook | Checks | `pre-commit run …` | Direct |
|---|---|---|---|
| `clear-notebook-outputs` | strips outputs + `execution_count` from `*.ipynb` (**mutating** — re-stage after) | `clear-notebook-outputs --files <paths>` | `python3 ci/clear-notebook-outputs.py templates/<name>/*.ipynb` |
| `check-image-urls` | image refs in `*.ipynb`/`*.md` are absolute URLs | `check-image-urls --files <paths>` | `python3 ci/check-image-urls.py templates/<name>/*.ipynb templates/<name>/*.md` |
| `check-readme` | `README.md` == nbconvert of `README.ipynb` | `check-readme --files <paths>` | `bash ci/check-readme.sh templates/<name>/README.ipynb` |
| `check-build-yaml` | BUILD.yaml schema + referenced paths exist | `check-build-yaml --files BUILD.yaml` | `python3 ci/validate_build_yaml.py --no-network` |

Run everything at once: `pre-commit run --all-files`. Regenerate a stale `README.md`: `jupyter nbconvert --to markdown templates/<name>/README.ipynb --output-dir templates/<name>/`.
