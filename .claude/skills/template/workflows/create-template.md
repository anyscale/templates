# Create a new template

Interactive flow. `../references/conventions.md` owns conventions/schemas; `../references/testing-template.md` owns the test canon — this file is the create sequence. Read `../references/conventions.md` before generating.

`<name>` is the template's identifier — reused in `BUILD.yaml`, `templates/<name>/`, `configs/<name>/`, `tests/<name>/`.

## 1. Intake — ask before generating anything

Interview the user; explain each piece when you reach it. Collect:

- **What should the template demonstrate?** — the user-facing story.
- **What do you have already?**
  - *Nothing / just an idea* → bootstrap with **anyscale/anyscale-template-agent** (https://github.com/anyscale/anyscale-template-agent), a Claude Code agentic system: feed it source material (notebook, `.py`, markdown, a URL, or a GitHub repo) or just a prompt, and it runs an Author→Debug→Diagram→Finalize pipeline that delivers an execution-verified template `.ipynb` (pinned deps + architecture diagram) in `final/`. Setup: its README. Drop the result into `templates/<name>/`, then integrate. (Can't use the agent? Author the notebook yourself, then continue from §2.)
  - *Rough / partial* → integrate as-is, or send to anyscale-template-agent to polish (diagrams, debugging, finalizing).
  - *Complete* → integrate.
- **Notebook or script?** — sets the test shape (`../references/testing-template.md`).
- **Run it on a workspace yet?** — if yes, get the Anyscale console workspace URL (best compute-config source, step 4). If no, you'll fill compute configs by guided Q&A.
- **Which image case?** (SKILL.md "Image URI cases") — deps via notebook `!pip` / `requirements.txt` on stock Ray → **anyscale base**; extra system libraries or heavy/conflicting deps → **custom GCP** (needs a Dockerfile); an upstream-maintained image → **third-party**.

## 2. Drop in the content

Move the content into `templates/<name>/`. For a notebook template, the main notebook **is** `README.ipynb` (the test runs it; `README.md` is its rendered copy — README convention in `../references/conventions.md`); scripts and Dockerfiles sit alongside it.

**Dependencies.** If the template pins deps via a `python_depset.lock` (most do — system in `../references/dependencies.md`), drop its `requirements.txt` into `templates/<name>/` and add a per-template `compile` entry to `dependencies/template.depsets.yaml`: `build_arg_sets` = the matching `ray<ver>_py<XX>` bundle, plus a `seed-image-freeze.sh` pre_hook naming the freeze for **the image you set in step 3** (if that image isn't in `dependencies/images/tracked-images.txt` yet, add it and run `refresh-image-freezes.sh <ray-version>`). List only what the template *adds* — the freeze already holds everything the image ships. Compile its lock — `./update_deps.sh --name <the entry's name>` — and confirm `./update_deps.sh --check` is clean. A pure workspace tutorial with only `!pip` installs needs no lock (workspace auto-propagation covers workers); a ship-path template (Service/Job) must deliver worker deps via a `.lock` + `runtime_env`, not bare `!pip` — see `../references/dependencies.md`. **Serve/LLM template, or adding a dep that could move the image's framework (`fastapi`/`starlette`/`pydantic`/`vllm`/`torch`)?** Read `../references/dependencies.md` "Runtime skew" first — the wrong delivery scope or a downgraded framework passes tests but dies on the real multi-node launch.

## 3. BUILD.yaml entry

Append a list item per `../schemas/build-yaml-schema.yaml`. Set `owner_team` (required — `ray-serve` | `ray-data` | `llm` | `ray-train` | `general`, deduced from the template's center of gravity; the schema file has the rule + tie-break). Set the image for the chosen case: `cluster_env.image_uri` (anyscale base) or `cluster_env.byod.{docker_image,ray_version}` (custom or third-party). For custom GCP, publish the image first (`.claude/skills/template/scripts/push-custom-image-to-gcp.sh <dockerfile-dir> <name> <ray-version>`) and use the printed URI. The entry also wires `compute_config` (step 4) and the `test` block (`command: bash tests.sh`, `tests_path: tests/<name>/`, and `timeout_in_sec` set a bit above the test's measured runtime — target < 30 min).

## 4. Compute configs

**Preferred — translate from the tested workspace.** Extract the workspace ID (`expwrk_*` from `/workspaces/<id>` in the URL) and fetch its config:

```
anyscale workspace_v2 get --id expwrk_<id> --json | jq '.config.compute_config'
```

That returns the ComputeConfig shape `configs/` uses directly (full fields + patterns in `../schemas/compute-config-schema.yaml`). Copy it, pruned to the minimal form:

- drop `cloud` / `cloud_resource` (injected at clone time)
- drop fields matching their defaults: `min_nodes: 0`, `market_type: ON_DEMAND`, `auto_select_worker_config: false`, `enable_cross_zone_scaling: false`
- drop auto-detected node `resources` (with workers present, the head is unschedulable by default); keep explicit overrides like `CPU: 0`
- keep `max_nodes` explicit on every worker group

Write `configs/<name>/aws.yaml` and `gce.yaml` by instance family.

**Fallback — guided Q&A.** No tested workspace → walk the user through those same fields.

## 5. Test

Write `tests/<name>/tests.sh` per `../references/testing-template.md` — shape (papermill notebook vs custom script) follows the intake answer.

## 6. Format

Apply `../references/conventions.md` to the new template.

## 7. Test gate — non-skippable

Commit on a branch and open a PR against `main`. Run `/test-template`, get it green **before publishing**. Dispatch, monitoring, and failure recovery: `../references/testing-template.md`. (Exception — an event template under time pressure can publish *test-free* by routing through `archive/`: see "Publish without the test gate" in `../references/publish-to-backend.md`.)

## 8. Merge to `main`

**Merge the green PR to `main` first** — the publish pipeline ships templates `main`, so nothing publishes from an unmerged branch.

## 9. Publish + register via the product gallery

Hand off to **`/register-template`** (the `console-template-plugin` in anyscale/product). For a **new** template it owns the whole publish: the `workspace-templates.yaml` gallery entry **and** the `tmpl-publish` run (dev → dev-console test → staging → prod), interleaved in that order — the dev-console test needs both the artifact and the gallery entry to exist. **Don't run `tmpl-publish` yourself** here, and do no other product-repo work.

## 10. Optional — surface it as a Ray docs example

IMPORTANT ! Only do this for a template meant to be a Ray docs example — most templates are console-gallery only. If you're not sure, ASK the user to confirm they have the green light from Ray docs team.

Step 9 puts the template in the Anyscale console gallery. It does **not** put it on [docs.ray.io](https://docs.ray.io) — that's a separate registration in `ray-project/ray`. The Ray docs build fetches published template builds from `templates.ci.ray.io` and renders each one's `README.md` under `_collections/`, so a template must be published (step 9) before Ray can pull it. Three Ray-side edits: a `_TEMPLATE_COLLECTIONS` entry (`doc/source/template_collections.py`), a build pin (`doc/source/template_pins.json`), and an `examples.yml` entry for the owning library's gallery.

Procedure: **[Publishing an example](https://docs.ray.io/en/master/ray-contribute/publishing-examples.html)** in the Ray contributor docs. (Linked on `/en/master/` because the page is newer than the current Ray release; it reaches `/en/latest/` with the next one.)
