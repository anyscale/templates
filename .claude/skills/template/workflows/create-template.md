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

## 3. BUILD.yaml entry

Append a list item per `../schemas/build-yaml-schema.yaml`. Set the image for the chosen case: `cluster_env.image_uri` (anyscale base) or `cluster_env.byod.{docker_image,ray_version}` (custom or third-party). For custom GCP, publish the image first (`.claude/skills/template/scripts/push-custom-image-to-gcp.sh <dockerfile-dir> <name> <ray-version>`) and use the printed URI. The entry also wires `compute_config` (step 4) and the `test` block (`command: bash tests.sh`, `tests_path: tests/<name>/`, and `timeout_in_sec` set a bit above the test's measured runtime — target < 30 min).

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

Commit on a branch and open a PR against `main`. Run `/test-template`, get it green **before publishing**. Dispatch, monitoring, and failure recovery: `../references/testing-template.md`.

## 8. Merge to `main`

**Merge the green PR to `main` first** — the publish pipeline ships templates `main`, so nothing publishes from an unmerged branch.

## 9. Publish + register via the product gallery

Hand off to **`/register-template`** (the `console-template-plugin` in anyscale/product). For a **new** template it owns the whole publish: the `workspace-templates.yaml` gallery entry **and** the `tmpl-publish` run (dev → dev-console test → staging → prod), interleaved in that order — the dev-console test needs both the artifact and the gallery entry to exist. **Don't run `tmpl-publish` yourself** here, and do no other product-repo work.
