#!/usr/bin/env bash
set -euxo pipefail

# Driver-side install only: `--system` bypasses workspace propagation and reaches this
# head node alone, which is all papermill and the asserts below need. The agent/MCP
# replicas get the same lock via `pip:` on each app in serve_multi_config.yaml — the
# head is unschedulable here, so they land on the worker with only the base image.
uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
uv pip install -q --system papermill "nbconvert==7.16.6" ipykernel

set +x  # don't echo the resolved secret under xtrace
BRAVE_API_KEY=$(aws secretsmanager get-secret-value \
  --secret-id brave-search-api-key \
  --query SecretString \
  --output text)
export BRAVE_API_KEY
set -x

# Two configs derived from the shipped serve_multi_config.yaml, so apps keep their
# runtime_env verbatim:
#   * .ci.yaml    — all apps, plus BRAVE_API_KEY injected into mcp_web_search's
#                   runtime_env (`serve run` doesn't forward the runner's shell env;
#                   customers set it via the Dependencies tab).
#   * .infra.yaml — llm + the two MCP servers only.
# Agents deploy in a second pass against an already-warm LLM: build_agent() resolves
# MCP/LLM during __init__ and its retries expire before the L4 cold start finishes,
# marking the agent apps DEPLOY_FAILED (terminal).
# TODO: template-side fix — retry MCP/LLM discovery with backoff, or defer it past
# __init__, so one `serve run` suffices and the two passes collapse back into one.
python - <<'PY'
import os
import yaml

with open("serve_multi_config.yaml") as f:
    cfg = yaml.safe_load(f)

for app in cfg["applications"]:
    if app["name"] == "mcp_web_search":
        app.setdefault("runtime_env", {}).setdefault("env_vars", {})[
            "BRAVE_API_KEY"
        ] = os.environ["BRAVE_API_KEY"]

with open("/tmp/serve_multi_config.ci.yaml", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

infra = dict(cfg)
infra["applications"] = [
    a for a in cfg["applications"] if a["name"] in {"llm", "mcp_web_search", "mcp_weather"}
]
with open("/tmp/serve_multi_config.infra.yaml", "w") as f:
    yaml.safe_dump(infra, f, sort_keys=False)
PY

serve run /tmp/serve_multi_config.infra.yaml --non-blocking
trap 'serve shutdown -y || true' EXIT

# Block until the LLM app finishes loading on the L4 worker (the slowest deployment).
# L4 provisioning + vLLM cold-start can exceed 10 min, so wait ~20 and fail loudly with
# serve status + routes. Without this gate the loop falls through silently, the
# notebook/smoke test below hit the proxy's 404 fallback, and
# the run dies with an opaque JSONDecodeError instead of "LLM never became ready".
llm_ready=0
for _ in $(seq 1 240); do
  if curl -sf http://127.0.0.1:8000/llm/v1/models >/dev/null 2>&1; then
    llm_ready=1
    break
  fi
  sleep 5
done
if [ "$llm_ready" -ne 1 ]; then
  echo "ERROR: LLM app did not become ready within ~20 min." >&2
  echo "----- serve status -----" >&2
  serve status >&2 || true
  echo "----- /-/routes -----" >&2
  curl -s http://127.0.0.1:8000/-/routes >&2 || true
  exit 1
fi

# Second pass: adds the six agent apps. llm/mcp_* are byte-identical to the infra pass,
# so Serve leaves them running and only builds the agents.
serve run /tmp/serve_multi_config.ci.yaml --non-blocking

# Wait for every app to reach RUNNING, then exercise an agent end to end. This is the
# assertion that the scoped runtime_env actually worked: an agent replica without the
# locked deps can't import langchain/langgraph/mcp/a2a-sdk and the app goes
# DEPLOY_FAILED instead.
python - <<'PY'
import subprocess
import sys
import time

import requests

# Same endpoint `serve status` reads, but it hands back JSON instead of YAML holding
# enum objects.
SERVE_API = "http://localhost:8265/api/serve/applications/"


def app_statuses():
    resp = requests.get(SERVE_API, timeout=30)
    resp.raise_for_status()
    apps = resp.json().get("applications") or {}
    return {name: info.get("status") for name, info in apps.items()}


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    print("----- serve status -----", file=sys.stderr)
    subprocess.run(["serve", "status"], stderr=sys.stderr)
    sys.exit(1)


deadline = time.time() + 900
while True:
    statuses = app_statuses()
    failed = {n: s for n, s in statuses.items() if s in ("DEPLOY_FAILED", "UNHEALTHY")}
    if failed:
        die(f"apps failed to deploy: {failed}")
    if statuses and all(s == "RUNNING" for s in statuses.values()):
        print(f"OK: all {len(statuses)} apps RUNNING")
        break
    if time.time() > deadline:
        die(f"apps did not all reach RUNNING within 15 min: {statuses}")
    time.sleep(10)

# The agent streams LangChain updates as SSE; an in-band `event: error` frame means the
# replica came up but the agent itself failed (bad MCP/LLM wiring, missing tool deps).
resp = requests.post(
    "http://127.0.0.1:8000/weather-agent/chat",
    json={"user_request": "What is the weather in San Francisco?"},
    timeout=300,
)
if resp.status_code != 200:
    die(
        f"POST /weather-agent/chat returned HTTP {resp.status_code}.\n"
        f"Body (first 500 chars): {resp.text[:500]!r}"
    )
body = resp.text
if "event: error" in body:
    die(f"weather agent emitted an error frame: {body[:1000]!r}")
if "data:" not in body:
    die(f"weather agent streamed no update frames: {body[:1000]!r}")
print("OK: weather agent answered over /weather-agent/chat")
PY

jupyter nbconvert --to notebook README.ipynb \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
  --output /tmp/multi_agent_a2a.ci.ipynb
papermill /tmp/multi_agent_a2a.ci.ipynb /tmp/multi_agent_a2a.out.ipynb --log-output --kernel python3 --cwd .

# Smoke: LLM serves real completions.
python - <<'PY'
import sys
import requests
BASE = "http://127.0.0.1:8000/llm"


def json_or_die(resp, what):
    """Surface route-not-found / non-JSON bodies clearly instead of JSONDecodeError."""
    ctype = resp.headers.get("content-type", "")
    if resp.status_code != 200 or "application/json" not in ctype:
        sys.exit(
            f"ERROR: {what} returned HTTP {resp.status_code} ({ctype or 'no content-type'}).\n"
            f"Body (first 500 chars): {resp.text[:500]!r}\n"
            f"The Serve route is likely not up — check http://127.0.0.1:8000/-/routes."
        )
    return resp.json()


models = json_or_die(requests.get(f"{BASE}/v1/models", timeout=10), "GET /v1/models")["data"]
assert models, "no models served"
model_id = models[0]["id"]
r = requests.post(f"{BASE}/v1/chat/completions",
    headers={"Authorization": "Bearer local"},
    json={"model": model_id, "messages": [{"role": "user", "content": "Say hi."}]},
    timeout=30)
content = json_or_die(r, "POST /v1/chat/completions")["choices"][0]["message"]["content"].strip()
assert content, "LLM returned empty content"
print(f"OK: LLM chat completion via {model_id}")
PY

# Reached only on success; the test pipeline fails a "pass" that lacks it.
echo "RAYAPP_TESTS_COMPLETE"
