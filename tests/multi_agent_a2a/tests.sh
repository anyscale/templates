#!/usr/bin/env bash
set -euxo pipefail

uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match
uv pip install -q --system papermill "nbconvert==7.16.6" ipykernel

set +x  # don't echo the resolved secret under xtrace
BRAVE_API_KEY=$(aws secretsmanager get-secret-value \
  --secret-id brave-search-api-key \
  --query SecretString \
  --output text)
export BRAVE_API_KEY
set -x

# `serve run` doesn't propagate the runner's shell env to Ray Serve replicas (customers
# normally set BRAVE_API_KEY via the workspace Dependencies tab). Inject it into the
# mcp_web_search app's runtime_env via a temp YAML so the Brave tool can authenticate.
python - > /tmp/serve_multi_config.ci.yaml <<'PY'
import os, yaml
with open('serve_multi_config.yaml') as f:
    cfg = yaml.safe_load(f)
for app in cfg['applications']:
    if app['name'] == 'mcp_web_search':
        app.setdefault('runtime_env', {}).setdefault('env_vars', {})['BRAVE_API_KEY'] = os.environ['BRAVE_API_KEY']
print(yaml.safe_dump(cfg, sort_keys=False))
PY

serve run /tmp/serve_multi_config.ci.yaml --non-blocking
trap 'serve shutdown -y || true' EXIT

# Block until the LLM app finishes loading on the L4 worker (the slowest deployment).
# L4 provisioning + vLLM cold-start can exceed 10 min, so wait up to ~20 min and fail
# loudly (dumping serve status + routes) if it never comes up. Without this gate the loop
# silently falls through, the notebook/smoke test below hit the proxy's 404 fallback, and
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

jupyter nbconvert --to notebook README.ipynb \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
  --output /tmp/multi_agent_a2a.ci.ipynb
papermill /tmp/multi_agent_a2a.ci.ipynb /tmp/multi_agent_a2a.out.ipynb --log-output --kernel python3 --cwd .

# Smoke: LLM serves real completions. The full `python tests/run_all.py` is intentionally
# excluded — the 3 agent deployments race the LLM cold-start (~2:40 min) during a single
# `serve run serve_multi_config.yaml`: agents call MCP/LLM during build_agent(), retries
# exhaust before /llm/v1/models is up, Ray Serve marks them DEPLOY_FAILED (terminal).
# TODO(@kunling, @aydin): template-side fix — make build_agent() retry MCP/LLM
# discovery with backoff (or defer it past __init__) so cold deps don't fail-fast.
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
