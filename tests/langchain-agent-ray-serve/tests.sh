#!/usr/bin/env bash
set -euxo pipefail

uv pip install -r python_depset.lock --system --no-deps --no-cache-dir --index-strategy unsafe-best-match

# README.ipynb deploys via `anyscale service deploy` (prod); run the same 3 apps locally instead
# — LLM + weather MCP + LangGraph agent — then query the agent.
trap 'serve shutdown -y >/dev/null 2>&1 || true' EXIT
serve run --non-blocking --name llm --route-prefix /llm llm_deploy_qwen:app
# runtime_env pip: the `uv pip install --system` above reaches the head only
# (it bypasses Anyscale's worker propagation), and under the published (probe)
# compute config the head is non-schedulable — replicas land on workers that
# lack the locked deps (ModuleNotFoundError: mcp.server.fastmcp). Point the
# weather/agent apps' runtime_env at python_depset.lock so replicas install
# the same pinned closure wherever they land (same pattern as #783). The llm
# app needs nothing beyond the ray-llm image.
LOCK_RUNTIME_ENV="{\"pip\": \"$(pwd)/python_depset.lock\"}"
serve run --non-blocking --name weather --route-prefix /weather \
  --runtime-env-json "$LOCK_RUNTIME_ENV" \
  weather_mcp_ray:app

# Wait for the Qwen LLM to load (GPU autoscale + vLLM); /v1/models 200 means query-ready.
for _ in $(seq 1 150); do
  if curl -sf http://localhost:8000/llm/v1/models >/dev/null 2>&1; then break; fi
  sleep 10
done
curl -sf http://localhost:8000/llm/v1/models >/dev/null

# Agent deploys last (startup connects to the MCP); point it at the local LLM+MCP via the env vars it already reads.
serve run --non-blocking --name agent --route-prefix / \
  --runtime-env-json "{\"env_vars\":{\"OPENAI_COMPAT_BASE_URL\":\"http://localhost:8000/llm/\",\"WEATHER_MCP_BASE_URL\":\"http://localhost:8000/weather/\",\"OPENAI_API_KEY\":\"local\",\"WEATHER_MCP_TOKEN\":\"local\"},\"pip\":\"$(pwd)/python_depset.lock\"}" \
  ray_serve_agent_deployment:app

# Agent ready when its FastAPI route is mounted (GET /chat -> 405 Method Not Allowed).
for _ in $(seq 1 60); do
  code="$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/chat || true)"
  if [ "$code" = "405" ]; then break; fi
  sleep 5
done

python query_agent.py

# Reached only on success; the test pipeline fails a "pass" that lacks it.
echo "RAYAPP_TESTS_COMPLETE"
