#!/usr/bin/env bash
set -euxo pipefail

pip install -q -r requirements.txt

# Local end-to-end: deploy the LLM + weather MCP + LangGraph agent as 3 Serve apps and query the agent.
serve run --non-blocking --name llm --route-prefix /llm llm_deploy_qwen:app
trap 'serve shutdown -y >/dev/null 2>&1 || true' EXIT
serve run --non-blocking --name weather --route-prefix /weather weather_mcp_ray:app

# Wait for the Qwen LLM to load (GPU autoscale + vLLM); /v1/models 200 means query-ready.
for _ in $(seq 1 150); do
  if curl -sf http://localhost:8000/llm/v1/models >/dev/null 2>&1; then break; fi
  sleep 10
done
curl -sf http://localhost:8000/llm/v1/models >/dev/null

# Agent deploys last (its startup connects to the MCP); point it at the local LLM + MCP
# via the env vars the template already reads (no code edits).
serve run --non-blocking --name agent --route-prefix / \
  --runtime-env-json '{"env_vars":{"OPENAI_COMPAT_BASE_URL":"http://localhost:8000/llm/","WEATHER_MCP_BASE_URL":"http://localhost:8000/weather/","OPENAI_API_KEY":"local","WEATHER_MCP_TOKEN":"local"}}' \
  ray_serve_agent_deployment:app

# Agent ready when its FastAPI route is mounted (GET /chat -> 405 Method Not Allowed).
for _ in $(seq 1 60); do
  code="$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/chat || true)"
  if [ "$code" = "405" ]; then break; fi
  sleep 5
done

python query_agent.py
