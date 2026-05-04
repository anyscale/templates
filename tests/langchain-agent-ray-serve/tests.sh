#!/usr/bin/env bash
set -euo pipefail

echo "=== langchain-agent-ray-serve template tests ==="

echo "[1/4] Installing dependencies..."
pip install -r requirements.txt

echo "[2/4] Validating llm_deploy_qwen.py (Ray Serve LLM app construction)..."
python -c "
from llm_deploy_qwen import app, llm_config
print(f'  LLM config model: {llm_config.model_loading_config[\"model_id\"]}')
print(f'  Accelerator: {llm_config.accelerator_type}')
print('  LLM app constructed successfully')
"

echo "[3/4] Validating weather_mcp_ray.py (MCP service construction)..."
python -c "
from weather_mcp_ray import app, mcp
print(f'  MCP server name: {mcp.name}')
print('  MCP app constructed successfully')
"

echo "[4/4] Validating ray_serve_agent_deployment.py (imports and class definition)..."
python -c "
from ray_serve_agent_deployment import LangGraphServeDeployment, app
print('  Agent deployment class defined successfully')
print('  Agent app bound successfully')
"

echo ""
echo "=== All tests passed ==="
