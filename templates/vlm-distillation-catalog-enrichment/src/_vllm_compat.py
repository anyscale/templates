"""Compat shim for vLLM ↔ Ray LLM batch stage.

vLLM 0.20+ moved ``TokensPrompt`` and ``TextPrompt`` up from
``vllm.inputs.data`` to ``vllm.inputs`` directly. Ray 2.55's
``vllm_engine_stage.py`` still references the old path, so we alias
``vllm.inputs.data`` back to ``vllm.inputs`` if missing.

Wired in via ``vLLMEngineProcessorConfig.runtime_env={"worker_process_setup_hook": "src._vllm_compat.patch"}``
so it runs once per LLM worker on startup, before the stage UDF runs.
"""


def patch() -> None:
    import vllm.inputs

    if not hasattr(vllm.inputs, "data"):
        vllm.inputs.data = vllm.inputs
