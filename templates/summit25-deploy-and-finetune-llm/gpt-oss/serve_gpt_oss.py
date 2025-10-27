# serve_gpt_oss.py
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="my-gpt-oss",
        # If issues downloading from hugging face, use model_source="s3://llm-guide/data/ray-serve-llm/hf_repo/gpt-oss-20b"
        model_source="openai/gpt-oss-20b",
    ),
    accelerator_type="L4",
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1,
            max_replicas=2,
        )
    ),
    engine_kwargs=dict(
        max_model_len=32768
    ),
    log_engine_metrics= True,
)

app = build_openai_app({"llm_configs": [llm_config]})
