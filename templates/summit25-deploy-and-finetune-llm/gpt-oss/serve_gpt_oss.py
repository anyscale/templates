# serve_gpt_oss.py
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config=dict(
        model_id="my-gpt-oss",
        model_source="s3://llm-guide/data/ray-serve-llm/hf_repo/gpt-oss-20b", # also support huggingface repo syntax like openai/gpt-oss-20b
    ),
    accelerator_type="L4",
    deployment_config=dict(
        autoscaling_config=dict(
            min_replicas=1, # avoid cold starts by keeping at least 1 replica always on
            max_replicas=2, # limit max replicas to control cost
        )
    ),
    engine_kwargs=dict(
        max_model_len=32768
    ),
    log_engine_metrics= True,
)

app = build_openai_app({"llm_configs": [llm_config]})
