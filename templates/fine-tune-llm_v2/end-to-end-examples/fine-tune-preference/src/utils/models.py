from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataSchema:
    ARTICLE = "text"
    SUMMARY_GENERATION_RAW_OUTPUT = "summary_generation_raw_model_output"
    JUDGE_MCQ_ANSWERS = "judge_mc_answers"
    NUM_WORDS = "num_words"
    ACCURACY = "accuracy"
    QA_GENERATION_PROMPT = "qa_generation_prompt"
    QA_GENERATION_RAW_OUTPUT = "qa_generation_raw_model_output"
    MCQ_QUESTIONS = "qa_generation_questions"
    GROUND_TRUTH_MCQ_ANSWERS = "qa_generation_answers"
    JUDGE_MCQ_RAW_OUTPUT = "judge_mc_raw_model_output"
    JUDGE_MCQ_INPUT = "judge_mc_prompt"
    SUMMARY_GENERATION_INPUT = "summary_generation_prompt"

    @classmethod
    def get_all_items(cls):
        return {
            key: value
            for key, value in vars(cls).items()
            if not key.startswith("__") and not callable(getattr(cls, key))
        }

    @classmethod
    def get_all_values(cls):
        return [
            value
            for key, value in vars(cls).items()
            if not key.startswith("__") and not callable(getattr(cls, key))
        ]


class BaseModelExtended(BaseModel):
    # NOTE: We use attributes such as model_id, etc which start with model_, a protected namespace in pydantic
    # We override this here to suppress pydantic's warnings
    # We forbit extra entries in our models
    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class MapperScalingConfig(BaseModelExtended):
    concurrency: int = Field(
        description="Number of Ray workers to use concurrently for the map operation."
    )
    batch_size: Optional[int] = Field(
        default=None,
        description="Batch size per worker for the map operation. If `None`, an entrie block of data is used as a batch.",
    )
    accelerator_type: Optional[str] = Field(
        default=None,
        description="Accelerator type for running on GPUs. More details in https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html#accelerator-types ",
    )
    num_gpus_per_instance: Optional[int] = Field(
        default=None, description="Number of GPUs per instance"
    )


class OnlineInferenceConfig(BaseModelExtended):
    model_id: str = Field(
        default="gpt-4o", description="Model ID for the OpenAI-compatible endpoint"
    )
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base url for the OpenAI-compatible server",
    )
    api_key_env_var: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable to read the api key from",
    )
    temperature: float = Field(description="Temperature while sampling from the model")
    max_tokens: int = Field(default=4096, description="Max tokens for generation")
    concurrency: int = Field(
        description="Number of concurrent requests to send to the server."
    )


class OfflineInferenceConfig(BaseModelExtended):
    model_id_or_path: str = Field(
        description="Model ID or local path to model checkpoint for the base model weights"
    )
    tokenizer_id_or_path: Optional[str] = Field(
        default=None,
        description="model ID or local path for the tokenizer to use. This is optional and can be useful if the model to be evaluated does not have a chat template in its tokenizer config. For example, a merged model can sometimes not have a default chat template.",
    )
    scaling_config: MapperScalingConfig = Field(
        description="Scaling config for batched inference on the model. Internally, this is a Ray Data operation."
    )
    adapter_id_or_path: Optional[str] = Field(
        default=None, description="HuggingFace model ID or local path to LoRA weights"
    )
    temperature: float = Field(description="Temperature while sampling from the model")
    top_p: float = Field(
        default=1,
        description="`top_p` sampling parameter controlling diversity of tokens sampled",
    )
    max_tokens: int = Field(default=4096, description="Max tokens for generation")
