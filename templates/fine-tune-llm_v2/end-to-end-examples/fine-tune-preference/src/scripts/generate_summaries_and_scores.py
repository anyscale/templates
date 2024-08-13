"""
Summary generation with support for offline and online inference.

Offline batched inference is implemented with Ray Data and vLLM while Online inference expects an OpenAI-compatible server
"""

import argparse
import os
import re
from enum import Enum
from typing import Optional, Union

import numpy as np
import ray
import yaml
from pydantic import Field, model_validator
from transformers import AutoTokenizer

from src.utils.models import OfflineInferenceConfig, OnlineInferenceConfig, BaseModelExtended, DataSchema
from src.utils.prompt_templates import (
    PROMPT_TEMPLATE_MCQ_ANSWERING,
    PROMPT_TEMPLATE_SUMMARY,
)
from src.utils.synthetic_data_utils import (
    InferenceType,
    duplicate_rows,
    extract_answers,
    format_into_prompt,
    format_into_prompt_rawtext,
    get_predictions_on_dataset,
)
from src.utils.common import init_logger

logger = init_logger()


class Mode(Enum):
    TRAIN = "train"
    EVAL = "eval"


class SummaryGenerationConfig(BaseModelExtended):
    mode: Mode = Field(description="Evaluation mode")
    inference_type: InferenceType = Field(
        default=InferenceType.OFFLINE,
        description="Inference type. Can be online (through an OpenAI-compatible server) or Offline (Batched inference with Ray + vLLM)",
    )
    input_folder: str = Field(description="Input Folder")
    model_inference_config: Union[OnlineInferenceConfig, OfflineInferenceConfig] = (
        Field(description="inference config for the model being evaluated by the judge")
    )
    num_generations: int = Field(
        default=1,
        description="Number of generations to sample from the model being evaluated by the judge",
    )
    judge_inference_config: OfflineInferenceConfig = Field(
        description="Batched inference config for the judge model"
    )
    num_mcq_questions: int = Field(
        default=5,
        description="Number of MCQ questions in the provided input dataset. Note that only those input dataset samples with `num_mcq_questions` questions will be used in evaluation.",
    )

    @model_validator(mode="after")
    def validate_model_config_and_type(self):
        if self.inference_type == InferenceType.OFFLINE:
            assert isinstance(self.model_inference_config, OfflineInferenceConfig)
        else:
            assert isinstance(self.model_inference_config, OnlineInferenceConfig)
        return self


def get_output_folder_name(model_config) -> str:
    if isinstance(model_config, OfflineInferenceConfig):
        output_model_name = (
            model_config.adapter_id_or_path
            if model_config.adapter_id_or_path is not None
            else model_config.model_id_or_path
        )
    elif isinstance(model_config, OnlineInferenceConfig):
        output_model_name = model_config.model_id
    else:
        raise NotImplementedError(
            f"Model config type {type(model_config)} not supported"
        )

    output_model_name = output_model_name.replace("/", "_")
    user_name = re.sub(r"\s+", "__", os.environ.get("ANYSCALE_USERNAME", "user"))
    output_folder = f"{os.environ.get('ANYSCALE_ARTIFACT_STORAGE')}/{user_name}/preference_tuning_summarization_example/summary_{config.mode.value}_generation_{output_model_name}_temp_{model_config.temperature}_judge_{judge_config.model_id_or_path.replace('/', '_')}/"
    return output_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A simple script for summary generation and scoring with support for offline and online inference."
    )
    parser.add_argument(
        "config_path", help="Path to the config file for summary generation"
    )
    args = parser.parse_args()

    config = SummaryGenerationConfig.from_yaml(args.config_path)
    model_config = config.model_inference_config
    judge_config = config.judge_inference_config

    output_folder = get_output_folder_name(model_config)

    # Initialize Ray with a Runtime Environment.
    env_vars = (
        {}
        if config.inference_type == InferenceType.OFFLINE
        else {
            config.model_inference_config.api_key_env_var: os.environ[
                config.model_inference_config.api_key_env_var
            ]
        }
    )
    ray.init(
        runtime_env={
            "env_vars": {
                "HF_TOKEN": os.environ["HF_TOKEN"],
                "HF_HOME": "/mnt/local_storage/.cache/huggingface",
                **env_vars,
            },
        },
        logging_config=ray.LoggingConfig(log_level="INFO"),
    )

    logger.info(f"OUTPUT FOLDER: {output_folder}")

    ds = ray.data.read_parquet(config.input_folder, file_extensions=["parquet"])

    ds = ds.filter(
        lambda row: row[DataSchema.GROUND_TRUTH_MCQ_ANSWERS_FIELD] is not None
        and len(row[DataSchema.GROUND_TRUTH_MCQ_ANSWERS_FIELD]) == config.num_mcq_questions,
        num_cpus=0,
    )

    tokenizer = None
    if config.inference_type == InferenceType.OFFLINE:
        tokenizer_id_or_path = (
            model_config.tokenizer_id_or_path
            if model_config.tokenizer_id_or_path
            else model_config.model_id_or_path
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id_or_path)

    ds = ds.map(
        format_into_prompt,
        fn_kwargs=dict(
            template=PROMPT_TEMPLATE_SUMMARY,
            type=config.inference_type,
            tokenizer=tokenizer,
            col_name=DataSchema.SUMMARY_GENERATION_INPUT_FIELD,
        ),
        num_cpus=0,
    )

    if config.num_generations > 1:
        ds = ds.flat_map(
            duplicate_rows,
            fn_kwargs=dict(
                count=config.num_generations,
                id_col="response_num",
            ),
            num_cpus=0,
        )
    ds = get_predictions_on_dataset(ds, model_config, col_in=DataSchema.SUMMARY_GENERATION_INPUT_FIELD, col_out="summary_generation_raw_model_output")

    # Input pre-processing for the judge model
    tokenizer_id_or_path = (
        judge_config.tokenizer_id_or_path
        if judge_config.tokenizer_id_or_path
        else judge_config.model_id_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id_or_path)
    ds = ds.map(
        format_into_prompt_rawtext,
        fn_kwargs=dict(
            template=PROMPT_TEMPLATE_MCQ_ANSWERING,
            tokenizer=tokenizer,
            col_name=DataSchema.JUDGE_MCQ_INPUT_FIELD,
        ),
        num_cpus=0,
    )
    # Get scores
    ds = get_predictions_on_dataset(ds, judge_config, col_in=DataSchema.JUDGE_MCQ_INPUT_FIELD, col_out=DataSchema.JUDGE_MCQ_RAW_OUTPUT_FIELD)

    ds = ds.map(
        extract_answers,
        fn_kwargs=dict(
            col_in=DataSchema.JUDGE_MCQ_RAW_OUTPUT_FIELD,
            col_out=DataSchema.JUDGE_MCQ_ANSWERS_FIELD,
            num_questions=config.num_mcq_questions,
        ),
        num_cpus=0,
    )

    ds.write_parquet(output_folder)

    logger.info(f"Dataset saved at: {output_folder}")
