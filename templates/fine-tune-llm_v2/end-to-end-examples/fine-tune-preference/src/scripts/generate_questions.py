"""
Generates questions and answers to each article in the dataset using an LLM
"""

import argparse
import os
import re

import datasets
import ray
from pydantic import Field
from transformers import AutoTokenizer

from src.utils.common import init_logger, MODEL_HOME
from src.utils.models import (
    BaseModelExtended, DataSchema, OfflineInferenceConfig
)
from src.utils.prompt_templates import PROMPT_TEMPLATE_QUESTION_GENERATION
from src.utils.synthetic_data_utils import (
    format_into_prompt_rawtext,
    shuffle_qa,
)
from src.utils.predictors import get_predictions_on_dataset

logger = init_logger()

parser = argparse.ArgumentParser()


class QuestionGenerationConfig(BaseModelExtended):
    model_inference_config: OfflineInferenceConfig = Field(
        description="Inference config for the model"
    )
    num_samples_total: int = Field(description="Number of articles to sample in total")
    output_folder: str = Field(
        description="Output folder in artifact storage to store the results in"
    )
    num_data_blocks_per_device: int = Field(
        default=1,
        description="Number of Ray data blocks per GPU device. If unsure, use the default value",
    )
    train_test_split: float = Field(
        default=0.01, description="Percentage of articles to use for the test set"
    )


def get_full_output_folder_path(output_folder: str) -> str:
    user_name = re.sub(r"\s+", "__", os.environ.get("ANYSCALE_USERNAME", "user"))
    output_folder = (
        f"{os.environ.get('ANYSCALE_ARTIFACT_STORAGE')}/{user_name}/{output_folder}"
    )
    return output_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A simple script for generating questions and answers for articles in the CNN dataset"
    )
    parser.add_argument(
        "config_path", help="Path to the config file for summary generation"
    )
    args = parser.parse_args()

    # Initialize Ray with a Runtime Environment.
    ray.init(
        runtime_env={
            "env_vars": {
                "HF_HOME": MODEL_HOME,
            },
        },
        logging_config=ray.LoggingConfig(log_level="INFO"),
    )

    config = QuestionGenerationConfig.from_yaml(args.config_path)
    output_folder = get_full_output_folder_path(config.output_folder)
    logger.info(f"Output folder: {output_folder}")

    hf_ds = datasets.load_dataset(
        "abisee/cnn_dailymail", "3.0.0", split="train"
    ).shuffle(seed=21)
    hf_ds = hf_ds.rename_columns({"article": "text"})
    # the resulting keys for the dataset are "article" (which contains the text) and "id" only
    hf_ds = hf_ds.remove_columns(["highlights"])
    hf_ds = hf_ds.select(range(config.num_samples_total))

    ds = ray.data.from_huggingface(hf_ds)

    model_config: OfflineInferenceConfig = config.model_inference_config
    scaling_config = model_config.scaling_config
    # By default, a HF dataset is converted to a Materialized dataset and the number of blocks can be low
    num_blocks = (
        scaling_config.concurrency
        * scaling_config.num_gpus_per_instance
        * config.num_data_blocks_per_device
    )
    ds = ds.repartition(num_blocks)

    ds = ds.map(
        format_into_prompt_rawtext,
        fn_kwargs=dict(
            template=PROMPT_TEMPLATE_QUESTION_GENERATION,
            tokenizer=AutoTokenizer.from_pretrained(model_config.model_id_or_path),
            col_name=DataSchema.QA_GENERATION_PROMPT,
        ),
        num_cpus=0,
    )
    ds = get_predictions_on_dataset(
        ds,
        model_config,
        col_in=DataSchema.QA_GENERATION_PROMPT,
        col_out=DataSchema.QA_GENERATION_RAW_OUTPUT,
    )

    ds = ds.flat_map(
        shuffle_qa,
        fn_kwargs=dict(
            col_in=DataSchema.QA_GENERATION_RAW_OUTPUT,
            col_out_prompt=DataSchema.MCQ_QUESTIONS,
            col_out_answers=DataSchema.GROUND_TRUTH_MCQ_ANSWERS,
        ),
        num_cpus=0,
    )

    train_ds, test_ds = ds.train_test_split(test_size=config.train_test_split)
    train_split_path = os.path.join(output_folder, "train")
    test_split_path = os.path.join(output_folder, "test")
    train_ds.write_parquet(train_split_path)
    test_ds.write_parquet(test_split_path)
    logger.info(f"Train split have been saved to: {train_split_path}")
    logger.info(f"Test split have been saved to: {test_split_path}")
