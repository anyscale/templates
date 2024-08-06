import argparse
import logging
import os

import datasets
import ray
from transformers import AutoTokenizer

from utils.synthetic_data_utils import (LLMPredictor,
                                        format_into_prompt_rawtext, shuffle_qa)
from utils.utils import init_logger

from utils.prompt_templates import PROMPT_TEMPLATE_QUESTION_GENERATION

logger = init_logger()

OUTPUT_PATH = f"{os.environ.get('ANYSCALE_ARTIFACT_STORAGE')}/preference_tuning_summarization_example/qa_annotations"
OUTPUT_PATH_TRAIN = f"{OUTPUT_PATH}_train"
OUTPUT_PATH_TEST = f"{OUTPUT_PATH}_test"
# New output columns
OUTPUT_QUESTION_FIELD = "qa_generation_questions"
OUTPUT_ANSWER_FIELD = "qa_generation_answers"
OUTPUT_RAW_GENERATION_FIELD = "qa_generation_raw_model_output"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-id",
    type=str,
    default="meta-llama/Meta-Llama-3-70B-Instruct",
    help="Model to use for generation",
)
parser.add_argument(
    "--concurrency",
    type=int,
    required=True,
    help="Number of LLM instances to use concurrently",
)
parser.add_argument(
    "--num-gpus-per-instance",
    type=int,
    required=True,
    help="Number of GPUs to use per instance. This also sets the tensor parallelism size with vLLM",
)
parser.add_argument(
    "--gpu-type",
    type=str,
    default="H100",
    help="Accelerator type to use. Recommended to be H100 or A100",
)
parser.add_argument(
    "--batch-size", type=int, required=True, help="Batch size for each instance"
)
parser.add_argument(
    "--num-data-blocks-per-device",
    type=int,
    default=1,
    help="Number of Ray data blocks per GPU device. If unsure, use the default value",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Initialize Ray with a Runtime Environment.
    ray.init(
        runtime_env={
            "env_vars": {
                "HF_TOKEN": os.environ["HF_TOKEN"],
                "HF_HOME": "/mnt/local_storage/.cache/huggingface",
            },
        },
        logging_config=ray.LoggingConfig(encoding="JSON", log_level="INFO"),
    )

    hf_ds = datasets.load_dataset("abisee/cnn_dailymail", "3.0.0", split="train").shuffle(
        seed=21
    )
    hf_ds = hf_ds.rename_columns({"article": "text"})
    # the resulting keys for the dataset are "article" (which contains the text) and "id" only
    hf_ds = hf_ds.remove_columns(["highlights"])

    ds = ray.data.from_huggingface(hf_ds)
    # By default, a HF dataset is converted to a Materialized dataset and the number of blocks can be low
    num_blocks = (
        args.concurrency * args.num_gpus_per_instance * args.num_data_blocks_per_device
    )
    ds = ds.repartition(num_blocks)

    prompt_field = "qa_generation_prompt"
    ds = ds.map(
        format_into_prompt_rawtext,
        fn_kwargs=dict(
            template=PROMPT_TEMPLATE_QUESTION_GENERATION,
            tokenizer=AutoTokenizer.from_pretrained(args.model_id),
            col_name=prompt_field,
        ),
        num_cpus=0,
    )
    ds = ds.map_batches(
        LLMPredictor,
        fn_constructor_kwargs=dict(
            model_location=args.model_id,
            col_in=prompt_field,
            col_out=OUTPUT_RAW_GENERATION_FIELD,
            temperature=0,
            max_tokens=4096,
            vllm_settings=dict(
                tensor_parallel_size=args.num_gpus_per_instance,
                max_model_len=8192,
                dtype="bfloat16",
            ),
        ),
        concurrency=args.concurrency,
        num_gpus=args.num_gpus_per_instance,
        accelerator_type=args.gpu_type,
        batch_size=args.batch_size,
    )

    ds = ds.flat_map(
        shuffle_qa,
        fn_kwargs=dict(
            col_in=OUTPUT_RAW_GENERATION_FIELD,
            col_out_prompt=OUTPUT_QUESTION_FIELD,
            col_out_answers=OUTPUT_ANSWER_FIELD,
        ),
        num_cpus=0,
    )

    train_ds, test_ds = ds.train_test_split(test_size=0.05)

    train_ds.write_parquet(OUTPUT_PATH_TRAIN)
    test_ds.write_parquet(OUTPUT_PATH_TEST)
