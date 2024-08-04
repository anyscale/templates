import re
from tqdm import tqdm
import ray
import glob
import requests
import random
import os

from transformers import AutoTokenizer
from synthetic_data_utils import LLMPredictor, format_into_prompt_rawtext, shuffle_qa
import datasets

from utils import get_a10g_or_equivalent_accelerator_type, prompt_for_hugging_face_token

PROMPT_TEMPLATE_QUESTION_GENERATION = """Given the following text, generate five multiple choice questions with the following format. The questions must be simple enough to be answerable with only general important details that would be included in a short two sentence summary of the text. The questions must be only answerable when given the text and should not be answerable with common knowledge. Do not write questions about minute details in the text, only the most important points.

Format:
Q1) Question
A. Choice 1
B. Choice 2
C. Choice 3
D. Choice 4
E. Choice 5

Q1 Answer: A/B/C/D/E

Q2) Question
A. Choice 1
B. Choice 2
C. Choice 3
D. Choice 4
E. Choice 5

Q2 Answer: A/B/C/D/E

etc...

Text:
{article}"""

OUTPUT_PATH = f"{os.environ.get('ANYSCALE_ARTIFACT_STORAGE')}/preference_tuning_summarization_example/qa_annotations"
OUTPUT_PATH_TRAIN = OUTPUT_PATH + "_train"
OUTPUT_PATH_TEST = OUTPUT_PATH + "_test"

HF_MODEL = "meta-llama/Meta-Llama-3-70B"
# The number of LLM instances to use.
NUM_LLM_INSTANCES = 2
# The number of GPUs to use per LLM instance. NUM_GPUS_PER_INSTANCE > 1 will use tensor parallelism across all GPUs.
NUM_GPUS_PER_INSTANCE = 8
# The type of GPU to use
GPU_TYPE = get_a10g_or_equivalent_accelerator_type()
# Batch size for each instance
BATCH_SIZE = 1

# Initialize Ray with a Runtime Environment.
ray.init(
    runtime_env={
        "env_vars": {"HF_TOKEN": os.environ["HF_TOKEN"], "HF_HOME": "/mnt/local_storage/.cache/huggingface"},
    }
)

hf_ds = datasets.load_dataset("abisee/cnn_dailymail", '3.0.0', split="train").shuffle(seed=21)
# extract a subset of 21000 articles, (20000 for train, 1000 for testing)
hf_ds_subset = hf_ds.select(range(100))
# the resulting keys for the dataset are "article" (which contains the text) and "id" only
hf_ds_subset.remove_columns(["highlights"])

ds = ray.data.from_huggingface(hf_ds_subset)

ds = ds.map(
    format_into_prompt_rawtext,
    fn_kwargs=dict(
        template=PROMPT_TEMPLATE_QUESTION_GENERATION,
        tokenizer=AutoTokenizer.from_pretrained(HF_MODEL),
        col_name="qa_generation_prompt"
    ),
    num_cpus=0.01
)

ds = ds.map_batches(
    LLMPredictor,
    fn_constructor_kwargs=dict(
        model_location=HF_MODEL,
        col_in="qa_generation_prompt",
        col_out="qa_generation_raw_model_output",
        temperature=0,
        max_tokens=4096,
        vllm_settings=dict(
            tensor_parallel_size=NUM_GPUS_PER_INSTANCE,
            max_model_len=8192,
        )
    ),
    concurrency=NUM_LLM_INSTANCES,
    num_gpus=NUM_GPUS_PER_INSTANCE,
    accelerator_type=GPU_TYPE,
    batch_size=BATCH_SIZE,
)

ds = ds.map(
    shuffle_qa,
    fn_kwargs=dict(
        col_in="qa_generation_raw_model_output",
        col_out_prompt="qa_generation_questions",
        col_out_answers="qa_generation_answers",
    ),
    num_cpus=0
)

train_ds, test_ds = ds.train_test_split(test_size=0.05)

train_ds.write_parquet(OUTPUT_PATH_TRAIN)
test_ds.write_parquet(OUTPUT_PATH_TEST)
