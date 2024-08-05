import ray
import os

from transformers import AutoTokenizer

from utils.synthetic_data_utils import *

from utils.prompt_templates import PROMPT_TEMPLATE_MCQ_ANSWERING, PROMPT_TEMPLATE_SUMMARY

logger = init_logger()

# summary model settings
MODEL_LOCATION = "mistralai/Mistral-7B-Instruct-v0.1" # can be an NFS folder or huggingface model
LORA_LOCATION = "fxwang-anyscale/dpo_mistral_instruct_lr_5e-06_beta_0.01_cpo_alpha_0.02_epoch1" # can be an NFS folder or huggingface model
TOKENIZER_LOCATION = MODEL_LOCATION # optionally specify alternate tokenizer location for chat template

# generation settings
TEMPERATURE = 1 # use temperature 0 for eval, temperature 1 for training data generation
NUM_GENERATIONS = 10 # use 1 generation for eval, 10 for training data generation

# summary generation compute settings
# The number of LLM instances to use.
NUM_LLM_INSTANCES = 8
# The number of GPUs to use per LLM instance. NUM_GPUS_PER_INSTANCE > 1 will use tensor parallelism across all GPUs.
NUM_GPUS_PER_INSTANCE = 1
# The type of GPU to use
GPU_TYPE = "H100" # get_a10g_or_equivalent_accelerator_type()
# Batch size for each instance
BATCH_SIZE = 1024

# judge model settings
JUDGE_MODEL_LOCATION = "meta-llama/Meta-Llama-3.1-8B-Instruct"
JUDGE_NUM_LLM_INSTANCES = 8
JUDGE_NUM_GPUS_PER_INSTANCE = 1
JUDGE_GPU_TYPE = "H100"
JUDGE_BATCH_SIZE = 1024

INPUT_FOLDER = f"{os.environ.get('ANYSCALE_ARTIFACT_STORAGE')}/preference_tuning_summarization_example/qa_annotations_train_full"
OUTPUT_FOLDER = f"{os.environ.get('ANYSCALE_ARTIFACT_STORAGE')}/preference_tuning_summarization_example/summary_generation_fxwang-anyscale-dpo_mistral_instruct_lr_5e-06_beta_0.01_cpo_alpha_0.02_epoch1_v4/"

NUM_MCQ_QUESTIONS = 5


if __name__ == "__main__":

    # Initialize Ray with a Runtime Environment.
    ray.init(
        runtime_env={
            "env_vars": {"HF_TOKEN": os.environ["HF_TOKEN"], "HF_HOME": "/mnt/local_storage/.cache/huggingface"},
        }
    )

    ds = ray.data.read_parquet(INPUT_FOLDER, file_extensions=["parquet"])

    ds = ds.filter(
        lambda row : row["qa_generation_answers"] is not None and len(row["qa_generation_answers"]) == NUM_MCQ_QUESTIONS,
        num_cpus=0
    )

    def qa_generation_answers_to_numpy(row):
        row["qa_generation_answers"] = np.array(row["qa_generation_answers"])
        return row

    ds = ds.map(qa_generation_answers_to_numpy, num_cpus=0)

    ds = ds.map(
        format_into_prompt_rawtext,
        fn_kwargs=dict(
            template=PROMPT_TEMPLATE_SUMMARY,
            tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_LOCATION),
            col_name="summary_generation_prompt"
        ),
        num_cpus=0
    )

    ds = ds.flat_map(
        duplicate_rows,
        fn_kwargs=dict(
            count=NUM_GENERATIONS,
            id_col="response_num",
        ),
        num_cpus=0,
    )

    ds = ds.repartition(NUM_LLM_INSTANCES * 32)

    ds = ds.map_batches(
        LLMPredictor,
        fn_constructor_kwargs=dict(
            col_in="summary_generation_prompt",
            col_out="summary_generation_raw_model_output",
            model_location=MODEL_LOCATION,
            lora_location=LORA_LOCATION,
            temperature=TEMPERATURE,
            max_tokens=4096,
            vllm_settings=dict(
                tensor_parallel_size=NUM_GPUS_PER_INSTANCE,
                max_model_len=8192,
            )
        ),
        num_gpus=NUM_GPUS_PER_INSTANCE,
        accelerator_type=GPU_TYPE,
        concurrency=NUM_LLM_INSTANCES,
        batch_size=BATCH_SIZE,
        zero_copy_batch=True,
        batch_format="numpy"
    )

    ds = ds.map(
        format_into_prompt_rawtext,
        fn_kwargs=dict(
            template=PROMPT_TEMPLATE_MCQ_ANSWERING,
            tokenizer=AutoTokenizer.from_pretrained(JUDGE_MODEL_LOCATION),
            col_name="judge_mc_prompt"
        ),
        num_cpus=0
    )

    ds = ds.map_batches(
        LLMPredictor,
        fn_constructor_kwargs=dict(
            col_in="judge_mc_prompt",
            col_out="judge_mc_raw_model_output",
            model_location=JUDGE_MODEL_LOCATION,
            temperature=0,
            max_tokens=4096,
            vllm_settings=dict(
                tensor_parallel_size=JUDGE_NUM_GPUS_PER_INSTANCE,
                max_model_len=8192,
            )
        ),
        num_gpus=JUDGE_NUM_GPUS_PER_INSTANCE,
        accelerator_type=JUDGE_GPU_TYPE,
        concurrency=JUDGE_NUM_LLM_INSTANCES,
        batch_size=JUDGE_BATCH_SIZE,
        zero_copy_batch=True,
        batch_format="numpy"
    )

    ds = ds.map(
        extract_answers,
        fn_kwargs=dict(
            col_in="judge_mc_raw_model_output",
            col_out="judge_mc_answers",
            num_questions=NUM_MCQ_QUESTIONS,
        ),
        num_cpus=0,
    )

    ds.write_parquet(OUTPUT_FOLDER)