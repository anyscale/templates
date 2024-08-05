import ray
import pandas as pd
import os

from utils.prompt_templates import PROMPT_TEMPLATE_SUMMARY
from utils.utils import init_logger

logger = init_logger()

INPUT_FOLDER = f"{os.environ.get('ANYSCALE_ARTIFACT_STORAGE')}/preference_tuning_summarization_example/summary_generation_fxwang-anyscale-dpo_mistral_instruct_lr_5e-06_beta_0.01_cpo_alpha_0.02_epoch1_v4/"
OUTPUT_TRAIN_FILE = f"{os.environ.get('ANYSCALE_ARTIFACT_STORAGE')}/preference_tuning_summarization_example/training_data/train.jsonl"
OUTPUT_VALID_FILE = f"{os.environ.get('ANYSCALE_ARTIFACT_STORAGE')}/preference_tuning_summarization_example/training_data/valid.jsonl"

MAX_PAIRS_PER_ARTICLE = 3

TRAIN_TEST_SPLIT = 0.01

def check_row(row):
    return row["summary_generation_raw_model_output"] is not None and row["qa_generation_answers"] is not None and row["judge_mc_answers"] is not None and "No Judge Output" not in row["judge_mc_answers"]

def eval_rows(row):
    return dict(
        **row,
        num_words = len(row["summary_generation_raw_model_output"].split()),
        accuracy = sum(row["qa_generation_answers"][i] == row["judge_mc_answers"][i] for i in range(len(row["judge_mc_answers"]))),
    )

def comp_ternary(num1, num2, thresh=0):
    if abs(num1 - num2) <= thresh:
        return 0
    elif num1 > num2:
        return 1
    elif num1 < num2:
        return -1
    else:
        return 0

def compare(row1, row2):

    len_comp = comp_ternary(row1["num_words"], row2["num_words"], thresh=2) * -1 # shorter is better
    acc_comp = comp_ternary(row1["accuracy"], row2["accuracy"], thresh=0)

    if min(row1["accuracy"], row2["accuracy"]) <= 2:
        return acc_comp
    return len_comp

def make_pairs(examples):
    pairs = []
    prompt = [{"content": PROMPT_TEMPLATE_SUMMARY.format(**examples.iloc[0]), "role": "user"}]
    for i in range(len(examples)):
        for j in range(i + 1, len(examples)):
            comp = compare(examples.iloc[i], examples.iloc[j])
            if comp == 0:
                continue
            elif comp == 1:
                pair = [examples.iloc[i], examples.iloc[j]]
            elif comp == -1:
                pair = [examples.iloc[j], examples.iloc[i]]
            pairs.append(dict(
                prompt=prompt,
                chosen=[{"content": pair[0]["summary_generation_raw_model_output"].strip(), "role": "assistant"}],
                rejected=[{"content": pair[1]["summary_generation_raw_model_output"].strip(), "role": "assistant"}],
                messages=[prompt[0], {"content": pair[0]["summary_generation_raw_model_output"].strip(), "role": "assistant"}],
                num_words_chosen=pair[0]["num_words"],
                num_words_rejected=pair[1]["num_words"],
                accuracy_chosen=pair[0]["accuracy"],
                accuracy_rejected=pair[1]["accuracy"],
            ))

    if len(pairs) == 0:
        return dict(
            prompt=[],
            chosen=[],
            rejected=[],
            num_words_chosen=[],
            num_words_rejected=[],
            accuracy_chosen=[],
            accuracy_rejected=[],
        )

    result = pd.DataFrame.from_records(pairs)
    if len(result) > MAX_PAIRS_PER_ARTICLE:
        result = result.sample(MAX_PAIRS_PER_ARTICLE)
    return result

ds = ray.data.read_parquet(INPUT_FOLDER)

ds = ds.filter(check_row, num_cpus=0)
ds = ds.map(eval_rows, num_cpus=0)
ds = ds.filter(lambda row : row["num_words"] < 200, num_cpus=0)

ds = ds.groupby("file").map_groups(make_pairs, num_cpus=0, batch_format="pandas")

train_ds, test_ds = ds.train_test_split(TRAIN_TEST_SPLIT)

train_df = train_ds.to_pandas()
test_df = test_ds.to_pandas()

logger.info(f"NUMBER TRAIN EXAMPLES: {len(train_df)}")
logger.info(f"NUMBER TEST EXAMPLES: {len(test_df)}")

train_df.to_json(OUTPUT_TRAIN_FILE, orient='records', lines=True)
test_df.to_json(OUTPUT_VALID_FILE, orient='records', lines=True)