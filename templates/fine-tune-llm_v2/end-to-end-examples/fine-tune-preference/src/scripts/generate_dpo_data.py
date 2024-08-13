"""
Script to generate training data for DPO based on the generated summaries and scores.
"""

import os
import argparse
from functools import partial
from typing import Literal, Dict, Any

import pandas as pd
import ray
from pydantic import Field

from src.utils.prompt_templates import PROMPT_TEMPLATE_SUMMARY
from src.utils.common import init_logger, check_num_bad_chars
from src.utils.models import BaseModelExtended, DataSchema

logger = init_logger()

# NOTE: For a pair of summaries where the accuracies are above the threshold, we compare them by length. We prefer smaller summaries. We set a minimum difference of 3 words for one example to be distinct from another.
MIN_LENGTH_DIFFERENCE = 3

MIN_NUM_WORDS_IN_SUMMARY = 5
MAX_NUM_WORDS_IN_SUMMARY = 200



class TrainingDataGenerationConfig(BaseModelExtended):
    input_folder: str = Field(description="Input folder path with generated summaries and scores from the base model, relative to the base artifact storage path.The folder is expected to be compatible with `ray.data.read_parquet`.")
    max_pairs_per_article: int = Field(default=3, description="Maximum number of chosen, rejected pairs to sample per article.")
    train_val_split: float = Field(default=0.02, description="Train validation split ratio")
    accuracy_threshold: int = Field(default=3, description="Score threshold to classify chosen and rejected samples.")
    output_folder: str = Field(description="Output folder path for train and validation files, relative to the base artifact storage path.")


def is_row_valid(row: Dict[str, Any]) -> bool:
    """Checks if a row is a valid entry for the DPO training.

    Bad entries include onces with empty summaries, empty judge responses or ones with invalid characters in the summary. (characters not present in the original dataset)

    Args:
        row: A dict representing an entry from the input DataFrame.
    """
    return (
        row[DataSchema.SUMMARY_GENERATION_RAW_OUTPUT_FIELD] is not None
        and row[DataSchema.GROUND_TRUTH_MCQ_ANSWERS_FIELD] is not None
        and row[DataSchema.JUDGE_MCQ_ANSWERS_FIELD] is not None
        and "No Judge Output" not in row[DataSchema.JUDGE_MCQ_ANSWERS_FIELD]
        and check_num_bad_chars(
            row[DataSchema.SUMMARY_GENERATION_RAW_OUTPUT_FIELD], normalize=True
        )
        == 0
    )


def eval_row(row: Dict[str, Any]):
    """Evaluates model summary and judge responses in a row.

    Returns the original row along with two additional columns - the number of words and the accuracy of the judge

    Args:
        row: A dict representing an entry from the input DataFrame.
    """
    return dict(
        **row,
        num_words=len(row[DataSchema.SUMMARY_GENERATION_RAW_OUTPUT_FIELD].split()),
        accuracy=sum(
            row[DataSchema.GROUND_TRUTH_MCQ_ANSWERS_FIELD][i] == row[DataSchema.JUDGE_MCQ_ANSWERS_FIELD][i]
            for i in range(len(row[DataSchema.JUDGE_MCQ_ANSWERS_FIELD]))
        ),
    )


def compare_summaries(row1: Dict[str, Any], row2: Dict[str, Any], *, accuracy_threshold) -> Literal[0, 1, -1]:
    """
    Compare two summaries based on accuracy (of judge responses) and length (of model summary).

    Args:
        row1: Input DataFrame row for the first summary
        row2: Input DataFrame row for the second summary

    Returns:
        1 if `row1` is preferred, -1 if `row2` is preferred, 0 if both are equivalent.
    """
    # If atleast one summary is worse than the threshold, choose based on the higher accuracy
    if min(row1[DataSchema.ACCURACY_FIELD], row2[DataSchema.ACCURACY_FIELD]) <= accuracy_threshold - 1:
        # First, compare based on accuracy
        if row1[DataSchema.ACCURACY_FIELD] > row2[DataSchema.ACCURACY_FIELD]:
            return 1
        elif row2[DataSchema.ACCURACY_FIELD] > row1[DataSchema.ACCURACY_FIELD]:
            return -1
        return 0

    # If accuracies are above the threshold, prefer the shorter summary
    length_diff = row1[DataSchema.NUM_WORDS_FIELD] - row2[DataSchema.NUM_WORDS_FIELD]
    if abs(length_diff) >= MIN_LENGTH_DIFFERENCE:
        return -1 if length_diff > 0 else 1

    # If lengths are similar, consider them equivalent
    return 0

def make_pairs(examples: pd.DataFrame, max_pairs_per_article: int, accuracy_threshold: int) -> pd.DataFrame:
    """Makes training input pairs for DPO for the given DataFrame.

    Args:
        examples: Input DataFrame
        max_pairs_per_article: Maximum number of training data pairs to sample for one article.
        accuracy_threshold: Score threshold to classify chosen and rejected samples.
    Returns:
        result: Output DataFrame in the preference tuning format.
    """
    pairs = []
    prompt = {
        "content": PROMPT_TEMPLATE_SUMMARY.format(**examples.iloc[0]),
        "role": "user",
    }
    for i in range(len(examples)):
        for j in range(i + 1, len(examples)):
            comp = compare_summaries(examples.iloc[i], examples.iloc[j], accuracy_threshold=accuracy_threshold)
            if comp == 0:
                continue
            pair = [examples.iloc[i], examples.iloc[j]] if comp == 1 else [examples.iloc[j], examples.iloc[i]]
            pairs.append(
                dict(
                    chosen=[
                        prompt,
                        {
                            "content": pair[0][
                                DataSchema.SUMMARY_GENERATION_RAW_OUTPUT_FIELD
                            ].strip(),
                            "role": "assistant",
                        },
                    ],
                    rejected=[
                        prompt,
                        {
                            "content": pair[1][
                                DataSchema.SUMMARY_GENERATION_RAW_OUTPUT_FIELD
                            ].strip(),
                            "role": "assistant",
                        },
                    ],
                    num_words_chosen=pair[0][DataSchema.NUM_WORDS_FIELD],
                    num_words_rejected=pair[1][DataSchema.NUM_WORDS_FIELD],
                    accuracy_chosen=pair[0][DataSchema.ACCURACY_FIELD],
                    accuracy_rejected=pair[1][DataSchema.ACCURACY_FIELD],
                )
            )

    if len(pairs) == 0:
        # return empty dataframe
        return pd.DataFrame(columns=["chosen", "rejected", "num_words_chosen", "num_words_rejected", "accuracy_chosen", "accuracy_rejected"])

    result = pd.DataFrame.from_records(pairs)
    if len(result) > max_pairs_per_article:
        result = result.sample(max_pairs_per_article)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A simple script for summary generation and scoring with support for offline and online inference."
    )
    parser.add_argument(
        "config_path", help="Path to the config file for training dataset generation"
    )
    args = parser.parse_args()
    config = TrainingDataGenerationConfig.from_yaml(args.config_path)
    input_folder = os.path.join(os.environ["ANYSCALE_ARTIFACT_STORAGE"], config.input_folder)
    output_folder = os.path.join(os.environ["ANYSCALE_ARTIFACT_STORAGE"], config.output_folder)
    ds = ray.data.read_parquet(input_folder, file_extensions=["parquet"])

    ds = ds.filter(is_row_valid, num_cpus=0)
    ds = ds.map(eval_row, num_cpus=0)
    ds = ds.filter(lambda row: MIN_NUM_WORDS_IN_SUMMARY <= row[DataSchema.NUM_WORDS_FIELD] < MAX_NUM_WORDS_IN_SUMMARY, num_cpus=0)

    ds = ds.groupby("id").map_groups(make_pairs, fn_kwargs=dict(max_pairs_per_article=config.max_pairs_per_article, accuracy_threshold=config.accuracy_threshold), num_cpus=0, batch_format="pandas")

    train_ds, val_ds = ds.train_test_split(config.train_val_split)

    train_df = train_ds.to_pandas()
    val_df = val_ds.to_pandas()


    logger.info(f"Number of train examples: {len(train_df)}")
    logger.info(f"Number of eval examples: {len(val_df)}")

    train_df.to_json(os.path.join(output_folder, "train.jsonl"), orient="records", lines=True)
    val_df.to_json(os.path.join(output_folder, "val.jsonl"), orient="records", lines=True)
