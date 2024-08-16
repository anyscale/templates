"""
Get evaluation statistics based on scores for the model-generated summaries
"""

import argparse
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import ray
from tabulate import tabulate

from src.utils.common import check_num_bad_chars
from src.utils.models import DataSchema

parser = argparse.ArgumentParser()

parser.add_argument(
    "--outputs-path",
    type=str,
    required=True,
    help="Path to the folder with parquets for the model.",
)

parser.add_argument(
    "--baseline-outputs-path",
    type=str,
    required=True,
    help="Path to the folder with parquets for the baseline model to find win rate against.",
)
parser.add_argument(
    "--gpt4o-outputs-path",
    default=None,
    type=str,
    required=False,
    help="Path to the folder with parquets for GPT-4o model. This is optional",
)

parser.add_argument(
    "--results-dir",
    type=str,
    required=False,
    default="all_results.csv",
    help="Path to the results folder.",
)
parser.add_argument(
    "--accuracy-threshold",
    default=3,
    type=int,
    help="Score threshold to classify chosen and rejected samples.",
)


def is_row_valid(row: Dict[str, Any]):
    return not (
        row[DataSchema.SUMMARY_GENERATION_RAW_OUTPUT] is None
        or row[DataSchema.GROUND_TRUTH_MCQ_ANSWERS] is None
        or row[DataSchema.JUDGE_MCQ_ANSWERS] is None
        or "No Judge Output" in row[DataSchema.JUDGE_MCQ_ANSWERS]
    )


def eval_row(row):
    return dict(
        **row,
        num_words=len(row[DataSchema.SUMMARY_GENERATION_RAW_OUTPUT].split()),
        accuracy=sum(
            row[DataSchema.GROUND_TRUTH_MCQ_ANSWERS][i]
            == row[DataSchema.JUDGE_MCQ_ANSWERS][i]
            for i in range(len(row[DataSchema.GROUND_TRUTH_MCQ_ANSWERS]))
        ),
        num_bad_chars=check_num_bad_chars(
            row[DataSchema.SUMMARY_GENERATION_RAW_OUTPUT], normalize=True
        ),
    )


def compare(
    acc1: float, num1: int, acc2: float, num2: int, *, accuracy_threshold
) -> bool:
    """Compare two summaries based on accuracy (of judge responses) and length (of model summary) for evaluation.

    Args:
        acc1: Accuracy (of judge responses based on the summary) for the first summary
        num1: Number of words in the first summary
        acc2: Accuracy (of judge responses based on the summary) for the second summary
        num2: Number of words in the second summary

    Returns:
        Whether the first summary is preferred or not.
    """
    if min(acc1, acc2) <= accuracy_threshold - 1:
        if acc1 != acc2:
            return acc1 > acc2
    return num1 < num2


def get_model_stats(merged_results: pd.DataFrame, suffix: str) -> pd.Series:
    lens = np.array([len(row["text"].split()) for _, row in merged_results.iterrows()])
    return pd.Series(
        {
            "Accuracy >=3": np.mean(
                merged_results[f"{DataSchema.ACCURACY}{suffix}"] >= 3
            ),
            "Accuracy >=4": np.mean(
                merged_results[f"{DataSchema.ACCURACY}{suffix}"] >= 4
            ),
            "Median Compression": np.median(
                merged_results[f"{DataSchema.NUM_WORDS}{suffix}"] / lens
            ),
            "Mean Compression": np.mean(
                merged_results[f"{DataSchema.NUM_WORDS}{suffix}"] / lens
            ),
            "Failed Compressions": np.mean(
                merged_results[
                    f"{DataSchema.SUMMARY_GENERATION_RAW_OUTPUT}{suffix}"
                ].str.len()
                >= merged_results["text"].str.len()
            ),
            "Contains OOD Characters": np.mean(
                merged_results[f"num_bad_chars{suffix}"] > 0
            ),
        }
    )


def get_win_rate(merged_results: pd.DataFrame, suffix1: str, suffix2: str) -> float:
    wins = [
        compare(*vals, accuracy_threshold=accuracy_threshold)
        for vals in zip(
            merged_results[f"{DataSchema.ACCURACY}{suffix1}"],
            merged_results[f"{DataSchema.NUM_WORDS}{suffix1}"],
            merged_results[f"{DataSchema.ACCURACY}{suffix2}"],
            merged_results[f"{DataSchema.NUM_WORDS}{suffix2}"],
        )
    ]

    losses = [
        compare(*vals, accuracy_threshold=accuracy_threshold)
        for vals in zip(
            merged_results[f"{DataSchema.ACCURACY}{suffix2}"],
            merged_results[f"{DataSchema.NUM_WORDS}{suffix2}"],
            merged_results[f"{DataSchema.ACCURACY}{suffix1}"],
            merged_results[f"{DataSchema.NUM_WORDS}{suffix1}"],
        )
    ]

    win_rate = np.mean(wins) + 0.5 * (1 - np.mean(wins) - np.mean(losses))
    return 100 * win_rate


def calculate_statistics(
    results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    gpt_4o_results: Optional[pd.DataFrame],
    accuracy_threshold: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    merge_col = DataSchema.ARTICLE
    merged_results = pd.merge(
        results, baseline_results, on=merge_col, suffixes=("_x", "_y")
    )
    if gpt_4o_results is not None:
        gpt_4o_results = gpt_4o_results.rename(
            columns={
                col: f"{col}_z" for col in gpt_4o_results.columns if col != merge_col
            }
        )
        merged_results = pd.merge(
            merged_results, gpt_4o_results, on=merge_col, suffixes=(None, "_")
        )

    # stores win rates against the baseline model
    win_rates = {"Model": get_win_rate(merged_results, suffix1="_x", suffix2="_y")}

    cols = {
        "Model": get_model_stats(merged_results, "_x"),
        "Baseline": get_model_stats(merged_results, "_y"),
    }
    if gpt_4o_results is not None:
        cols.update({"GPT-4o": get_model_stats(merged_results, "_z")})
        win_rates.update(
            {"GPT-4o": get_win_rate(merged_results, suffix1="_z", suffix2="_y")}
        )
    stats_df = pd.DataFrame(cols)
    return stats_df, win_rates


def format_dataframe(df: pd.DataFrame) -> str:
    """Formats the dataframe into a string"""
    # Format the float values to 4 decimal places
    formatted_df = df.applymap(lambda x: 100 * x).applymap(lambda x: f"{x:.4f} %")
    formatted_df.index.name = "Metric"
    # Create a table using tabulate
    table = tabulate(
        formatted_df, headers="keys", tablefmt="fancy_grid", stralign="center"
    )

    return table


def preprocess_ray_ds_for_eval(ds: ray.data.Dataset) -> pd.DataFrame:
    ds = ds.filter(is_row_valid)
    ds = ds.map(eval_row)
    results = ds.to_pandas()
    return results


if __name__ == "__main__":
    args = parser.parse_args()

    ds = ray.data.read_parquet(args.outputs_path, file_extensions=["parquet"])
    ds_baseline = ray.data.read_parquet(
        args.baseline_outputs_path, file_extensions=["parquet"]
    )

    results = preprocess_ray_ds_for_eval(ds)

    results_baseline = preprocess_ray_ds_for_eval(ds_baseline)

    results_ds_gpt4o = None
    if args.gpt4o_outputs_path:
        ds_gpt4o = ray.data.read_parquet(
            args.gpt4o_outputs_path, file_extensions=["parquet"]
        )
        results_ds_gpt4o = preprocess_ray_ds_for_eval(ds_gpt4o)

    print("Num Results:", len(results))
    print("Num Baseline Results:", len(results_baseline))

    accuracy_threshold = args.accuracy_threshold
    stats_df, win_rates = calculate_statistics(
        results=results,
        baseline_results=results_baseline,
        gpt_4o_results=results_ds_gpt4o,
        accuracy_threshold=accuracy_threshold,
    )
    print(format_dataframe(stats_df))
    print("\n")
    for name, win_rate in win_rates.items():
        print(f"{name} Win Rate against Baseline: {win_rate:.4f} %")
