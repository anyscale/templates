import argparse
import os

import numpy as np
import pandas as pd
import ray

from utils.synthetic_data_utils import check_num_bad_chars

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
    required=False,
    default=f"{os.environ['ANYSCALE_ARTIFACT_STORAGE']}/preference_tuning_summarization_example/summary_eval_generation_mistralai_Mistral-7B-Instruct-v0.1_temp_0_judge_meta-llama_Meta-Llama-3.1-70B-Instruct/",
    help="Path to the folder with parquets for the baselien model to find win rate against.",
)

parser.add_argument(
    "--results-dir",
    type=str,
    required=False,
    default="all_results.csv",
    help="Path to the results folder.",
)

parser.add_argument("--disable-csv", action="store_true")

args = parser.parse_args()

try:
    ds = ray.data.read_parquet(args.outputs_path, file_extensions=["parquet"])
except Exception as e:
    ds = ray.data.read_parquet(
        f"{os.environ['ANYSCALE_ARTIFACT_STORAGE']}/preference_tuning_summarization_example/"
        + args.outputs_path,
        file_extensions=["parquet"],
    )
ds_orig = ray.data.read_parquet(args.baseline_outputs_path, file_extensions=["parquet"])


def eval_rows(row):
    if (
        row["summary_generation_raw_model_output"] is None
        or row["qa_generation_answers"] is None
        or row["judge_mc_answers"] is None
        or "No Judge Output" in row["judge_mc_answers"]
    ):
        return []

    return [
        dict(
            **row,
            num_words=len(row["summary_generation_raw_model_output"].split()),
            accuracy=sum(
                row["qa_generation_answers"][i] == row["judge_mc_answers"][i]
                for i in range(len(row["qa_generation_answers"]))
            ),
            num_bad_chars=check_num_bad_chars(
                row["summary_generation_raw_model_output"], normalize=True
            ),
            # accuracy_filtered = sum(row["qa_generation_answers"][i] == row["judge_mc_answers"][i] for i in good_questions) / len(good_questions),
        )
    ]


def compare(acc1, num1, acc2, num2):
    if min(acc1, acc2) <= 2:
        if acc1 != acc2:
            return acc1 > acc2
    return num1 < num2


ds = ds.flat_map(eval_rows)
results = ds.to_pandas()

ds_orig = ds_orig.flat_map(eval_rows)
results_orig = ds_orig.to_pandas()

print("NUM RESULTS:", len(results))
print("NUM BASELINE RESULTS:", len(results_orig))

merged_results = pd.merge(results, results_orig, on="text")

lens = np.array(
    [len(merged_results.iloc[i]["text"].split()) for i in range(len(merged_results))]
)

wins = [
    compare(*vals)
    for vals in zip(
        merged_results["accuracy_x"],
        merged_results["num_words_x"],
        merged_results["accuracy_y"],
        merged_results["num_words_y"],
    )
]
losses = [
    compare(*vals)
    for vals in zip(
        merged_results["accuracy_y"],
        merged_results["num_words_y"],
        merged_results["accuracy_x"],
        merged_results["num_words_x"],
    )
]

if not args.disable_csv:
    all_stats = pd.read_csv(args.results_dir, index_col=0)

    new_row = pd.DataFrame(
        [
            {
                "Win Rate": np.mean(wins) + 0.5 * (1 - np.mean(wins) - np.mean(losses)),
                "% Accuracy >=3": np.mean(merged_results["accuracy_x"] >= 3),
                "% Accuracy >=4": np.mean(merged_results["accuracy_x"] >= 4),
                "Median Compression": np.median(merged_results["num_words_x"] / lens),
                "Failed Compressions": np.mean(
                    merged_results["summary_generation_raw_model_output_x"].str.len()
                    >= merged_results["text"].str.len()
                ),
                "Num Contains /******/": np.mean(
                    merged_results["summary_generation_raw_model_output_x"].str.find(
                        "/******/"
                    )
                    != -1
                ),
                "Num Contains Bad Characters": np.mean(
                    merged_results["num_bad_chars_x"] > 0
                ),
            }
        ],
        index=[args.outputs_path.strip("/").split("/")[-1]],
    )

    all_stats = new_row.combine_first(all_stats)
    all_stats.to_csv(args.results_dir, index_label="Name")

print("Win Rate:", np.mean(wins) + 0.5 * (1 - np.mean(wins) - np.mean(losses)))
print("% Accuracy >=3:", np.mean(merged_results["accuracy_x"] >= 3))
print("% Accuracy >=4:", np.mean(merged_results["accuracy_x"] >= 4))
print("Median Compression:", np.median(merged_results["num_words_x"] / lens))
print("Mean Compression:", np.mean(merged_results["num_words_x"] / lens))
print(
    "Failed Compressions:",
    np.mean(
        merged_results["summary_generation_raw_model_output_x"].str.len()
        >= merged_results["text"].str.len()
    ),
)
print(
    "Contains /******/:",
    np.mean(
        merged_results["summary_generation_raw_model_output_x"].str.find("/******/")
        != -1
    ),
)
print("Contains Bad Characters:", np.mean(merged_results["num_bad_chars_x"] > 0))

print("Baseline % Accuracy >=3:", np.mean(merged_results["accuracy_y"] >= 3))
print("Baseline % Accuracy >=4:", np.mean(merged_results["accuracy_y"] >= 4))
print("Baseline Median Compression", np.median(merged_results["num_words_y"] / lens))
print("Baseline Mean Compression", np.mean(merged_results["num_words_y"] / lens))
print(
    "Baseline Failed Compressions:",
    np.mean(
        merged_results["summary_generation_raw_model_output_y"].str.len()
        >= merged_results["text"].str.len()
    ),
)
print(
    "Baseline Contains /******/:",
    np.mean(
        merged_results["summary_generation_raw_model_output_y"].str.find("/******/")
        != -1
    ),
)
print(
    "Baseline Contains Bad Characters:", np.mean(merged_results["num_bad_chars_y"] > 0)
)
