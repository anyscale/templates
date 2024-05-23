"""
Plot utils for visualizing the results of the finetuned model and GPT-4.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from fc_utils.eval_core import Mistakes, Result

# Colors for the different mistake types
COLORS = {
    Mistakes.UNWANTED_FUNCTION_CALL: "springgreen",
    Mistakes.NO_FUNCTION_CALL: "limegreen",
    Mistakes.INCORRECT_FORMAT: "deepskyblue",
    Mistakes.INCORRECT_NUMBER_OF_FUNCTION_CALLS: "turquoise",
    Mistakes.WRONG_FUNCTION_NAME: "gold",
    Mistakes.WRONG_ARGUMENT_VALUE: "salmon",
    Mistakes.MISSING_ARGUMENT: "orange",
    Mistakes.NONE: "blueviolet",
}


def get_count_by_flag(results: List[Result], flags: List[Mistakes]) -> Dict[str, int]:
    """Returns the count of mistakes by flag for the given results."""
    count_by_flag = {
        flag: len(
            [result for result in results if result.mistake_type.value == flag.value]
        )
        for flag in flags
    }
    return count_by_flag


def plot_results(
    results_base: List[Result],
    results_finetuned: List[Result],
    results_gpt: List[Result],
):
    """
    Plots results for the finetuned model and GPT-4

    Args:
        results_finetuned: List of results for the finetuned model
        results_gpt: List of results for GPT-4
    """
    # Data for plotting
    total_count = len(results_finetuned)
    # Get all mistake types
    flags = Mistakes.instances()
    results_base = [result for result in results_base if result.is_correct == False]
    results_finetuned = [
        result for result in results_finetuned if result.is_correct == False
    ]
    results_gpt = [result for result in results_gpt if result.is_correct == False]
    total_incorrect_base = len(results_base)
    total_incorrect_finetuned = len(results_finetuned)
    total_incorrect_gpt = len(results_gpt)

    counts_base = get_count_by_flag(results_base, flags)
    counts_finetuned = get_count_by_flag(results_finetuned, flags)
    counts_gpt = get_count_by_flag(results_gpt, flags)

    # Data for stacked bars
    mistakes_by_flag_base = [counts_base[flag] for flag in flags]
    mistakes_by_flag_finetuned = [counts_finetuned[flag] for flag in flags]
    mistakes_by_flag_gpt = [counts_gpt[flag] for flag in flags]

    # Bar positions
    positions = np.arange(3)

    # Create the plot
    fig, ax = plt.subplots()

    # Create stacked bars
    bottom_base = np.zeros(1)
    bottom_finetuned = np.zeros(1)
    bottom_gpt = np.zeros(1)

    for i, flag in enumerate(flags):
        if (
            mistakes_by_flag_base[i] == 0
            and mistakes_by_flag_finetuned[i] == 0
            and mistakes_by_flag_gpt[i] == 0
        ):
            # Skip if no mistakes of this type
            continue
        ax.bar(
            0,
            mistakes_by_flag_base[i],
            bottom=bottom_base,
            color=COLORS[flag],
            label=f"{flag.value}",
        )
        ax.bar(
            1,
            mistakes_by_flag_finetuned[i],
            bottom=bottom_finetuned,
            color=COLORS[flag],
        )
        ax.bar(2, mistakes_by_flag_gpt[i], bottom=bottom_gpt, color=COLORS[flag])
        bottom_base += mistakes_by_flag_base[i]
        bottom_finetuned += mistakes_by_flag_finetuned[i]
        bottom_gpt += mistakes_by_flag_gpt[i]

    # Add labels and title
    ax.set_xlabel("Results")
    ax.set_ylabel("Number of Mistakes")
    ax.set_title("Error Analysis")
    ax.set_xticks(positions)
    ax.set_ylim(
        # Set the y-axis limit to be at least 20% of the total count for a nicer plot
        ymax=max(
            total_incorrect_base,
            total_incorrect_finetuned,
            total_incorrect_gpt,
            0.2 * total_count,
        )
    )
    ax.set_xticklabels(["Base Model", "Finetuned Model", "GPT-4"])
    ax.legend()

    # Display the chart
    plt.show()
