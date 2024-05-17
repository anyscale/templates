"""
Plot utils for visualizing the results of the finetuned model and GPT-4.
"""
import numpy as np
import matplotlib.pyplot as plt

from fc_utils.eval_utils import Mistakes

 # 10 colors for different flags
COLORS = ['salmon', 'deepskyblue', 'springgreen', 'orange', "gold", "limegreen", "turquoise", "blueviolet", "darkred", "darkorange"]

def plot_results(results_finetuned, results_gpt):
    """
    Plots results for the finetuned model and GPT-4
    """
    # Data for plotting
    total_count = len(results_finetuned)
    flags = Mistakes.values()
    results_finetuned = [result for result in results_finetuned if result["correct"] == False]
    results_gpt = [result for result in results_gpt if result["correct"] == False]
    total_incorrect_finetuned = len(results_finetuned)
    total_incorrect_gpt = len(results_gpt)
    # function to count different types of mistakes
    count_by_flag = lambda results: {flag: len([result for result in results if result["mistake_type"].value == flag]) for flag in flags}

    counts_1 = count_by_flag(results_finetuned)
    counts_2 = count_by_flag(results_gpt)

    # Data for stacked bars
    mistakes_by_flag_1 = [counts_1[flag] for flag in flags]
    mistakes_by_flag_2 = [counts_2[flag] for flag in flags]

    # Bar positions
    positions = np.arange(2)

    # Create the plot
    fig, ax = plt.subplots()

    # Create stacked bars
    bottom_1 = np.zeros(1)
    bottom_2 = np.zeros(1)

    for i, flag in enumerate(flags):
        if mistakes_by_flag_1[i] == 0 and mistakes_by_flag_2[i] == 0:
            continue # skip if no mistakes of this type
        bar_1 = ax.bar(0, mistakes_by_flag_1[i], bottom=bottom_1, color=COLORS[i], label=f'{flag}')
        bar_2 = ax.bar(1, mistakes_by_flag_2[i], bottom=bottom_2, color=COLORS[i])
        bottom_1 += mistakes_by_flag_1[i]
        bottom_2 += mistakes_by_flag_2[i]

    # Add labels and title
    ax.set_xlabel('Results')
    ax.set_ylabel('Number of Mistakes')
    ax.set_title('Error Analysis')
    ax.set_xticks(positions)
    ax.set_ylim(ymax=max(total_incorrect_finetuned, total_incorrect_gpt, 0.2*total_count))
    ax.set_xticklabels(['Finetuned Model', 'GPT-4'])
    ax.legend()

    # Display the chart
    plt.show()
