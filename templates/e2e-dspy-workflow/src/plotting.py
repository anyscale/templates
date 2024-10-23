import matplotlib.pyplot as plt
import numpy as np

def create_performance_graph(scores, labels, title, y_label):
    colors = ['skyblue', 'lightgreen', 'skyblue', 'lightgreen']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(scores)), scores, color=colors)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}',
               ha='center', va='bottom', fontsize=10)

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='No Prompt Optimization/Zero shot'),
        Patch(facecolor='lightgreen', label='BootstrapFewShotRandomSearch')
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()

def graph_testset_results(ft_results):
    """
    Graphs the performance of the models on the real test set.

    Expects ft_results to be a dictionary with the following structure:
    {
        'base': {'vanilla': {'testset': float}, 'bfrs': {'testset': float}},
        '<any_other_model_name>': {'vanilla': {'testset': float}, 'bfrs': {'testset': float}},
    }
    Currently only supports two models.
    """
    best_ft_model_vanilla = max((model for model in ft_results if model != "base"),
                                key=lambda m: ft_results[m]['vanilla'].get('testset', 0))
    best_ft_model_bfrs = max((model for model in ft_results if model != "base"),
                             key=lambda m: ft_results[m]['bfrs'].get('testset', 0))

    labels = [
        'Base Model (Zero shot)',
        'Base Model (BFRS)',
        'Best Fine-tuned (Zero shot)',
        'Best Fine-tuned (BFRS)'
    ]
    scores = [
        ft_results['base']['vanilla'].get('testset', 0),
        ft_results['base']['bfrs'].get('testset', 0),
        ft_results[best_ft_model_vanilla]['vanilla'].get('testset', 0),
        ft_results[best_ft_model_bfrs]['bfrs'].get('testset', 0)
    ]

    create_performance_graph(scores, labels, 'Model Performance Comparison\n(Real Test Set; N=1000)', 'Test Set Scores')

def graph_devset_results(ft_results):
    """
    Graphs the performance of the models on the synthetic dev set.

    Expects ft_results to be a dictionary with the following structure:
    {
        'base': {'vanilla': {'devset': float}, 'bfrs': {'devset': float}},
        '<any_other_model_name>': {'vanilla': {'devset': float}, 'bfrs': {'devset': float}},
    }
    Currently only supports two models.
    """
    labels = ['Base Model' if model == 'base' else 'Fine-tuned' for model in ft_results]
    vanilla_scores = [results['vanilla']['devset'] for results in ft_results.values()]
    bfrs_scores = [results['bfrs']['devset'] for results in ft_results.values()]

    sorted_data = sorted(zip(labels, vanilla_scores, bfrs_scores),
                         key=lambda x: (x[0] != "Base Model", x[0] != "Fine-tuned", x[0]))
    labels, vanilla_scores, bfrs_scores = zip(*sorted_data)

    all_labels = [f'{labels[0]} (Zero shot)', f'{labels[0]} (BFRS)',
                  f'{labels[-1]} (Zero shot)', f'{labels[-1]} (BFRS)']
    all_scores = [vanilla_scores[0], bfrs_scores[0],
                  vanilla_scores[-1], bfrs_scores[-1]]

    create_performance_graph(all_scores, all_labels, 'Model Performance Comparison\n(Synthetic Dev Set; N=1000)', 'Dev Set Scores')

    highest_vanilla_score = max(vanilla_scores)
    highest_bfrs_score = max(bfrs_scores)

    best_vanilla_model = labels[vanilla_scores.index(highest_vanilla_score)]
    best_bfrs_model = labels[bfrs_scores.index(highest_bfrs_score)]

    print(f"Highest Dev Set Score (No Opt): {highest_vanilla_score:.1f}, Model: {best_vanilla_model}")
    print(f"Highest Dev Set Score (BFRS): {highest_bfrs_score:.1f}, Model: {best_bfrs_model}")
