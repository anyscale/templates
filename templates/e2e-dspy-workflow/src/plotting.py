"""Utilities for plotting evaluation results"""
import matplotlib.pyplot as plt
import numpy as np

def graph_testset_results(ft_results):
    # Find the best fine-tuned model based on highest average score
    best_ft_model = None
    best_ft_score = -float('inf')

    for model, results in ft_results.items():
        if model != "base":
            score = results['bfrs'].get('testset', 0)
            if score > best_ft_score:
                best_ft_score = score
                best_ft_model = model

    # Prepare data for the graph
    labels = [
        'Base Model (No Opt)',
        'Base Model (BFRS)',
        'Fine-tuned (No Opt)',
        'Fine-tuned (BFRS)'
    ]
    scores = [
        ft_results['base']['vanilla'].get('testset', 0),
        ft_results['base']['bfrs'].get('testset', 0),
        ft_results[best_ft_model]['vanilla'].get('testset', 0),
        ft_results[best_ft_model]['bfrs'].get('testset', 0)
    ]

    # Create color list (alternating between vanilla and BFRS colors)
    colors = ['skyblue', 'lightgreen', 'skyblue', 'lightgreen']

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(scores)), scores, color=colors)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}',
               ha='center', va='bottom', fontsize=10)

    # Customize the plot
    ax.set_ylabel('Test Set Scores')
    ax.set_title('Model Performance Comparison\n(Real Test Set; N=1000)')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='No Prompt Optimization'),
        Patch(facecolor='lightgreen', label='BootstrapFewShotRandomSearch')
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()

def graph_devset_results(ft_results):
    # Prepare data for the graph
    labels = []
    vanilla_scores = []
    bfrs_scores = []

    for model, results in ft_results.items():
        labels.append('Base Model' if model == 'base' else 'Fine-tuned')
        vanilla_scores.append(results['vanilla']['devset'])
        bfrs_scores.append(results['bfrs']['devset'])

    # Keep "Base Model" at the beginning and "Fine-tuned" at the end
    sorted_data = sorted(zip(labels, vanilla_scores, bfrs_scores),
                         key=lambda x: (x[0] != "Base Model", x[0] != "Fine-tuned", x[0]))
    labels, vanilla_scores, bfrs_scores = zip(*sorted_data)

    # Prepare data for the graph in the specified order
    all_labels = [
        f'{labels[0]} (No Opt)',
        f'{labels[0]} (BFRS)',
        f'{labels[-1]} (No Opt)',
        f'{labels[-1]} (BFRS)'
    ]
    all_scores = [
        vanilla_scores[0],
        bfrs_scores[0],
        vanilla_scores[-1],
        bfrs_scores[-1]
    ]

    # Create color list (alternating between vanilla and BFRS colors)
    colors = ['skyblue', 'lightgreen', 'skyblue', 'lightgreen']

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(all_scores)), all_scores, color=colors)

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}',
               ha='center', va='bottom', fontsize=10)

    # Customize the plot
    ax.set_ylabel('Dev Set Scores')
    ax.set_title('Model Performance Comparison\n(Synthetic Dev Set; N=1000)')
    ax.set_xticks(range(len(all_labels)))
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='No Prompt Optimization'),
        Patch(facecolor='lightgreen', label='BootstrapFewShotRandomSearch')
    ]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()

    # Find the highest devset score and its corresponding model
    highest_devset_score = max(bfrs_scores)
    highest_score_model = labels[bfrs_scores.index(highest_devset_score)]

    print(f"Highest Dev Set Score: {highest_devset_score:.1f}, Model: {highest_score_model}")
