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
    models = []
    vanilla_devset = []
    bfrs_devset = []

    for model, results in ft_results.items():
        if model == "base":
            models.append("base")
        else:
            models.append("fine-tuned")
        vanilla_devset.append(results['vanilla']['devset'])
        bfrs_devset.append(results['bfrs']['devset'])

    # Keep "base" at the beginning
    sorted_data = sorted(zip(models, vanilla_devset, bfrs_devset),
                         key=lambda x: (x[0] != "base", x[0]))
    models, vanilla_devset, bfrs_devset = zip(*sorted_data)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Adjust bar positions and width
    x = np.arange(len(models))
    width = 0.35

    # Plot bars for Dev Set
    vanilla_bars = ax.bar(x - width/2, vanilla_devset, width, label='No Prompt Optimization', color='skyblue')
    bfrs_bars = ax.bar(x + width/2, bfrs_devset, width, label='BootstrapFewShotRandomSearch', color='lightgreen')

    # Add value labels on top of each bar
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)

    add_labels(vanilla_bars)
    add_labels(bfrs_bars)

    # Customize the plot
    ax.set_ylabel('Dev Set Scores')
    ax.set_title('Model Performance (Synthetic Dev Set; N=1000)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Find the highest devset score and its corresponding model
    highest_devset_score = max(bfrs_devset)
    highest_score_model = models[bfrs_devset.index(highest_devset_score)]

    print(f"Highest Dev Set Score: {highest_devset_score:.1f}, Model: {highest_score_model}")
