import matplotlib.pyplot as plt
import numpy as np

def graph_testset_results(ft_results):
    # Prepare data for the graph
    models = []
    vanilla_testset = []
    bfrs_testset = []

    for model, results in ft_results.items():
        if model == "base":
            models.append("Base Model")
        else:
            models.append(f"Epoch {model.split(':')[1].split('-')[1]}")
        vanilla_testset.append(results['vanilla'].get('testset', None))
        bfrs_testset.append(results['bfrs'].get('testset', None))

    vanilla_testset = [x if x is not None else 0 for x in vanilla_testset]
    bfrs_testset = [x if x is not None else 0 for x in bfrs_testset]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35

    # Plot bars
    vanilla_bars = ax.bar(x - width/2, vanilla_testset, width, label='No Prompt Optimization', color='skyblue')
    bfrs_bars = ax.bar(x + width/2, bfrs_testset, width, label='BootstrapFewShotRandomSearch', color='lightgreen')

    # Add value labels on top of each bar
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)

    add_labels(vanilla_bars)
    add_labels(bfrs_bars)

    # Customize the plot
    ax.set_ylabel('Test Set Scores')
    ax.set_title('Model Performance Comparison (Real Test Set; N=1000)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

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
            models.append("Epoch " + model.split(':')[1].split('-')[1])  # Extract epoch information
        vanilla_devset.append(results['vanilla']['devset'])
        bfrs_devset.append(results['bfrs']['devset'])

    # Sort the data by epoch, keeping "base" at the beginning
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
    ax.set_title('Model Performance Comparison Across Epochs (Synthetic Dev Set; N=1000)')
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
