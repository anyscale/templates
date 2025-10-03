"""Visualization utilities for ETL optimization templates."""

import matplotlib.pyplot as plt
import numpy as np


def visualize_etl_performance():
    """Visualize ETL pipeline performance metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Processing throughput
    stages = ['Extract', 'Transform', 'Load']
    throughput = [15000, 12000, 18000]  # Records per second
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    bars = axes[0].bar(stages, throughput, color=colors, alpha=0.7)
    axes[0].set_title('ETL Stage Throughput', fontweight='bold')
    axes[0].set_ylabel('Records/Second')
    
    for bar, val in zip(bars, throughput):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{val:,}', ha='center', fontweight='bold')
    
    # 2. Memory usage
    operations = ['Read', 'Transform', 'Aggregate', 'Write']
    memory = [2.5, 4.2, 3.8, 1.9]  # GB
    
    axes[1].barh(operations, memory, color='orange', alpha=0.7)
    axes[1].set_title('Memory Usage by Operation', fontweight='bold')
    axes[1].set_xlabel('Memory (GB)')
    
    # 3. Optimization impact
    configs = ['Baseline', 'Optimized\nBatch Size', 'Optimized\nConcurrency', 'Fully\nOptimized']
    speedup = [1.0, 2.3, 3.1, 4.2]
    
    axes[2].plot(configs, speedup, 'o-', linewidth=2, markersize=10, color='green')
    axes[2].set_title('Optimization Impact', fontweight='bold')
    axes[2].set_ylabel('Speedup Factor')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('etl_performance.png', dpi=150, bbox_inches='tight')
    print("ETL performance visualization saved")
    
    return fig

