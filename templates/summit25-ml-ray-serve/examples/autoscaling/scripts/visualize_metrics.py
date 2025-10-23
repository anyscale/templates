#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys


def load_locust_stats(csv_path):
    """Load Locust statistics CSV file."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def load_ray_metrics(json_path):
    """Load Ray Serve metrics from JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {json_path}")
        return None
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def plot_requests_per_second(df, output_dir):
    """Plot requests per second over time."""
    if 'Timestamp' not in df.columns or 'Requests/s' not in df.columns:
        print("Warning: Required columns not found for RPS plot")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df['Requests/s'], linewidth=2)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Requests per Second', fontsize=12)
    plt.title('Throughput Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'rps_over_time.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_latency(df, output_dir):
    """Plot latency percentiles over time."""
    plt.figure(figsize=(12, 6))

    if '50%' in df.columns:
        plt.plot(df['Timestamp'], df['50%'], label='P50', linewidth=2)
    if '90%' in df.columns:
        plt.plot(df['Timestamp'], df['90%'], label='P90', linewidth=2)
    if '99%' in df.columns:
        plt.plot(df['Timestamp'], df['99%'], label='P99', linewidth=2)

    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.title('Response Time Percentiles Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'latency_over_time.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_user_count(df, output_dir):
    """Plot number of users over time."""
    if 'Timestamp' not in df.columns or 'User Count' not in df.columns:
        print("Warning: Required columns not found for user count plot")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df['User Count'], linewidth=2, color='green')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Number of Users', fontsize=12)
    plt.title('Active Users Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'users_over_time.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_replica_count(metrics_data, output_dir):
    """Plot replica count over time from Ray Serve metrics."""
    if not metrics_data or 'replicas' not in metrics_data:
        print("Warning: Replica metrics not found")
        return

    timestamps = [entry['timestamp'] for entry in metrics_data['replicas']]
    replica_counts = [entry['count'] for entry in metrics_data['replicas']]

    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, replica_counts, linewidth=2, marker='o', markersize=4)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Number of Replicas', fontsize=12)
    plt.title('Replica Count Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / 'replicas_over_time.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_dashboard(stats_df, output_dir):
    """Create a dashboard with multiple plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Load Test Summary Dashboard', fontsize=16, fontweight='bold')

    if 'Timestamp' in stats_df.columns and 'User Count' in stats_df.columns:
        axes[0, 0].plot(stats_df['Timestamp'], stats_df['User Count'], linewidth=2, color='green')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Users')
        axes[0, 0].set_title('Active Users')
        axes[0, 0].grid(True, alpha=0.3)

    if 'Timestamp' in stats_df.columns and 'Requests/s' in stats_df.columns:
        axes[0, 1].plot(stats_df['Timestamp'], stats_df['Requests/s'], linewidth=2, color='blue')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('RPS')
        axes[0, 1].set_title('Requests per Second')
        axes[0, 1].grid(True, alpha=0.3)

    if '50%' in stats_df.columns:
        axes[1, 0].plot(stats_df['Timestamp'], stats_df['50%'], linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Latency (ms)')
        axes[1, 0].set_title('P50 Latency')
        axes[1, 0].grid(True, alpha=0.3)

    if 'Total Request Count' in stats_df.columns:
        axes[1, 1].plot(stats_df['Timestamp'], stats_df['Total Request Count'], linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Total Requests')
        axes[1, 1].set_title('Cumulative Requests')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'dashboard.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize Ray Serve autoscaling metrics and load test results'
    )
    parser.add_argument(
        '--stats-history',
        type=str,
        help='Path to Locust stats history CSV file (*_stats_history.csv)'
    )
    parser.add_argument(
        '--ray-metrics',
        type=str,
        help='Path to Ray Serve metrics JSON file (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./plots',
        help='Directory to save plots (default: ./plots)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("====================================")
    print("Ray Serve Autoscaling Visualization")
    print("====================================")
    print(f"Output directory: {output_dir}")
    print()

    if args.stats_history:
        print("Loading Locust statistics...")
        stats_df = load_locust_stats(args.stats_history)

        if stats_df is not None:
            print(f"Loaded {len(stats_df)} data points")
            print("Generating plots...")

            plot_user_count(stats_df, output_dir)
            plot_requests_per_second(stats_df, output_dir)
            plot_latency(stats_df, output_dir)
            create_summary_dashboard(stats_df, output_dir)
    else:
        print("No stats history file provided. Use --stats-history to specify.")

    if args.ray_metrics:
        print("\nLoading Ray Serve metrics...")
        metrics_data = load_ray_metrics(args.ray_metrics)

        if metrics_data is not None:
            print("Generating replica count plot...")
            plot_replica_count(metrics_data, output_dir)

    print("\n====================================")
    print("Visualization complete!")
    print("====================================")
    print(f"All plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
