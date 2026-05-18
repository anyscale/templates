#!/usr/bin/env python3
"""
Simple Python script for demonstrating Anyscale Job submission from workspace.
This script prints a hello message and demonstrates basic job execution.
"""

import ray

def main():
    print("Hello from Anyscale Job!")
    print(f"Job is running on Ray cluster")
    print(f"Available resources: {ray.cluster_resources()}")
    print("Job completed successfully!")

if __name__ == "__main__":
    main()
