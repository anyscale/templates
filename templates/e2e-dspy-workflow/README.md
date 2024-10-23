# Building an Efficient LLM Pipeline with DSPy and Anyscale

Time to complete: 1.5 hours


In this guide, we'll show how you can build an efficient pipeline covering synthetic data generation, data processing, fine-tuning, evaluation and serving with DSPy and Anyscale.

**What is DSPy?**

DSPy is a framework for building and optimizing programs involving language models.

It allows you to define a complex pipeline with simple Python code, and then optimize the pipeline for better performance on whatever your task is.

See the [DSPy Documentation](https://dspy-docs.vercel.app/intro/) for more information.

## Why DSPy and Anyscale?
DSPy simplifies the complex workflow of:
- Data Collection/Labeling
- Fine-tuning
- Prompt Optimization
- Evaluation
  
We'll use Anyscale for scalable infrastructure for training and serving/deploying models.

## Scenario: Cost-Effective Customer Support Query Classification

Consider an example of classification with limited labelled data. The specific dataset we'll be working with is the [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) dataset, which consists of customer support queries for a bank. We'll simulate a scenario with low labelled data: let's say we have a dataset has limited labeled data (100 queries) and 4,000 unlabeled customer queries. We'll build a solution that leverages DSPy on Anyscale to distill knowledge from a 70B model into a more cost-effective 1B model, making it practical for production deployment.

- DSPy enables easy creation of a pipeline for knowledge distillation from a 70B model to a 1B model in a low data environment
- Anyscale's infrastructure supports efficient fine-tuning and deployment
- Result: A cost-effective, accurate classification system for 25 categories

## Table of Contents

1. Setup
2. Data Processing and Labeling
3. Model Fine-tuning
4. Evaluation and Optimization
5. Production Deployment
6. Future Improvements

## Set up

We will be running everything on A100-80GB GPUs. This is not necessary, especially for running a 1B model. You can edit the serving configuration files used throughout the notebook to use different GPUs if you do not have access to A100s.

We use Anyscale's Auto-select worker node feature to launch and manage child nodes that are running our LLM. You can also set your own compute configuration to autoscale different types of GPUs at different ranges.


```python
%load_ext autoreload
%autoreload 2
```


```python
import importlib.util

if importlib.util.find_spec("dspy") is None:
    print("Installing dspy")
    !pip install git+https://github.com/stanfordnlp/dspy.git@main

else:
    print("dspy is already installed")

!pip install matplotlib python-dotenv
```

    dspy is already installed
    Requirement already satisfied: matplotlib in /home/ray/anaconda3/lib/python3.9/site-packages (3.9.2)
    Requirement already satisfied: python-dotenv in /home/ray/anaconda3/lib/python3.9/site-packages (1.0.1)
    Requirement already satisfied: contourpy>=1.0.1 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (1.3.0)
    Requirement already satisfied: cycler>=0.10 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (4.54.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (1.4.7)
    Requirement already satisfied: numpy>=1.23 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (1.23.5)
    Requirement already satisfied: packaging>=20.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (23.0)
    Requirement already satisfied: pillow>=8 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (9.2.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (3.1.4)
    Requirement already satisfied: python-dateutil>=2.7 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (2.8.2)
    Requirement already satisfied: importlib-resources>=3.2.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (6.4.5)
    Requirement already satisfied: zipp>=3.1.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from importlib-resources>=3.2.0->matplotlib) (3.19.2)
    Requirement already satisfied: six>=1.5 in /home/ray/anaconda3/lib/python3.9/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)



```python
import dspy
dspy.settings.configure(experimental=True)

import ujson

from src import set_dspy_cache_location
set_dspy_cache_location("/home/ray/default/dspy/cache")
```

In order to run this notebook, you need to have the following environment variables set:
- HF_HOME
- HF_TOKEN
- (optional) WANDB_API_KEY

You can get a HF_TOKEN [here](https://huggingface.co/settings/tokens). You will need to request access to the Meta-Llama-3.1-70B-Instruct model and the Llama-3.2-1B-Instruct model.

You can get a WANDB_API_KEY [here](https://wandb.ai/authorize).

<b style="background-color: yellow;">&nbsp;🔄 REPLACE&nbsp;</b>: Set the HF_TOKEN, HF_HOME, and optionally the WANDB_API_KEY environment variables in the notebook.


```python
import os
import ray 
# By default, the cache directory used by HuggingFace is in the home directory -`/home/ray`
# We'll use `/mnt/local_storage` here for downloading large model weight files

os.environ["HF_HOME"] = "/mnt/local_storage/huggingface"
os.environ["HF_TOKEN"] = "Add your HF token here"
# Optional: Add your Wandb token
# os.environ["WANDB_API_KEY"] = "12345"

# You can also use a .env file to store your HF_TOKEN and WANDB_API_KEY
# from dotenv import load_dotenv
# load_dotenv(override=True)
```




    True



We will make use of a random number generator in this notebook to ensure that our notebook is reproducible.


```python
from src import set_random_seed
rng = set_random_seed()
```


```python
from src import check_env_vars

# Check if env vars are set correctly
check_env_vars()
```


```python
from src import init_ray
init_ray()
```

    2024-10-23 00:37:29,643	INFO worker.py:1601 -- Connecting to existing Ray cluster at address: 10.0.0.28:6379...
    2024-10-23 00:37:29,654	INFO worker.py:1777 -- Connected to Ray cluster. View the dashboard at https://session-6frbgpfbuzs27m75xhz3c8vnde.i.anyscaleuserdata.com 
    2024-10-23 00:37:29,686	INFO packaging.py:531 -- Creating a file package for local directory '/home/ray/anaconda3/lib/python3.9/site-packages/dspy'.
    2024-10-23 00:37:29,730	INFO packaging.py:359 -- Pushing file package 'gcs://_ray_pkg_3456a89b1fcd9948.zip' (1.17MiB) to Ray cluster...
    2024-10-23 00:37:29,744	INFO packaging.py:372 -- Successfully pushed file package 'gcs://_ray_pkg_3456a89b1fcd9948.zip'.
    2024-10-23 00:37:29,759	INFO packaging.py:531 -- Creating a file package for local directory '/home/ray/anaconda3/lib/python3.9/site-packages/dsp'.
    2024-10-23 00:37:29,779	INFO packaging.py:359 -- Pushing file package 'gcs://_ray_pkg_eac1031112d62b99.zip' (0.55MiB) to Ray cluster...
    2024-10-23 00:37:29,787	INFO packaging.py:372 -- Successfully pushed file package 'gcs://_ray_pkg_eac1031112d62b99.zip'.
    2024-10-23 00:37:29,790	INFO packaging.py:359 -- Pushing file package 'gcs://_ray_pkg_c62d21a49b226e4a53a17f9d7aa3bed4b56a5efb.zip' (0.83MiB) to Ray cluster...
    2024-10-23 00:37:29,798	INFO packaging.py:372 -- Successfully pushed file package 'gcs://_ray_pkg_c62d21a49b226e4a53a17f9d7aa3bed4b56a5efb.zip'.


    (autoscaler +1m53s) Tip: use `ray status` to view detailed cluster status. To disable these messages, set RAY_SCHEDULER_EVENTS=0.
    (autoscaler +1m53s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +3m32s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launched 1 instances.
    (autoscaler +17m6s) [autoscaler] Downscaling node g-f69c56902ddd40001 (node IP: 10.0.0.57) due to node idle termination.
    (autoscaler +31m41s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +32m27s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launched 1 instances.
    (autoscaler +40m35s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 3 node(s).
    (autoscaler +40m36s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:3;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +40m39s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +40m39s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +40m39s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +40m40s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +40m41s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +40m42s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +40m43s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +40m44s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +40m46s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +40m46s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +40m48s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +40m48s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +40m50s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +40m50s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +40m51s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 3 node(s).
    (autoscaler +40m53s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:3;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +40m54s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +40m56s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +40m57s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +40m57s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +40m58s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +40m58s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +40m58s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +40m59s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m0s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +41m1s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +41m2s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m2s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m5s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +41m6s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m7s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +41m9s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m10s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 3 node(s).
    (autoscaler +41m10s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:3;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m13s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +41m13s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +41m14s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m14s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m15s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +41m16s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m18s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +41m19s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-c] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:true]: googleapi: Error 400: Invalid value for field 'resource.instanceProperties.machineType': 'a2-ultragpu-1g'. Machine type with name 'a2-ultragpu-1g' does not exist in zone 'us-east5-c'.
    (autoscaler +41m19s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-c] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 400: Invalid value for field 'resource.instanceProperties.machineType': 'a2-ultragpu-1g'. Machine type with name 'a2-ultragpu-1g' does not exist in zone 'us-east5-c'.
    (autoscaler +41m22s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 3 node(s).
    (autoscaler +41m23s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:3;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m25s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +41m25s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m27s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +41m27s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +41m28s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m28s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m31s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +41m32s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m33s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +41m33s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m34s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +41m34s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +41m34s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m35s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m38s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +41m39s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m40s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 3 node(s).
    (autoscaler +41m41s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:3;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m44s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +41m44s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +41m44s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m44s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m46s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +41m48s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m49s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +41m49s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m50s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +41m50s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +41m50s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m51s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m52s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +41m54s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m57s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 3 node(s).
    (autoscaler +41m58s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:3;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +41m59s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +41m59s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +42m0s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m0s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m2s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +42m4s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m6s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +42m7s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m8s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +42m8s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +42m9s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m9s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m12s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +42m12s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m15s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +42m16s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m17s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +42m17s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +42m17s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m18s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m19s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +42m21s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m23s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 3 node(s).
    (autoscaler +42m24s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:3;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m25s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +42m25s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m26s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +42m26s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +42m27s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m27s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m30s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +42m30s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m31s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 3 node(s).
    (autoscaler +42m33s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:3;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m34s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +42m34s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +42m35s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m35s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m37s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +42m39s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m40s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +42m41s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m42s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +42m42s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +42m42s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m42s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m44s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +42m44s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m47s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +42m48s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m49s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +42m49s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +42m50s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m51s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m51s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +42m53s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m54s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 3 node(s).
    (autoscaler +42m56s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:3;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m57s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +42m58s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +42m59s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +42m59s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +43m0s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m0s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m2s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +43m2s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m5s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +43m5s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +43m5s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m6s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m8s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +43m8s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m10s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +43m12s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m13s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +43m13s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +43m14s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m14s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m16s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +43m16s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m18s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +43m19s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m22s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 3 node(s).
    (autoscaler +43m23s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:3;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m25s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +43m25s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +43m25s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m25s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m27s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +43m29s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m31s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +43m31s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-c] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:true]: googleapi: Error 400: Invalid value for field 'resource.instanceProperties.machineType': 'a2-ultragpu-1g'. Machine type with name 'a2-ultragpu-1g' does not exist in zone 'us-east5-c'.
    (autoscaler +43m32s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-c] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 400: Invalid value for field 'resource.instanceProperties.machineType': 'a2-ultragpu-1g'. Machine type with name 'a2-ultragpu-1g' does not exist in zone 'us-east5-c'.
    (autoscaler +43m33s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +43m34s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m35s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +43m35s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +43m36s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m36s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m39s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +43m40s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m41s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 6 node(s).
    (autoscaler +43m43s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:6;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m44s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 3 node(s).
    (autoscaler +43m46s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:3;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m48s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +43m48s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +43m49s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m49s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m50s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +43m51s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m53s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +43m53s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +43m53s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m55s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m57s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 3 node(s).
    (autoscaler +43m58s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:3;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +43m59s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +44m1s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +44m1s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +44m2s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +44m5s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +44m5s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +44m55s) [autoscaler] Downscaling node g-754bab5ac3b0b0002 (node IP: 10.0.0.46) due to node idle termination.
    (autoscaler +44m55s) [autoscaler] Downscaling node g-754bab5ac3b0b0001 (node IP: 10.0.0.41) due to node idle termination.
    (autoscaler +44m55s) [autoscaler] Downscaling node g-754bab5ac3b0b0003 (node IP: 10.0.0.45) due to node idle termination.
    (autoscaler +44m57s) [autoscaler] Cluster resized to {12 CPU, 1 GPU}.
    (autoscaler +1h1m41s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h1m41s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h1m43s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h1m43s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h1m44s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h1m45s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h1m48s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h1m48s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h1m48s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h1m48s) [autoscaler] Cluster upscaled to {48 CPU, 4 GPU}.
    (autoscaler +1h1m48s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h1m50s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h1m51s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h1m52s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h1m53s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h1m53s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h1m53s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h1m57s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h1m57s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h1m57s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h1m57s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h1m59s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h1m59s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m0s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h2m0s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h2m1s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m1s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m4s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h2m4s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h2m5s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m5s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m7s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h2m7s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m9s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h2m9s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h2m10s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m10s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m11s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h2m11s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h2m13s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m13s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m15s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h2m16s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m16s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h2m16s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h2m17s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m17s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m18s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h2m18s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h2m19s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m19s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m23s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h2m24s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m24s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h2m25s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h2m26s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m26s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m28s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h2m28s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h2m28s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m30s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m31s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h2m32s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m34s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h2m34s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h2m35s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m35s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m37s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h2m37s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h2m37s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m39s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m40s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h2m41s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m41s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h2m41s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h2m42s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m43s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m44s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h2m44s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h2m45s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m45s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m49s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h2m49s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m50s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h2m50s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h2m50s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m50s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m52s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h2m52s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h2m53s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m53s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m56s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h2m57s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h2m58s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h2m58s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m0s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m0s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m2s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h3m2s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m2s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m2s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m5s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h3m6s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m6s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h3m7s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m8s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m8s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m9s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h3m9s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m11s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m11s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m14s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h3m14s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m16s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m16s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h3m16s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m16s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m17s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h3m17s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m18s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m18s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m21s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h3m22s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m23s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h3m23s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m25s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m25s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m27s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h3m27s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m27s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m27s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m30s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h3m30s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m32s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m32s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h3m32s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m33s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m33s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h3m33s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m34s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m35s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m36s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h3m39s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m41s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h3m41s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m41s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m41s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m42s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h3m42s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m42s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m42s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m43s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h3m44s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m47s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h3m47s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m48s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m48s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m50s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h3m50s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m51s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m51s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m53s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h3m53s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m55s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h3m55s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m56s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m57s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m58s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h3m58s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h3m59s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h3m59s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m0s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h4m2s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m4s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m4s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h4m5s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m5s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m6s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h4m6s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m7s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m7s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m9s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h4m9s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m11s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m11s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h4m11s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m13s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m15s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h4m15s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m15s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m15s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m16s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h4m17s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m18s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h4m18s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m19s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m21s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m22s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h4m22s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m24s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m24s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m25s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h4m25s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m27s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h4m27s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m28s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m28s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m31s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h4m31s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m31s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m31s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m32s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h4m32s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m35s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h4m35s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m35s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m35s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m38s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h4m38s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m39s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m40s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m41s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h4m41s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m43s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m43s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h4m43s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m45s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m46s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m46s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h4m47s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m47s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m49s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h4m50s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m51s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h4m51s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m52s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m52s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m53s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m53s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h4m53s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m55s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m57s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h4m57s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m58s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h4m58s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h4m59s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h4m59s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m0s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m0s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h5m1s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m1s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m3s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h5m4s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m5s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h5m5s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m6s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m6s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m8s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h5m8s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m8s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m9s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m12s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h5m13s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m14s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h5m14s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m15s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m15s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m16s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h5m16s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m17s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m17s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m19s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h5m21s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m22s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h5m22s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m24s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m24s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m25s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m25s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h5m25s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m26s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m28s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h5m28s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m30s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h5m31s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m31s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m31s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m32s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m33s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h5m33s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m33s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m35s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h5m35s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m36s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h5m36s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m38s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m38s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m41s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h5m41s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m41s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m41s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m43s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h5m44s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m45s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h5m45s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m47s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m47s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m48s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h5m49s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m49s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m49s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m52s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h5m52s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m52s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h5m52s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m53s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m53s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m57s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h5m57s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h5m58s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m58s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h5m59s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h5m59s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m1s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h6m1s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m2s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m2s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m5s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h6m5s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m5s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m5s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m7s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h6m8s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m10s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m10s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h6m10s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m10s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m11s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h6m11s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m13s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m14s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m16s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h6m16s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m17s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m17s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h6m18s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m18s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m21s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h6m21s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m21s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m22s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m22s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h6m24s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m25s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m25s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h6m26s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m27s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m28s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h6m28s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m30s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m30s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m32s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h6m33s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m36s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m36s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h6m36s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m36s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m38s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m38s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h6m38s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m38s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m40s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h6m40s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m41s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h6m41s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m42s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m42s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m44s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h6m44s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m47s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m47s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m48s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h6m48s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m50s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h6m50s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m50s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m50s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m52s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h6m52s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m52s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m52s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m54s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h6m56s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m57s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 2 node(s).
    (autoscaler +1h6m57s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h6m58s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:2;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h6m58s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h7m0s) [autoscaler] [4xA100-80GB:48CPU-680GB] Upscaling 1 node(s).
    (autoscaler +1h7m0s) [autoscaler] [1xA100-80GB:12CPU-170GB] Upscaling 1 node(s).
    (autoscaler +1h7m1s) [autoscaler] [4xA100-80GB:48CPU-680GB|a2-ultragpu-4g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-4g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h7m1s) [autoscaler] [1xA100-80GB:12CPU-170GB|a2-ultragpu-1g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-1g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h7m5s) [autoscaler] [8xA100-80GB:96CPU-1360GB] Upscaling 1 node(s).
    (autoscaler +1h7m5s) [autoscaler] [8xA100-80GB:96CPU-1360GB|a2-ultragpu-8g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-8g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h7m7s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +1h7m7s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h7m8s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +1h7m8s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h7m9s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +1h7m11s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h7m11s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +1h7m13s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h7m15s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +1h7m15s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h7m16s) [autoscaler] [2xA100-80GB:24CPU-340GB] Upscaling 1 node(s).
    (autoscaler +1h7m17s) [autoscaler] [2xA100-80GB:24CPU-340GB|a2-ultragpu-2g] [us-east5-b] [on-demand] Launching instances failed: NewInstances[a2-ultragpu-2g;num:1;all:false]: googleapi: Error 403: Quota 'NVIDIA_A100_80GB_GPUS' exceeded. Limit: 16.0 in region us-east5.
    (autoscaler +1h8m4s) [autoscaler] Downscaling node g-93baa4121b79f0001 (node IP: 10.0.0.53) due to node idle termination.
    (autoscaler +1h8m5s) [autoscaler] Cluster resized to {36 CPU, 3 GPU}.
    (autoscaler +1h11m36s) [autoscaler] Downscaling node g-73bd411113f590001 (node IP: 10.0.0.22) due to node idle termination.
    (autoscaler +1h11m37s) [autoscaler] Cluster resized to {12 CPU, 1 GPU}.


# Dataset Preparation

We will be using the `PolyAI/banking77` dataset for this tutorial. We use the built in dspy DataLoader to load the dataset from Huggingface as a list of dspy.Example objects.


```python
# Prepare the dataset
from src import load_data_from_huggingface, convert_int_label_to_string
full_trainset, full_testset = load_data_from_huggingface()

full_trainset_processed, full_testset_processed = convert_int_label_to_string(full_trainset, full_testset)
```

The dataset is originally called "banking77" because there are 77 labels. We will be reducing this to the top 25 most frequent labels.


```python
from src import filter_to_top_n_labels
full_trainset_filtered, full_testset_filtered, top_25_labels = filter_to_top_n_labels(full_trainset_processed, full_testset_processed, n=25)
labels_in_use = top_25_labels
print(f"Dataset filtered to top 25 labels. New sizes:")
print(f"Training set size: {len(full_trainset_filtered)}; Test set size: {len(full_testset_filtered)}")
print(f"Top 25 labels: {', '.join(str(label) for label in top_25_labels)}")
print(f"Example training set: {full_trainset_filtered[0]}")
print(f"Example test set: {full_testset_filtered[0]}")
```

    Dataset filtered to top 25 labels. New sizes:
    Training set size: 4171; Test set size: 1000
    Top 25 labels: card_payment_fee_charged, direct_debit_payment_not_recognised, balance_not_updated_after_cheque_or_cash_deposit, wrong_amount_of_cash_received, cash_withdrawal_charge, transaction_charged_twice, declined_cash_withdrawal, transfer_fee_charged, balance_not_updated_after_bank_transfer, transfer_not_received_by_recipient, request_refund, card_payment_not_recognised, card_payment_wrong_exchange_rate, extra_charge_on_statement, wrong_exchange_rate_for_cash_withdrawal, refund_not_showing_up, reverted_card_payment, cash_withdrawal_not_recognised, activate_my_card, pending_card_payment, cancel_transfer, beneficiary_not_allowed, card_arrival, declined_card_payment, pending_top_up
    Example training set: Example({'label': 'card_arrival', 'text': 'I am still waiting on my card?'}) (input_keys={'text'})
    Example test set: Example({'label': 'card_arrival', 'text': 'How do I locate my card?'}) (input_keys={'text'})


Now we will shuffle our training set and split it into a training and labeled set.

The scenario we are emulating is that we only trust our 70B model to do a good job at classifying the queries, but don't want to serve a 70B parameter model for classification. We are saying that we have 4K (length of the training set) unlabeled examples we can then label using an oracle model, and then distill the knowledge from the oracle model into our 1B model.


```python
from src import adjusted_exact_match, delete_labels, NUM_THREADS


shuffled_trainset = [d for d in full_trainset_filtered]
rng.shuffle(shuffled_trainset)
ft_trainset = shuffled_trainset
# For realism of this scenario, we are going to delete all our labels except for our test set
ft_trainset_to_label = delete_labels(ft_trainset)

testset = full_testset_filtered

common_kwargs = dict(metric=adjusted_exact_match, num_threads=NUM_THREADS, display_progress=True, max_errors=10000)
# evaluate_testset is our "eval harness for this program"
evaluate_testset = dspy.Evaluate(devset=testset, **common_kwargs)
```

# Implementing a Simple Chain of Thought Program in DSPy

## Defining the Signature

At the heart of our DSPy program is the `Signature` class. This class serves as a blueprint, outlining the inputs and outputs of our language model task. Here's how we structure it:

1. **Docstring for Context**: We utilize the docstring to provide context to the LLM. In this case, we're passing our fixed set of 25 labels directly in the docstring. This approach is ideal when dealing with a static set of options.

2. **Input Field**: We define an `intent` field as the input to our program. This will contain the natural language query we want to classify.

3. **Output Field**: The `label` field represents our desired output - the classified intent.

Both input and output fields are accompanied by concise descriptions, just to help the LLM understand the task.

By structuring our program this way, we utilize DSPy's capabilities to create a clear, modular design that's both powerful and easy to maintain. 


```python
class IntentClassification(dspy.Signature):
    """As a part of a banking issue traiging system, classify the intent of a natural language query into one of the 25 labels.
    The intent should exactly match one of the following:
    ['card_payment_fee_charged', 'direct_debit_payment_not_recognised', 'balance_not_updated_after_cheque_or_cash_deposit', 'wrong_amount_of_cash_received', 'cash_withdrawal_charge', 'transaction_charged_twice', 'declined_cash_withdrawal', 'transfer_fee_charged', 'balance_not_updated_after_bank_transfer', 'transfer_not_received_by_recipient', 'request_refund', 'card_payment_not_recognised', 'card_payment_wrong_exchange_rate', 'extra_charge_on_statement', 'wrong_exchange_rate_for_cash_withdrawal', 'refund_not_showing_up', 'reverted_card_payment', 'cash_withdrawal_not_recognised', 'activate_my_card', 'pending_card_payment', 'cancel_transfer', 'beneficiary_not_allowed', 'card_arrival', 'declined_card_payment', 'pending_top_up']
    """

    intent = dspy.InputField(desc="Intent of the query")
    label = dspy.OutputField(desc="Type of the intent; Should just be one of the 25 labels with no other text")
```

For the module, we create a dspy.Module class that contains the Chain of Thought predictor using the signature we defined above.
We also pass in the valid labels to the module.

Inside the forward method, we pass the text to the predictor, do a little cleaning, and return the prediction.


```python
class IntentClassificationModule(dspy.Module):
    def __init__(self, labels_in_use):
        self.intent_classifier = dspy.ChainOfThought(IntentClassification)
        self.valid_labels = set(labels_in_use)

    def forward(self, text):
        prediction = self.intent_classifier(intent=text)
        sanitized_prediction = dspy.Prediction(label=prediction.label.lower().strip().replace(" ", "_"), reasoning=prediction.reasoning)
        return sanitized_prediction
```

Lastly, we set up some the vanilla program we will use throughout the notebook.


```python
from src import MODEL_PARAMETERS, LOCAL_API_PARAMETERS
vanilla_program = IntentClassificationModule(labels_in_use)
```

# Deploying and Utilizing a 70B Language Model

This section outlines the process of deploying and utilizing a 70B parameter language model for data gathering and training. Key steps include:

1. Infrastructure: Leverage Anyscale's [RayLLM](https://docs.anyscale.com/llms/serving/intro) with "Auto-select worker nodes" for dynamically allocating GPUs.
2. Configuration: Use a pre-generated serve config file (created via `rayllm gen-config`) to configure the RayLLM instance.

Below we show the contents of the serve config and its corresponding model config file.

`serve_70B.yaml` is a file that we created using rayllm gen-config for the purposes of this notebook. A serve config needs to have access to your HF_TOKEN so that it can correctly download the model from hugging face.


```python
from src import print_serve_and_model_config, update_serve_config_hf_token

# First we update the config to contain your actual HF_TOKEN to get access to the Meta-LLama huggingface repositories
update_serve_config_hf_token("serve_70B.yaml")

print_serve_and_model_config("serve_70B.yaml")
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">serve_config:
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'applications'</span>: <span style="font-weight: bold">[</span>
        <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'args'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'llm_configs'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'./model_config/meta-llama--Meta-Llama-3_1-70B-Instruct.yaml'</span><span style="font-weight: bold">]}</span>,
            <span style="color: #008000; text-decoration-color: #008000">'import_path'</span>: <span style="color: #008000; text-decoration-color: #008000">'rayllm:app'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'name'</span>: <span style="color: #008000; text-decoration-color: #008000">'llm-endpoint'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'route_prefix'</span>: <span style="color: #008000; text-decoration-color: #008000">'/'</span>
        <span style="font-weight: bold">}</span>
    <span style="font-weight: bold">]</span>,
    <span style="color: #008000; text-decoration-color: #008000">'query_auth_token_enabled'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>
<span style="font-weight: bold">}</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">==================================================
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">model_config:
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'accelerator_type'</span>: <span style="color: #008000; text-decoration-color: #008000">'A100-80G'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'deployment_config'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'autoscaling_config'</span>: <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'initial_replicas'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
            <span style="color: #008000; text-decoration-color: #008000">'max_replicas'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
            <span style="color: #008000; text-decoration-color: #008000">'min_replicas'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span>,
            <span style="color: #008000; text-decoration-color: #008000">'target_ongoing_requests'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128</span>
        <span style="font-weight: bold">}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'max_ongoing_requests'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">300</span>
    <span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'engine_kwargs'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'enable_chunked_prefill'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>,
        <span style="color: #008000; text-decoration-color: #008000">'max_num_batched_tokens'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8192</span>,
        <span style="color: #008000; text-decoration-color: #008000">'max_num_seqs'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">256</span>,
        <span style="color: #008000; text-decoration-color: #008000">'tokenizer_pool_extra_config'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'runtime_env'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'pip'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span><span style="font-weight: bold">}}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'tokenizer_pool_size'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
        <span style="color: #008000; text-decoration-color: #008000">'trust_remote_code'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>
    <span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'generation_config'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'prompt_format'</span>: <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'assistant'</span>: <span style="color: #008000; text-decoration-color: #008000">'&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;\n\n{instruction}&lt;|eot_id|&gt;'</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'bos'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #008000; text-decoration-color: #008000">'&lt;|begin_of_text|&gt;'</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'default_system_message'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #008000; text-decoration-color: #008000">''</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'system'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #008000; text-decoration-color: #008000">'&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;\n\n{instruction}&lt;|eot_id|&gt;'</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'system_in_user'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'trailing_assistant'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #008000; text-decoration-color: #008000">'&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;\n\n'</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'user'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #008000; text-decoration-color: #008000">'&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt;\n\n{instruction}&lt;|eot_id|&gt;'</span>
        <span style="font-weight: bold">}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'stopping_sequences'</span>: <span style="font-weight: bold">[]</span>,
        <span style="color: #008000; text-decoration-color: #008000">'stopping_tokens'</span>: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128001</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128009</span><span style="font-weight: bold">]</span>
    <span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'input_modality'</span>: <span style="color: #008000; text-decoration-color: #008000">'text'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'json_mode'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'enabled'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'llm_engine'</span>: <span style="color: #008000; text-decoration-color: #008000">'VLLMEngine'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'lora_config'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span>,
    <span style="color: #008000; text-decoration-color: #008000">'max_request_context_length'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8192</span>,
    <span style="color: #008000; text-decoration-color: #008000">'model_loading_config'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'model_id'</span>: <span style="color: #008000; text-decoration-color: #008000">'meta-llama/Meta-Llama-3.1-70B-Instruct'</span>,
        <span style="color: #008000; text-decoration-color: #008000">'model_source'</span>: <span style="color: #008000; text-decoration-color: #008000">'meta-llama/Meta-Llama-3.1-70B-Instruct'</span>
    <span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'runtime_env'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'env_vars'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'HUGGING_FACE_HUB_TOKEN'</span>: <span style="color: #008000; text-decoration-color: #008000">'REDACTED'</span><span style="font-weight: bold">}}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'tensor_parallelism'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'degree'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span><span style="font-weight: bold">}</span>
<span style="font-weight: bold">}</span>
</pre>



The `serve run [file]` command is used in order to launch any ray serve deployments. For RayLLM specifically, that will allow us to query `localhost:8000` as an OpenAI compatible API with whatever model is served.


```python
!serve run --non-blocking serve_70B.yaml
```

    2024-10-23 00:38:50,089	INFO scripts.py:489 -- Running config file: 'serve_70B.yaml'.
    2024-10-23 00:38:50,418	INFO worker.py:1601 -- Connecting to existing Ray cluster at address: 10.0.0.28:6379...
    2024-10-23 00:38:50,426	INFO worker.py:1777 -- Connected to Ray cluster. View the dashboard at https://session-6frbgpfbuzs27m75xhz3c8vnde.i.anyscaleuserdata.com 
    2024-10-23 00:38:50,430	INFO packaging.py:359 -- Pushing file package 'gcs://_ray_pkg_fd90c1638cd32bfadef22fb05da8f982b8bee1ed.zip' (0.83MiB) to Ray cluster...
    2024-10-23 00:38:50,439	INFO packaging.py:372 -- Successfully pushed file package 'gcs://_ray_pkg_fd90c1638cd32bfadef22fb05da8f982b8bee1ed.zip'.
    (ProxyActor pid=143530) INFO 2024-10-23 00:38:54,251 proxy 10.0.0.28 proxy.py:1235 - Proxy starting on node 76e2eab0a833e02208d948f67f54806e64e4d0b26f1a49507548855a (HTTP port: 8000).
    INFO 2024-10-23 00:38:54,280 serve 143394 api.py:277 - Started Serve in namespace "serve".
    2024-10-23 00:38:54,288	SUCC scripts.py:540 -- Submitted deploy config successfully.
    (ServeController pid=143461) INFO 2024-10-23 00:38:54,283 controller 143461 application_state.py:881 - Deploying new app 'llm-endpoint'.
    (ServeController pid=143461) INFO 2024-10-23 00:38:54,284 controller 143461 application_state.py:457 - Importing and building app 'llm-endpoint'.


Now we instantiate a `dspy.LM` object pointing at our RayLLM deployment. This will allow us to query it inside of DSPy programs.


```python
print("Model Parameters:", MODEL_PARAMETERS)
print("Local API Parameters:", LOCAL_API_PARAMETERS)
llama_70b = dspy.LM(model="openai/meta-llama/Meta-Llama-3.1-70B-Instruct", **MODEL_PARAMETERS, **LOCAL_API_PARAMETERS)
```

    Model Parameters: {'max_tokens': 1000, 'temperature': 0}
    Local API Parameters: {'api_base': 'http://localhost:8000/v1', 'api_key': 'fake-key-doesnt-matter'}


Below is a sanity check to make sure that our program is running properly.

All it is doing is passing a single request to the DSPy program.

The request will wait until the server starts up before it finishes.

When the server starts up, it may need to recruit a worker node and also download the 70B model weights. Once that returns, our server is good to go, and we can use it like any other endpoint.

You can expect the cell below to take around 8-10 minutes to run, as it waits for a worker node and to download the weights.


```python
from src import sanity_check_program

sanity_check_program(llama_70b, vanilla_program, ft_trainset[0])
```

    Program input: Example({'text': "I was just giving the wrong amount by the ATM. How do I request cash back at this point? The app is showing the amount that I actually requested even though that's not what I got."}) (input_keys={'text'})
    Program output label: wrong_amount_of_cash_received


### Bootstrap Data

In this section, we bootstrap and prepare data for fine-tuning.

Recall that our dataset only contains 100 labelled examples. Using DSPy, we will now "bootstrap" our training dataset with these labelled examples and generate synthetic labels using the Llama 70B model.

As a part of data validation ("Is this a correct label?"), we will use a simple metric: we returns True if the prediction is in the desired set of labels, else we return False. Entries for which the metric is False are filtered out.

Finally, we convert the filtered dataset into the OpenAI conversational format for use in fine-tuning.


```python
from dspy.teleprompt.finetune_teleprompter import bootstrap_data, convert_to_module_level_message_data
from src import NUM_THREADS, get_valid_label_metric_fn

with dspy.context(lm=llama_70b):
    bootstrap_data_kwargs = {
        "program": vanilla_program, 
        "dataset": ft_trainset_to_label, 
        "num_threads": NUM_THREADS, 
        "max_errors": 10000, 
        "metric": get_valid_label_metric_fn(labels_in_use)
    }
    collected_data = bootstrap_data(**bootstrap_data_kwargs)
    # Make sure to only include the labels we are actively using or that arent hallucinated by the oracle
    collected_data_filtered = [x for x in collected_data if x["prediction"]["label"] in labels_in_use]

dataset = convert_to_module_level_message_data(collected_data_filtered, program=vanilla_program, exclude_demos=True)

dataset_formatted = [{"messages": item} for item in dataset]
```

    Average Metric: 4162 / 4171  (99.8): 100%|██████████| 4171/4171 [06:24<00:00, 10.85it/s]  



```python
import rich

rich.print(dataset_formatted[0])
print("Length of dataset:\t", len(dataset))
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'messages'</span>: <span style="font-weight: bold">[</span>
        <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'role'</span>: <span style="color: #008000; text-decoration-color: #008000">'system'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'content'</span>: <span style="color: #008000; text-decoration-color: #008000">"Your input fields are:\n1. `intent` (str): Intent of the query\n\nYour output fields </span>
<span style="color: #008000; text-decoration-color: #008000">are:\n1. `reasoning` (str): ${produce the output fields}. We ...\n2. `label` (str): Type of the intent; Should just</span>
<span style="color: #008000; text-decoration-color: #008000">be one of the 25 labels with no other text\n\nAll interactions will be structured in the following way, with the </span>
<span style="color: #008000; text-decoration-color: #008000">appropriate values filled in.\n\n[[ ## intent ## ]]\n{intent}\n\n[[ ## reasoning ## ]]\n{reasoning}\n\n[[ ## label </span>
<span style="color: #008000; text-decoration-color: #008000">## ]]\n{label}\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is: \n        As a part of</span>
<span style="color: #008000; text-decoration-color: #008000">a banking issue traiging system, classify the intent of a natural language query into one of the 25 labels.\n      </span>
<span style="color: #008000; text-decoration-color: #008000">The intent should exactly match one of the following:\n        ['card_payment_fee_charged', </span>
<span style="color: #008000; text-decoration-color: #008000">'direct_debit_payment_not_recognised', 'balance_not_updated_after_cheque_or_cash_deposit', </span>
<span style="color: #008000; text-decoration-color: #008000">'wrong_amount_of_cash_received', 'cash_withdrawal_charge', 'transaction_charged_twice', 'declined_cash_withdrawal',</span>
<span style="color: #008000; text-decoration-color: #008000">'transfer_fee_charged', 'balance_not_updated_after_bank_transfer', 'transfer_not_received_by_recipient', </span>
<span style="color: #008000; text-decoration-color: #008000">'request_refund', 'card_payment_not_recognised', 'card_payment_wrong_exchange_rate', 'extra_charge_on_statement', </span>
<span style="color: #008000; text-decoration-color: #008000">'wrong_exchange_rate_for_cash_withdrawal', 'refund_not_showing_up', 'reverted_card_payment', </span>
<span style="color: #008000; text-decoration-color: #008000">'cash_withdrawal_not_recognised', 'activate_my_card', 'pending_card_payment', 'cancel_transfer', </span>
<span style="color: #008000; text-decoration-color: #008000">'beneficiary_not_allowed', 'card_arrival', 'declined_card_payment', 'pending_top_up']"</span>
        <span style="font-weight: bold">}</span>,
        <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'role'</span>: <span style="color: #008000; text-decoration-color: #008000">'user'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'content'</span>: <span style="color: #008000; text-decoration-color: #008000">"[[ ## intent ## ]]\nI was just giving the wrong amount by the ATM. How do I request cash </span>
<span style="color: #008000; text-decoration-color: #008000">back at this point? The app is showing the amount that I actually requested even though that's not what I </span>
<span style="color: #008000; text-decoration-color: #008000">got.\n\nRespond with the corresponding output fields, starting with the field `reasoning`, then `label`, and then </span>
<span style="color: #008000; text-decoration-color: #008000">ending with the marker for `completed`."</span>
        <span style="font-weight: bold">}</span>,
        <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'role'</span>: <span style="color: #008000; text-decoration-color: #008000">'assistant'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'content'</span>: <span style="color: #008000; text-decoration-color: #008000">'[[ ## reasoning ## ]]\nThe user is reporting that they received the wrong amount of cash </span>
<span style="color: #008000; text-decoration-color: #008000">from an ATM, and the amount shown on the app is different from what they actually received. They are asking how to </span>
<span style="color: #008000; text-decoration-color: #008000">request cash back in this situation.\n\n[[ ## label ## ]]\nwrong_amount_of_cash_received\n\n[[ ## completed ## ]]'</span>
        <span style="font-weight: bold">}</span>
    <span style="font-weight: bold">]</span>
<span style="font-weight: bold">}</span>
</pre>



    Length of dataset:	 4162



```python
# Now we are done with the 70B model, so we can kill the server
!serve shutdown -y
```

    2024-10-23 00:50:14,214	WARN scripts.py:132 -- The `RAY_AGENT_ADDRESS` env var has been deprecated in favor of the `RAY_DASHBOARD_ADDRESS` env var. The `RAY_AGENT_ADDRESS` is ignored.
    2024-10-23 00:50:16,880	SUCC scripts.py:747 -- Sent shutdown request; applications will be deleted asynchronously.


# Fine-tuning

We will use Anyscale's [LLMForge](https://docs.anyscale.com/llms/finetuning/intro) to fine-tune the 1B model.

To fine-tune with LLMForge, we will make use of DSPy's native integration with Anyscale. We can simply pass the desired Anyscale job configuration and DSPy will handle the rest.

Be sure to checkout the fine-tuning documentation for the latest on how to use our [API](https://docs.anyscale.com/llms/finetuning/intro) and additional [capabilities](https://www.anyscale.com/library/llmforge).


We will be starting out by fine-tuning using LoRA.


```python
from dspy.clients.lm import TrainingMethod

train_data = dataset_formatted
method = TrainingMethod.SFT

# There are a few important files that we are providing for you in this template in order to finetune + serve using LLMForge and RayLLM
job_config_path = "configs/job.yaml"
llmforge_config_path = "configs/training/lora/llama-3-8b.yaml"
# Optional: DSPY supports passing a serve config for serving the fine-tuned model. This particular config is generated
serve_config_path = "serve_1B.yaml"
```

When you want to launch a finetuning run, you run the command `llmforge anyscale finetune [config file]`. You can do this locally if you are on a GPU head node, but because you generally run on a CPU head node, we will launch it as a job on the Anyscale platform.

In order to interface properly with DSPy, there are new abstractions that were introduced in order to lower the complexity of combining a finetuning platform and DSPy.

DSPy will verify and format your data, submit your finetuning job using the **Job Config** and **LLMForge Config** files, then update your **Serve Config** so that your serve config knows where the finetuned weights are stored.

There are 4 important files here, and we will go through them one by one:
1. **Job Config**: job config points to a yaml file that will launch the LLMForge finetuning run as a job on the Anyscale platform.
    All that a job config contains are the image to use, any environment variables, and an entrypoint that is the command to run, in this case `llmforge anyscale finetune [config file]`. [Job Config Documentation](https://docs.anyscale.com/reference/job-api/#jobconfig).


```python
import yaml
rich.print(yaml.safe_load(open(job_config_path)))
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'name'</span>: <span style="color: #008000; text-decoration-color: #008000">'dspy-llmforge-fine-tuning-job'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'entrypoint'</span>: <span style="color: #008000; text-decoration-color: #008000">'llmforge anyscale finetune configs/training/lora/llama-3-8b.yaml'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'working_dir'</span>: <span style="color: #008000; text-decoration-color: #008000">'.'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'image_uri'</span>: <span style="color: #008000; text-decoration-color: #008000">'localhost:5555/anyscale/llm-forge:0.5.7'</span>
<span style="font-weight: bold">}</span>
</pre>



2. **LLMForge Config**:
The LLMForge config contains the relevant details for how to actually finetune the model. This contains every parameter you might expect for doing the actual finetuning: model, training data, epochs to run, batch size, LoRA parameters, etc.



```python
rich.print(yaml.safe_load(open(llmforge_config_path)))
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'context_length'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2048</span>,
    <span style="color: #008000; text-decoration-color: #008000">'num_devices'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span>,
    <span style="color: #008000; text-decoration-color: #008000">'num_epochs'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span>,
    <span style="color: #008000; text-decoration-color: #008000">'train_batch_size_per_device'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>,
    <span style="color: #008000; text-decoration-color: #008000">'eval_batch_size_per_device'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,
    <span style="color: #008000; text-decoration-color: #008000">'learning_rate'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3e-05</span>,
    <span style="color: #008000; text-decoration-color: #008000">'padding'</span>: <span style="color: #008000; text-decoration-color: #008000">'longest'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'num_checkpoints_to_keep'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span>,
    <span style="color: #008000; text-decoration-color: #008000">'dataset_size_scaling_factor'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10000</span>,
    <span style="color: #008000; text-decoration-color: #008000">'output_dir'</span>: <span style="color: #008000; text-decoration-color: #008000">'/mnt/local_storage'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'deepspeed'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'config_path'</span>: <span style="color: #008000; text-decoration-color: #008000">'configs/deepspeed/zero_3.json'</span><span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'flash_attention_2'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>,
    <span style="color: #008000; text-decoration-color: #008000">'worker_resources'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'accelerator_type:A100-80G'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.001</span><span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'generation_config'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'prompt_format'</span>: <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'system'</span>: <span style="color: #008000; text-decoration-color: #008000">'&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;\n\n{instruction}&lt;|eot_id|&gt;'</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'user'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #008000; text-decoration-color: #008000">'&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt;\n{instruction}&lt;|eot_id|&gt;'</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'assistant'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #008000; text-decoration-color: #008000">'&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;\n\n{instruction}&lt;|eot_id|&gt;'</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'trailing_assistant'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #008000; text-decoration-color: #008000">'&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;\n\n'</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'bos'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #008000; text-decoration-color: #008000">'&lt;|begin_of_text|&gt;'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'system_in_user'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
            <span style="color: #008000; text-decoration-color: #008000">'default_system_message'</span>: <span style="color: #008000; text-decoration-color: #008000">''</span>
        <span style="font-weight: bold">}</span>
    <span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'lora_config'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'r'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span>,
        <span style="color: #008000; text-decoration-color: #008000">'lora_alpha'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,
        <span style="color: #008000; text-decoration-color: #008000">'lora_dropout'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0.05</span>,
        <span style="color: #008000; text-decoration-color: #008000">'target_modules'</span>: <span style="font-weight: bold">[</span>
            <span style="color: #008000; text-decoration-color: #008000">'q_proj'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'v_proj'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'k_proj'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'o_proj'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'gate_proj'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'up_proj'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'down_proj'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'embed_tokens'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'lm_head'</span>
        <span style="font-weight: bold">]</span>,
        <span style="color: #008000; text-decoration-color: #008000">'task_type'</span>: <span style="color: #008000; text-decoration-color: #008000">'CAUSAL_LM'</span>,
        <span style="color: #008000; text-decoration-color: #008000">'modules_to_save'</span>: <span style="font-weight: bold">[]</span>,
        <span style="color: #008000; text-decoration-color: #008000">'bias'</span>: <span style="color: #008000; text-decoration-color: #008000">'none'</span>,
        <span style="color: #008000; text-decoration-color: #008000">'fan_in_fan_out'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
        <span style="color: #008000; text-decoration-color: #008000">'init_lora_weights'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>
    <span style="font-weight: bold">}</span>
<span style="font-weight: bold">}</span>
</pre>



3. **Serve Config**: This is a RayLLM serving configuration file that points to the model config file we want to serve.
For what Job Config is to the LLMForge config, Serve Config is to the model config.

Serve config just points to the model config file that we want to serve.

4. **Model Config**: This is the file that contains the details of the model we want to finetune. It contains parameters like where to read in LoRA weights from, what model to serve, what resources to use to serve the model, etc.


```python
print_serve_and_model_config(serve_config_path)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">serve_config:
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'applications'</span>: <span style="font-weight: bold">[</span>
        <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'args'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'llm_configs'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'./model_config/meta-llama--Llama-3_2-1B-Instruct.yaml'</span><span style="font-weight: bold">]}</span>,
            <span style="color: #008000; text-decoration-color: #008000">'import_path'</span>: <span style="color: #008000; text-decoration-color: #008000">'rayllm:app'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'name'</span>: <span style="color: #008000; text-decoration-color: #008000">'llm-endpoint'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'route_prefix'</span>: <span style="color: #008000; text-decoration-color: #008000">'/'</span>
        <span style="font-weight: bold">}</span>
    <span style="font-weight: bold">]</span>,
    <span style="color: #008000; text-decoration-color: #008000">'query_auth_token_enabled'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>
<span style="font-weight: bold">}</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">==================================================
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">model_config:
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'accelerator_type'</span>: <span style="color: #008000; text-decoration-color: #008000">'A100-80G'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'deployment_config'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'autoscaling_config'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'target_ongoing_requests'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">}</span>, <span style="color: #008000; text-decoration-color: #008000">'max_ongoing_requests'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span><span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'engine_kwargs'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'enable_chunked_prefill'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>,
        <span style="color: #008000; text-decoration-color: #008000">'enable_lora'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>,
        <span style="color: #008000; text-decoration-color: #008000">'max_lora_rank'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span>,
        <span style="color: #008000; text-decoration-color: #008000">'max_loras'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span>,
        <span style="color: #008000; text-decoration-color: #008000">'max_num_batched_tokens'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8192</span>,
        <span style="color: #008000; text-decoration-color: #008000">'max_num_seqs'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">64</span>,
        <span style="color: #008000; text-decoration-color: #008000">'tokenizer_pool_extra_config'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'runtime_env'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'pip'</span>: <span style="color: #800080; text-decoration-color: #800080; font-style: italic">None</span><span style="font-weight: bold">}}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'tokenizer_pool_size'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span>,
        <span style="color: #008000; text-decoration-color: #008000">'trust_remote_code'</span>: <span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">True</span>
    <span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'generation_config'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'prompt_format'</span>: <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'assistant'</span>: <span style="color: #008000; text-decoration-color: #008000">'&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;\n\n{instruction}&lt;|eot_id|&gt;'</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'bos'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #008000; text-decoration-color: #008000">'&lt;|begin_of_text|&gt;'</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'default_system_message'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #008000; text-decoration-color: #008000">''</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'system'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #008000; text-decoration-color: #008000">'&lt;|start_header_id|&gt;system&lt;|end_header_id|&gt;\n\n{instruction}&lt;|eot_id|&gt;'</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'system_in_user'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'trailing_assistant'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #008000; text-decoration-color: #008000">'&lt;|start_header_id|&gt;assistant&lt;|end_header_id|&gt;\n\n'</span><span style="color: #000000; text-decoration-color: #000000">,</span>
<span style="color: #000000; text-decoration-color: #000000">            </span><span style="color: #008000; text-decoration-color: #008000">'user'</span><span style="color: #000000; text-decoration-color: #000000">: </span><span style="color: #008000; text-decoration-color: #008000">'&lt;|start_header_id|&gt;user&lt;|end_header_id|&gt;\n\n{instruction}&lt;|eot_id|&gt;'</span>
        <span style="font-weight: bold">}</span>,
        <span style="color: #008000; text-decoration-color: #008000">'stopping_sequences'</span>: <span style="font-weight: bold">[]</span>,
        <span style="color: #008000; text-decoration-color: #008000">'stopping_tokens'</span>: <span style="font-weight: bold">[</span><span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128001</span>, <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">128009</span><span style="font-weight: bold">]</span>
    <span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'input_modality'</span>: <span style="color: #008000; text-decoration-color: #008000">'text'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'json_mode'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'enabled'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span><span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'llm_engine'</span>: <span style="color: #008000; text-decoration-color: #008000">'VLLMEngine'</span>,
    <span style="color: #008000; text-decoration-color: #008000">'lora_config'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'dynamic_lora_loading_path'</span>: <span style="color: #008000; text-decoration-color: #008000">'none'</span>, <span style="color: #008000; text-decoration-color: #008000">'max_num_adapters_per_replica'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">32</span><span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'max_request_context_length'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8192</span>,
    <span style="color: #008000; text-decoration-color: #008000">'model_loading_config'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'model_id'</span>: <span style="color: #008000; text-decoration-color: #008000">'meta-llama/Llama-3.2-1B-Instruct'</span>,
        <span style="color: #008000; text-decoration-color: #008000">'model_source'</span>: <span style="color: #008000; text-decoration-color: #008000">'meta-llama/Llama-3.2-1B-Instruct'</span>
    <span style="font-weight: bold">}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'runtime_env'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'env_vars'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'HUGGING_FACE_HUB_TOKEN'</span>: <span style="color: #008000; text-decoration-color: #008000">'REDACTED'</span><span style="font-weight: bold">}}</span>,
    <span style="color: #008000; text-decoration-color: #008000">'tensor_parallelism'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'degree'</span>: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span><span style="font-weight: bold">}</span>
<span style="font-weight: bold">}</span>
</pre>



DSPy has an abstraction around LLMForge finetuning in order to allow you to pass in your configuration files and not have to worry about the details of how to interface with LLMForge.

What we do is we create a `dspy.LM` object that points to the model we want to finetune.

Part of the new behavior added in order to support better finetuning flows, is that `dspy.LM` objects now have a `.finetune()` method that will take in the train data, training arguments, a method, and a provider. Each provider can individually handle the finetuning process differently, but will return a `dspy.FinetuningJob` object that you can use to access your finetuned model.

The end of this process is the `dspy.FinetuningJob` object, which contains a `.result()` method that will return the finetuned `dspy.LM` object.

Note that `dspy.LM` objects do not manage the weights themselves, and can be considered essentially immutable. The returned `dspy.LM` object has a `.model` attribute that points to the finetuned model, in this example, `meta-llama/Llama-3.2-1B-Instruct:name:suffix`, where name and suffix come from LLMForge.

When you call `LM.finetune()`, DSPy will:
1. Verify your data is formatted correctly
2. Upload your data to the cloud using the Anyscale Datasets API
3. Update your LLMForge config to point to the data that has been uploaded
4. Launch a LLMForge job using the provided job config, LLMForge config, and the data you uploaded
5. Update the serve config to point to the finetuned model
6. Wait for the finetuning job to complete
7. Return the finetuned `dspy.LM` object

You can see what DSPy is doing under the hood [here](https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/anyscale.py)


```python
finetuneable_lm = dspy.LM(model="meta-llama/Llama-3.2-1B-Instruct", **MODEL_PARAMETERS, **LOCAL_API_PARAMETERS)

try:
    finetuning_kwargs = {
        "train_data": train_data, 
        "train_kwargs": {
            "job_config_path": job_config_path, 
            "llmforge_config_path": llmforge_config_path, 
            "serve_config_path": serve_config_path
        }, 
        "train_method": method, 
        "provider": "anyscale"
    }

    finetuning_job = finetuneable_lm.finetune(**finetuning_kwargs)
    finetuned_llama = finetuning_job.result()
    print("Finetuned Model ID:", finetuned_llama.model)
except Exception as e:
    print(e)
```


    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">Upload complete!</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
</pre>



    (anyscale +13m39.4s) Uploading local dir '.' to cloud storage.
    (anyscale +13m41.8s) Including workspace-managed pip dependencies.
    (anyscale +13m42.5s) Job 'dspy-llmforge-fine-tuning-job' submitted, ID: 'prodjob_r4nwu2mlijl1ag7heb755cjw8x'.
    (anyscale +13m42.5s) View the job in the UI: https://console.anyscale.com/jobs/prodjob_r4nwu2mlijl1ag7heb755cjw8x
    (anyscale +13m42.7s) Waiting for job 'prodjob_r4nwu2mlijl1ag7heb755cjw8x' to reach target state SUCCEEDED, currently in state: STARTING
    (anyscale +16m7.5s) Job 'prodjob_r4nwu2mlijl1ag7heb755cjw8x' transitioned from STARTING to RUNNING
    (anyscale +29m42.0s) Job 'prodjob_r4nwu2mlijl1ag7heb755cjw8x' transitioned from RUNNING to SUCCEEDED
    (anyscale +29m42.0s) Job 'prodjob_r4nwu2mlijl1ag7heb755cjw8x' reached target state, exiting


    Finetuned Model ID: openai/meta-llama/Llama-3.2-1B-Instruct:isaac:ngffk


# Evaluation

## Performance comparisons

**Datasets:**
- Synthetic Devset - we care about how well a model does on the dataset, as it is our closest thing to ground truth in our scenario with a lot of unlabeled data.
- Real Testset - in this case, we do actually have a test set, so we can see how well a model does on a representative dataset of unseen user queries.

**LM Comparison:**
We will compare the finetuned model to the non-finetuned model, both with and without prompt optimization. We expect to see that the finetuned model does better on the synthetic devset compared to the testset, as it was trained on exactly that data.
- 1B Non-finetuned
- 1B Non-finetuned + Prompt Optimization
- 1B Finetuned (last checkpoint)
- 1B Finetuned (last checkpoint) + Prompt Optimization

Note that we do not provide an eval set when finetuning, as the eval loss of a checkpoint isn't necessarily predictive of the downstream performance of the program. We use the last checkpoint by default.


```python
print(finetuned_llama.model)
```

    openai/meta-llama/Llama-3.2-1B-Instruct:isaac:ngffk


We will run a local RayLLM instance that serves the model.

Provided with this template is are two files, `serve_1B.yaml` and `model_configs/meta-llama--Llama-3_2-1B-Instruct.yaml`. 

The first file, `serve_1B.yaml`, contains the serve configuration to load the model with RayLLM.

The second file, `model_configs/meta-llama--Llama-3_2-1B-Instruct.yaml`, contains the necessary configurations to run the 1B model.


```python
from src import update_serve_config_hf_token

update_serve_config_hf_token("serve_1B.yaml")
```

Run this command to start the 1B RayLLM server:


```python
!serve run --non-blocking serve_1B.yaml
```

    2024-10-23 01:08:39,710	INFO scripts.py:489 -- Running config file: 'serve_1B.yaml'.
    2024-10-23 01:08:40,027	INFO worker.py:1601 -- Connecting to existing Ray cluster at address: 10.0.0.28:6379...
    2024-10-23 01:08:40,035	INFO worker.py:1777 -- Connected to Ray cluster. View the dashboard at https://session-6frbgpfbuzs27m75xhz3c8vnde.i.anyscaleuserdata.com 
    2024-10-23 01:08:40,039	INFO packaging.py:359 -- Pushing file package 'gcs://_ray_pkg_65f953c7794f5c3fb3b91d46cb1f7888057a961d.zip' (0.85MiB) to Ray cluster...
    2024-10-23 01:08:40,048	INFO packaging.py:372 -- Successfully pushed file package 'gcs://_ray_pkg_65f953c7794f5c3fb3b91d46cb1f7888057a961d.zip'.
    INFO 2024-10-23 01:08:43,956 serve 157622 api.py:277 - Started Serve in namespace "serve".
    2024-10-23 01:08:43,965	SUCC scripts.py:540 -- Submitted deploy config successfully.
    (ServeController pid=157685) INFO 2024-10-23 01:08:43,959 controller 157685 application_state.py:881 - Deploying new app 'llm-endpoint'.
    (ServeController pid=157685) INFO 2024-10-23 01:08:43,960 controller 157685 application_state.py:457 - Importing and building app 'llm-endpoint'.
    (ProxyActor pid=157763) INFO 2024-10-23 01:08:43,926 proxy 10.0.0.28 proxy.py:1235 - Proxy starting on node 76e2eab0a833e02208d948f67f54806e64e4d0b26f1a49507548855a (HTTP port: 8000).



```python
from src import MODEL_PARAMETERS, LOCAL_API_PARAMETERS

non_ft_llama = dspy.LM(model="openai/meta-llama/Llama-3.2-1B-Instruct", **MODEL_PARAMETERS, **LOCAL_API_PARAMETERS)

all_llamas = {"base": non_ft_llama, finetuned_llama.model: finetuned_llama}
```


```python
# Sanity check that the finetuned models are working

try:
    sanity_check_program(non_ft_llama, vanilla_program, ft_trainset[0])
except ValueError as e:
    # Sometimes the 1B model isn't capable of correctly outputting the label before prompt optimization, so we can just ignore this error.
    print("Non fine-tuned model returned invalid output out and errored out with", e)

try:
    sanity_check_program(finetuned_llama, vanilla_program, ft_trainset[0])
except ValueError as e:
    print("Fine-tuned model returned invalid output out and errored out with", e)
```

    Program input: Example({'text': "I was just giving the wrong amount by the ATM. How do I request cash back at this point? The app is showing the amount that I actually requested even though that's not what I got."}) (input_keys={'text'})


    Non fine-tuned model returned invalid output out and errored out with Expected dict_keys(['reasoning', 'label']) but got dict_keys([])
    Program input: Example({'text': "I was just giving the wrong amount by the ATM. How do I request cash back at this point? The app is showing the amount that I actually requested even though that's not what I got."}) (input_keys={'text'})
    Program output label: wrong_amount_of_cash_received


## Prompt Optimization with DSPy

We are going to be doing prompt optimization using DSPy's `BootstrapFewShotWithRandomSearch (BFRS)` function.

BootstrapFewShotWithRandomSearch will:
- Collect a set of chains of thought from the model
- Use these examples that lead to a correct prediction to "bootstrap" the program
- See which set of examples lead to the most correct predictions across your evaluation metric
- Continue this process for a set number of iterations, using the best performing programs to bootstrap the next iteration
- Return the best program

Let's go over what the hyperparameters mean:
- **max_bootstrapped_demos**: DSPy will "bootstrap" the program by collecting examples at each step that are successful and reusing those in the pipeline. This means that it will automatically collect and add chains of thought to the pipeline.
- **max_labeled_demos**: DSPy will also insert some labeled demonstrations from the training set. These would be unmodified examples from the training set that are just using the given answer.
- **num_candidate_programs**: This is the number of candidate programs that the optimizer will generate. The actual number of programs that are created is this plus three, as DSPy will also try a program with no examples, a program with just the labeled demonstrations, and a bootstrapped program with the first few examples.

Learn more about the BFRS optimizer [here](https://dspy-docs.vercel.app/deep-dive/optimizers/bootstrap-fewshot/).


```python
from src import bootstrap_fewshot_random_search_parameters, adjusted_exact_match

print("Parameters:")
for k, v in bootstrap_fewshot_random_search_parameters.items():
    print(f"{k}: {v}")
```

    Parameters:
    max_bootstrapped_demos: 3
    max_labeled_demos: 3
    num_candidate_programs: 6



```python
from src import split_into_devset_and_optimizer_sets

def collected_data_to_example(data):
    return dspy.Example(text=data["example"]["text"], label=data["prediction"]["label"]).with_inputs("text")

collected_data_examples = [collected_data_to_example(x) for x in collected_data_filtered]

devset_synthetic, ft_optimizer_trainset, ft_optimizer_devset = split_into_devset_and_optimizer_sets(collected_data_examples, dev_size=1000, optimizer_num_val=300)
print("Lengths:")
print("Synthetic Devset:\t", len(devset_synthetic))
print("Optimizer Trainset:\t", len(ft_optimizer_trainset))
print("Optimizer Devset:\t", len(ft_optimizer_devset))
print("Example from synthetic devset:")
print(devset_synthetic[0])
```

    Lengths:
    Synthetic Devset:	 1000
    Optimizer Trainset:	 2862
    Optimizer Devset:	 300
    Example from synthetic devset:
    Example({'text': "I was just giving the wrong amount by the ATM. How do I request cash back at this point? The app is showing the amount that I actually requested even though that's not what I got.", 'label': 'wrong_amount_of_cash_received'}) (input_keys={'text'})


Now we will evaluate our finetuned model and the base model, prompt optimize them, and evaluate them on the synthetic devset.

Note that there is a `%%capture` below. This is to suppress the output of the evaluation and prompt optimization because it is quite long and will slow down the notebook. We will graph the results in the cell after. You can remove the tag in order to see the output.

You can expect this to take around 15-20 minutes to run.

If you remove the `%%capture` tag, you will see that the output is quite long and full of errors. This is because the base LLama 1B model is not capable of formatting the outputs correctly in the way that DSPy expects, so it will throw errors. Our finetuned model is much better at this, and throws significantly less errors.


```python
%%capture
from src import evaluate_and_prompt_optimize

evaluation_kwargs = {
    "models": all_llamas,
    "module_class": IntentClassificationModule,
    "optimizer_trainset": ft_optimizer_trainset,
    "optimizer_valset": ft_optimizer_devset,
    "devset": devset_synthetic,
    "metric": adjusted_exact_match,
    "labels_in_use": labels_in_use
}

ft_results = evaluate_and_prompt_optimize(**evaluation_kwargs)
```


```python
print(ft_results)
```

    {'base': {'vanilla': {'devset': 0.3}, 'bfrs': {'devset': 23.5}}, 'openai/meta-llama/Llama-3.2-1B-Instruct:isaac:ngffk': {'vanilla': {'devset': 56.3}, 'bfrs': {'devset': 56.3}}}



```python
from src import graph_devset_results, graph_testset_results

graph_devset_results(ft_results)
```


    
![png](README_files/README_63_0.png)
    


    Highest Dev Set Score: 56.3, Model: Fine-tuned


We see that the our finetuned model significantly outperforms the base model on the synthetic devset, and prompt optimization isn't enough to bring 1B all the way up to FT performance.

We will now evaluate both the finetuned and base models on the real test set to see if we have improved performance. We will use the prompt optimized versions of the models that we created using the synthetic devset as our in context examples.

This should take around 10 minutes to run.


```python
%%capture
# Now we need to evaluate the test set
from src import run_testset_evaluation

testset_evaluation_kwargs = {
    "ft_results": ft_results,
    "all_llamas": all_llamas,
    "labels_in_use": labels_in_use,
    "testset": testset,
    "metric": adjusted_exact_match,
    "module_class": IntentClassificationModule
}

ft_results_testset, (best_program_path, best_model, best_score) = run_testset_evaluation(**testset_evaluation_kwargs)
```


```python
graph_testset_results(ft_results_testset)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /home/ray/default/templates/templates/e2e-dspy-workflow/README.ipynb Cell 67 line 1
    ----> <a href='vscode-notebook-cell://vscode-session-6frbgpfbuzs27m75xhz3c8vnde.i.anyscaleuserdata.com/home/ray/default/templates/templates/e2e-dspy-workflow/README.ipynb#Y124sdnNjb2RlLXJlbW90ZQ%3D%3D?line=0'>1</a> graph_testset_results(ft_results_testset)


    NameError: name 'graph_testset_results' is not defined



```python
print(f"Best testset result: \n{best_model} with score: {best_score}")
```

    Best testset result: 
    openai/meta-llama/Llama-3.2-1B-Instruct:isaac:ngffk with score: 53.5


We see that on the true test set, the prompt optimized pipeline using the finetuned model outperforms the prompt optimized pipeline using the non-finetuned model by approximately 12% in overall dataset accuracy. This is a relative performance improvement of almost 40%!

This is one small example of where DSPy finetuning can be useful in terms of increasing task specific performance and helping you to get the most out of your LLM.

This pipeline is very simple, but you can imagine that data collection would be much harder for longer pipelines, and DSPy can optimize it just as easily.

# Serving

The typical usecase for DSPy is for optimizing prompts and weights programmatically. DSPy allows you to define a complex pipeline with different components like a retriever and one or more LLMs. Often, we're interested in taking the same system we optimized during training to inference. 

To deploy, you can serve the optimized DSPy program directly: This is the simplest option to take your program to production. Since DSPy simply relies on a deployed inference endpoint for LLM calls, we can use it in conjunction with optimized serving libraries like RayLLM. We can leverage Ray Serve with our DSPy pipeline being our custom business logic while serving.

NOTE: As of DSPy 2.5, there are scalability limitations for high throughput scenarios with DSPy. DSPy compiled programs currently use threading for handling multiple queries in parallel, which might not scale as well as a native `async` implementation. A native `async` implementation is in the immediate roadmap for DSPy. If this is a concern, you can always try to stitch together the saved program from DSPy in native Python code. 

## Serving a DSPy Pipeline as a Multi-App Serve Application

We can break down our program into two distinct parts: 1) Fine-tuned LLM served behind an OpenAI compatible endpoint and 2) The DSPy program (our business logic tying all components together)

These two different parts can be served as two separate applications with different deployment configurations. There are two ways this can be done:
- Separate Ray Serve deployments: In this case, you would be managing two separate services for your DSPy program. One scenario where this is helpful is if you expect changes/updates to just one component (say the DSPy program) happening at a different cadence to the other. 
- Single multi-app deployment: This is the simpler way for managing your DSPy program in production by deploying one service with two applications. This is recommended when all your Ray Serve logic lies in one repository.


In this guide, we will deploy our DSPy program as a single Ray Serve deployment. 

First, let's save some important state for the compiled program into a JSON file for use in serving


```python
import json 

# Note: there are some caveats to how `best_program_path` can look like. All files for use in serving should be in the working directory passed to the serve config. 
# By default the working directory  is the current directory
my_state = {"best_model": best_model, "best_program_path": best_program_path, "labels_in_use": labels_in_use}
with open("configs/deploy_params.json", "w") as f:
    json.dump(my_state)
```

For the purpose of serving, it is recommended to place all the application logic in a python script. We've provided a script `deploy.py` which contains the DSPy application logic. 


```python
# Print out deploy.py 

from rich import print_json
from rich.syntax import Syntax
from rich import print as rprint 


def pretty_print_py(file_path):
    with open(file_path) as f:
        code = f.read()
    syntax = Syntax(code, "python", theme="monokai")
    rprint(syntax)

pretty_print_py("deploy.py")
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">import</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> dspy</span><span style="background-color: #272822">                                                                                                        </span>
<span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">from</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> ray </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">import</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> serve</span><span style="background-color: #272822">                                                                                              </span>
<span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">from</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> fastapi </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">import</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> FastAPI</span><span style="background-color: #272822">                                                                                        </span>
<span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">from</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> src </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">import</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> MODEL_PARAMETERS</span><span style="background-color: #272822">                                                                                   </span>
<span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">import</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> json</span><span style="background-color: #272822">                                                                                                        </span>
<span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">from</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> starlette.requests </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">import</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> Request</span><span style="background-color: #272822">                                                                             </span>
<span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">from</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> urllib.parse </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">import</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> urlparse</span><span style="background-color: #272822">                                                                                  </span>
<span style="background-color: #272822">                                                                                                                   </span>
<span style="background-color: #272822">                                                                                                                   </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">app </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> FastAPI()</span><span style="background-color: #272822">                                                                                                    </span>
<span style="background-color: #272822">                                                                                                                   </span>
<span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">def</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> </span><span style="color: #a6e22e; text-decoration-color: #a6e22e; background-color: #272822">read_params</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">(file_path: str):</span><span style="background-color: #272822">                                                                                   </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">    </span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"""Simple helper for reading dspy program parameters"""</span><span style="background-color: #272822">                                                        </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">    </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">with</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> open(file_path, </span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"r"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">) </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">as</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> f:</span><span style="background-color: #272822">                                                                                </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        params </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> json</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">load(f)</span><span style="background-color: #272822">                                                                                      </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">    </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">return</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> params</span><span style="background-color: #272822">                                                                                                  </span>
<span style="background-color: #272822">                                                                                                                   </span>
<span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">class</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> </span><span style="color: #a6e22e; text-decoration-color: #a6e22e; background-color: #272822">IntentClassification</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">(dspy</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">Signature):</span><span style="background-color: #272822">                                                                        </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">    </span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"""As a part of a banking issue traiging system, classify the intent of a natural language query into one of th</span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">    intent </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> dspy</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">InputField(desc</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"Intent of the query"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">)</span><span style="background-color: #272822">                                                           </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">    label </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> dspy</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">OutputField(desc</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"Type of the intent; Should just be one of the 25 labels with no other text"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">)</span><span style="background-color: #272822">    </span>
<span style="background-color: #272822">                                                                                                                   </span>
<span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">class</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> </span><span style="color: #a6e22e; text-decoration-color: #a6e22e; background-color: #272822">IntentClassificationModule</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">(dspy</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">Module):</span><span style="background-color: #272822">                                                                     </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">    </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">def</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> </span><span style="color: #a6e22e; text-decoration-color: #a6e22e; background-color: #272822">__init__</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">(self, labels_in_use):</span><span style="background-color: #272822">                                                                             </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        self</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">intent_classifier </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> dspy</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">ChainOfThought(IntentClassification)</span><span style="background-color: #272822">                                         </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        self</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">valid_labels </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> set(labels_in_use)</span><span style="background-color: #272822">                                                                     </span>
<span style="background-color: #272822">                                                                                                                   </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">    </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">def</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> </span><span style="color: #a6e22e; text-decoration-color: #a6e22e; background-color: #272822">forward</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">(self, text, </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">**</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">predictor_kwargs):</span><span style="background-color: #272822">                                                                   </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        prediction </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> self</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">intent_classifier(intent</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">text, </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">**</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">predictor_kwargs)</span><span style="background-color: #272822">                                       </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        sanitized_prediction </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> dspy</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">Prediction(label</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">prediction</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">label</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">lower()</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">strip()</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">replace(</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">" "</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">, </span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"_"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">), reasoning</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">if</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> sanitized_prediction</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">label </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">not</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">in</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> self</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">valid_labels:</span><span style="background-color: #272822">                                                    </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">            sanitized_prediction </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> dspy</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">Prediction(label</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"INVALID MODEL OUTPUT"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">)</span><span style="background-color: #272822">                                   </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">return</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> sanitized_prediction</span><span style="background-color: #272822">                                                                                </span>
<span style="background-color: #272822">                                                                                                                   </span>
<span style="color: #a6e22e; text-decoration-color: #a6e22e; background-color: #272822">@serve</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">deployment(</span><span style="background-color: #272822">                                                                                                 </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">    ray_actor_options</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">{</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"num_cpus"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">: </span><span style="color: #ae81ff; text-decoration-color: #ae81ff; background-color: #272822">0.1</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">},</span><span style="background-color: #272822">                                                                           </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">    autoscaling_config</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">dict(min_replicas</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #ae81ff; text-decoration-color: #ae81ff; background-color: #272822">1</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">, max_replicas</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #ae81ff; text-decoration-color: #ae81ff; background-color: #272822">5</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">)</span><span style="background-color: #272822">                                                        </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">)</span><span style="background-color: #272822">                                                                                                                  </span>
<span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">class</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> </span><span style="color: #a6e22e; text-decoration-color: #a6e22e; background-color: #272822">LLMClient</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">:</span><span style="background-color: #272822">                                                                                                   </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">    </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">def</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> </span><span style="color: #a6e22e; text-decoration-color: #a6e22e; background-color: #272822">__init__</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">(self, param_path, serve_args):</span><span style="background-color: #272822">                                                                    </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        params </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> read_params(param_path)</span><span style="background-color: #272822">                                                                           </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        self</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">params </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> params</span><span style="background-color: #272822">                                                                                       </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        base_url </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> serve_args[</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"api_base"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">]</span><span style="background-color: #272822">                                                                          </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        prefix </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> serve_args[</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"route_prefix"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">]</span><span style="background-color: #272822">                                                                        </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        full_url </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> base_url </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">+</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> prefix</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">lstrip(</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">'/'</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">) </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">if</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> len(prefix</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">lstrip(</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">'/'</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">)) </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">else</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> base_url</span><span style="background-color: #272822">                          </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        api_parameters </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> {</span><span style="background-color: #272822">                                                                                         </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">            </span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"api_base"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">: </span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">f"{</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">full_url</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">}/v1"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">,</span><span style="background-color: #272822">                                                                          </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">            </span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"api_key"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">: serve_args[</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"api_key"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">]</span><span style="background-color: #272822">                                                                       </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        }</span><span style="background-color: #272822">                                                                                                          </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        print(</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"API parameters"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">, api_parameters)</span><span style="background-color: #272822">                                                                    </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        self</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">llm </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> dspy</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">LM(model</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"openai/"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">+</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> self</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">params[</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"best_model"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">], </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">**</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">MODEL_PARAMETERS, </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">**</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">api_parameters)</span><span style="background-color: #272822">      </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        self</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">program </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> IntentClassificationModule(params[</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"labels_in_use"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">])</span><span style="background-color: #272822">                                         </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        self</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">program</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">load(params[</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"best_program_path"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">])</span><span style="background-color: #272822">                                                             </span>
<span style="background-color: #272822">                                                                                                                   </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">    </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">async</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">def</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> </span><span style="color: #a6e22e; text-decoration-color: #a6e22e; background-color: #272822">__call__</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">(self, query: str):</span><span style="background-color: #272822">                                                                          </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        </span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"""Answer the given question and provide sources."""</span><span style="background-color: #272822">                                                       </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">with</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> dspy</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">context(experimental</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">True</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">, lm</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">self</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">llm):</span><span style="background-color: #272822">                                                         </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">            retrieval_response </span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">=</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> self</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">program(query)</span><span style="background-color: #272822">                                                               </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">        </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">return</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> retrieval_response</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">label</span><span style="background-color: #272822">                                                                            </span>
<span style="background-color: #272822">                                                                                                                   </span>
<span style="background-color: #272822">                                                                                                                   </span>
<span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">def</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> </span><span style="color: #a6e22e; text-decoration-color: #a6e22e; background-color: #272822">construct_app</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">(args):</span><span style="background-color: #272822">                                                                                           </span>
<span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">    </span><span style="color: #66d9ef; text-decoration-color: #66d9ef; background-color: #272822">return</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822"> LLMClient</span><span style="color: #ff4689; text-decoration-color: #ff4689; background-color: #272822">.</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">bind(args[</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"program_param_path"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">], args[</span><span style="color: #e6db74; text-decoration-color: #e6db74; background-color: #272822">"rayllm_args"</span><span style="color: #f8f8f2; text-decoration-color: #f8f8f2; background-color: #272822">])</span><span style="background-color: #272822">                                         </span>
<span style="background-color: #272822">                                                                                                                   </span>
</pre>



As seen above, we've put our DSPy application logic in the `LLMClient` class. We've also passed in some basic configuration for resources and an autoscaling config. At initialization, each replica of `LLMClient` will run a copy of the DSPy program after reading in the saved parameters from `param_path` (`configs/deploy_params.json` here).

The main entrypoint for the app is `construct_app`. 

Our DSPy code will read in the API parameters for the RayLLM service through `args`.

Let's now go over the Ray Serve config for our DSPy deployment:


```python
import yaml 

deploy_config_path = "local_deploy_dspy.yaml"
with open(deploy_config_path, "r") as f:
    config = yaml.safe_load(f)

rprint(config)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'applications'</span>: <span style="font-weight: bold">[</span>
        <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'name'</span>: <span style="color: #008000; text-decoration-color: #008000">'dspy_client'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'args'</span>: <span style="font-weight: bold">{</span>
                <span style="color: #008000; text-decoration-color: #008000">'program_param_path'</span>: <span style="color: #008000; text-decoration-color: #008000">'configs/deploy_params.json'</span>,
                <span style="color: #008000; text-decoration-color: #008000">'rayllm_args'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'route_prefix'</span>: <span style="color: #008000; text-decoration-color: #008000">'/'</span>, <span style="color: #008000; text-decoration-color: #008000">'api_base'</span>: <span style="color: #008000; text-decoration-color: #008000">'http://localhost:8000'</span>, <span style="color: #008000; text-decoration-color: #008000">'api_key'</span>: <span style="color: #008000; text-decoration-color: #008000">'fake-key'</span><span style="font-weight: bold">}</span>
            <span style="font-weight: bold">}</span>,
            <span style="color: #008000; text-decoration-color: #008000">'import_path'</span>: <span style="color: #008000; text-decoration-color: #008000">'deploy:construct_app'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'route_prefix'</span>: <span style="color: #008000; text-decoration-color: #008000">'/classify_intent'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'runtime_env'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'pip'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'dspy'</span>, <span style="color: #008000; text-decoration-color: #008000">'matplotlib'</span><span style="font-weight: bold">]}</span>
        <span style="font-weight: bold">}</span>,
        <span style="font-weight: bold">{</span>
            <span style="color: #008000; text-decoration-color: #008000">'name'</span>: <span style="color: #008000; text-decoration-color: #008000">'llm-endpoint'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'args'</span>: <span style="font-weight: bold">{</span><span style="color: #008000; text-decoration-color: #008000">'llm_configs'</span>: <span style="font-weight: bold">[</span><span style="color: #008000; text-decoration-color: #008000">'./model_config/meta-llama--Llama-3_2-1B-Instruct.yaml'</span><span style="font-weight: bold">]}</span>,
            <span style="color: #008000; text-decoration-color: #008000">'import_path'</span>: <span style="color: #008000; text-decoration-color: #008000">'rayllm:app'</span>,
            <span style="color: #008000; text-decoration-color: #008000">'route_prefix'</span>: <span style="color: #008000; text-decoration-color: #008000">'/'</span>
        <span style="font-weight: bold">}</span>
    <span style="font-weight: bold">]</span>,
    <span style="color: #008000; text-decoration-color: #008000">'query_auth_token_enabled'</span>: <span style="color: #ff0000; text-decoration-color: #ff0000; font-style: italic">False</span>
<span style="font-weight: bold">}</span>
</pre>



We make use of [application args](https://docs.ray.io/en/latest/serve/advanced-guides/app-builder-guide.html) to provide the path to the program state json and the RayLLM API parameters.

We can now deploy the apps locally with


```python
!serve run local_deploy_dspy.yaml --non-blocking
```

    2024-10-23 06:06:15,039	INFO scripts.py:489 -- Running config file: 'local_deploy_dspy.yaml'.
    2024-10-23 06:06:15,376	INFO worker.py:1601 -- Connecting to existing Ray cluster at address: 10.0.0.25:6379...
    2024-10-23 06:06:15,384	INFO worker.py:1777 -- Connected to Ray cluster. View the dashboard at [1m[32mhttps://session-czqbf1bhvhp98gnjubkguupgc2.i.anyscaleuserdata.com [39m[22m
    2024-10-23 06:06:15,389	INFO packaging.py:359 -- Pushing file package 'gcs://_ray_pkg_b6d9cdaf7a037b8884d1bdce0ebc3376c2829a5d.zip' (1.34MiB) to Ray cluster...
    2024-10-23 06:06:15,404	INFO packaging.py:372 -- Successfully pushed file package 'gcs://_ray_pkg_b6d9cdaf7a037b8884d1bdce0ebc3376c2829a5d.zip'.
    INFO 2024-10-23 06:06:19,484 serve 279613 api.py:277 - Started Serve in namespace "serve".
    [36m(ServeController pid=279682)[0m INFO 2024-10-23 06:06:19,488 controller 279682 application_state.py:881 - Deploying new app 'dspy_client'.
    [36m(ServeController pid=279682)[0m INFO 2024-10-23 06:06:19,489 controller 279682 application_state.py:457 - Importing and building app 'dspy_client'.
    [36m(ProxyActor pid=279760)[0m INFO 2024-10-23 06:06:19,453 proxy 10.0.0.25 proxy.py:1235 - Proxy starting on node 9085971e2ef1cdf0cd5515d8cb45a7cb716afef38ea4a1ed736c820d (HTTP port: 8000).
    [36m(ServeController pid=279682)[0m INFO 2024-10-23 06:06:19,494 controller 279682 application_state.py:881 - Deploying new app 'llm-endpoint'.
    [36m(ServeController pid=279682)[0m INFO 2024-10-23 06:06:19,495 controller 279682 application_state.py:457 - Importing and building app 'llm-endpoint'.
    2024-10-23 06:06:19,498	SUCC scripts.py:540 -- [32mSubmitted deploy config successfully.[39m
    [0m

## Query the deployed DSPy service

We can query our app directly using HTTP requests.


```python
import requests

example_query = "My card was charged more than expected."
response = requests.post("http://localhost:8000/classify_intent", json={"query": example_query})

if not response.ok:
    print("Got response: ", response)
    print("Reason: ", response.reason)
else:
    print(response.text)
```

    request_refund


# Optional: Deploy DSPy program as an Anyscale Service

You can optionally deploy your program to Anyscale in order to use it in production. We will repeat the same steps as above, but this time deploy the two applications as separate Anyscale services for convenience.

## Step 1: Deploy RayLLM as a Anyscale Service

<b style="background-color: blue;">&nbsp;🔄 RUN (optional)&nbsp;</b>:
To deploy the fine-tuned model as an Anyscale Service, run the following command:

```
!anyscale service deploy -f serve_1B.yaml
```

Follow the URL in order to find your service URL and API key for your deployed service.



```python
# Run this if you want to deploy an Anyscale service
!anyscale service deploy -f serve_1B.yaml
```

    [1m[36m(anyscale +2.0s)[0m [0m[0m[0m[0mStarting new service 'dspy-rayllm-service'.[0m
    [0m[1m[36m(anyscale +4.0s)[0m [0m[0m[0m[0mUsing workspace runtime dependencies env vars: {'FOO': '"Bar"'}.[0m
    [0m[1m[36m(anyscale +4.0s)[0m [0m[0m[0m[0mUploading local dir '.' to cloud storage.[0m
    [0m[1m[36m(anyscale +5.7s)[0m [0m[0m[0m[0mIncluding workspace-managed pip dependencies.[0m
    [0m[1m[36m(anyscale +6.9s)[0m [0m[0m[0m[0mService 'dspy-rayllm-service' deployed.[0m
    [0m[1m[36m(anyscale +6.9s)[0m [0m[0m[0m[0mView the service in the UI: 'https://console.anyscale.com/services/service2_3kg2zf69v85dn3d6l3iluqkk4f'[0m
    [0m[1m[36m(anyscale +6.9s)[0m [0m[0m[0m[0mQuery the service once it's running using the following curl command:[0m
    [0m[1m[36m(anyscale +6.9s)[0m [0m[0m[0m[0mcurl https://dspy-rayllm-service-mglb8.cld-tffbxe9ia5phqr1u.s.anyscaleuserdata.com/[0m
    [0m[0m

<b style="background-color: yellow;">&nbsp;🔄 REPLACE&nbsp;</b>:
Replace the following variables with your Anyscale service URL and API key.


You can find them by clicking the query button on the Anyscale dashboard for your service.

<!-- <img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-dspy-workflow/files/service-query.png" alt="Service Query" width="500"> -->
![Service Query](README_files/service-query.png)


```python
ANYSCALE_RAYLLM_SERVICE_BASE_URL = None
ANYSCALE_RAYLLM_API_KEY = "dummy-key" # Use the provided API key if query_auth_token_enabled is `True`
```

## Step 2: Deploy the DSPy program as a Anyscale Service

To deploy our DSPy program, we first need to make sure we point to the Anyscale Service running RayLLM for the API parameters used. To do this, we should update the `rayllm_args` section of our config


```python
import yaml

def update_rayllm_config(yaml_path, new_api_base=None, new_api_key=None, new_route_prefix=None):
    """
    Update the RayLLM configuration in the YAML file.
    
    Args:
        yaml_path (str): Path to the YAML file
        new_api_base (str, Optional): New API base URL
        new_api_key (str, Optional): New API key
        new_route_prefix (str, Optional): New route prefix
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    rayllm_args = config['applications'][0]['args']['rayllm_args']
    
    if new_api_base:
        rayllm_args['api_base'] = new_api_base
    if new_api_key:
        rayllm_args['api_key'] = new_api_key
    if new_route_prefix:
        rayllm_args['route_prefix'] = new_route_prefix
    
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
```


```python
# Run this if you deployed an Anyscale Service with RayLLM

update_rayllm_config("configs/anyscale_deploy.yaml", new_api_base=ANYSCALE_RAYLLM_SERVICE_BASE_URL, new_api_key=ANYSCALE_RAYLLM_API_KEY)
```


```python
!anyscale service deploy -f configs/anyscale_deploy.yaml
```

    [1m[36m(anyscale +2.4s)[0m [0m[0m[0m[0mStarting new service 'dspy-service'.[0m
    [0m[1m[36m(anyscale +3.2s)[0m [0m[0m[0m[0mUsing workspace runtime dependencies env vars: {'FOO': '"Bar"'}.[0m
    [0m[1m[36m(anyscale +3.2s)[0m [0m[0m[0m[0mUploading local dir '.' to cloud storage.[0m
    [0m[1m[36m(anyscale +7.5s)[0m [0m[0m[0m[0mService 'dspy-service' deployed.[0m
    [0m[1m[36m(anyscale +7.5s)[0m [0m[0m[0m[0mView the service in the UI: 'https://console.anyscale.com/services/service2_l6ivxkmxypevqv5893bs3rq23e'[0m
    [0m[1m[36m(anyscale +7.5s)[0m [0m[0m[0m[0mQuery the service once it's running using the following curl command:[0m
    [0m[1m[36m(anyscale +7.5s)[0m [0m[0m[0m[0mcurl https://dspy-service-mglb8.cld-tffbxe9ia5phqr1u.s.anyscaleuserdata.com/[0m
    [0m[0m

Great! We should now have a service which runs the compiled DSPy program. Let's query this new service. Make sure to enter the details here: 


```python
ANYSCALE_DSPY_SERVICE_BASE_URL = None
ANYSCALE_DSPY_API_KEY = "fake-api-key" # Use the provided API key if query_auth_token_enabled is `True`
```


```python
import requests

headers = {
    "Authorization": f"Bearer {ANYSCALE_DSPY_API_KEY}"
}
example_query = "My card was charged more than expected."

response = requests.post(f"{ANYSCALE_DSPY_SERVICE_BASE_URL.rstrip('/')}/classify_intent", json={"query": example_query}, headers=headers)

if not response.ok:
    print("Got response: ", response)
    print("Reason: ", response.reason)
else:
    print(response.text)
```

    request_refund


<b style="background-color: yellow;">&nbsp;🛑 IMPORTANT&nbsp;</b>: Please `Terminate` your service from the Service page to avoid depleting your credits.


```python
# Clean up
!python src/clear_cell_nums.py
!find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
!find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
!rm -rf __pycache__ data .HF_TOKEN deploy/services
```
