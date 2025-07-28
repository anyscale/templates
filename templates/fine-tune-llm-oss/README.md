# Fine-tuning LLMs with LLaMA-Factory on Anyscale

**⏱️ Time to complete**: ~20 mins, which includes the time to train the model

Looking to get the most out of your LLM workloads? Fine-tuning pretrained LLMs can often be the most cost effective way to improve model performance on the tasks that you care about. This template walks through how to use the popular open source library [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune LLMs on Anyscale. LLaMA-Factory comes with a [Ray](https://www.ray.io/) integration for scaling training to multiple GPUs and nodes.

## Getting started with LLaMA-Factory 
First, you need to install the LLaMA-Factory code. You can view the latest changes from the [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory.git). 

```bash
git clone --branch v0.9.3 --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
# Install extras separately so Anyscale can track the dependencies on worker nodes.
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
pip install deepspeed==0.16.4 transformers==4.51.3 jieba nltk rouge-chinese 
pip install -e .
```

Next, you need to set your `HF_TOKEN` and `WANDB_API_KEY` environment variables. It's recommended to set them in the environment variables section of the [**Dependencies** tab](https://docs.anyscale.com/configuration/dependency-management/dependency-development/#environment-variables) in Workspaces. Simply running `export HF_TOKEN=<HF_TOKEN_HERE>`, or `huggingface-cli login` works for single node compute configurations, but doesn't propagate the environment variables to worker nodes.

Additionally it's recommended to use `hf_transfer` for fast model downloading by running `pip install hf_transfer`, and setting `HF_HUB_ENABLE_HF_TRANSFER=1` in the **Dependencies** tab. Setting `HF_HUB_ENABLE_HF_TRANSFER=1` without first installing `hf_transfer` will cause errors. An example of setting relevant environment variables in the **Dependencies** tab is shown below.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/fine-tune-llm-oss/assets/env_vars.png" width=500px />

You can find some example configs that use Ray with LLaMA-Factory for pretraining and SFT (instruction tuning) in the `llamafactory_config` directory. Find a full set of examples with various models, tasks, and datasets on the [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples). 

LLaMA-Factory provides a CLI `llamafactory-cli` that allows you to launch training with a config YAML file. For example, you can launch a supervised fine-tuning job with Ray by running the following command:
```bash
cd .. # Return to top level directory.
USE_RAY=1 llamafactory-cli train llamafactory_configs/llama3_lora_sft_ray.yaml
```
This command runs an instruction tuning training job with `Meta-Llama-3-8B-Instruct` on an example subset of the alpaca dataset. By default, Ray logs training statistics with all installed logging libraries, like W&B, MLflow, comet, tensorboard, because LLaMA-Factory relies on Hugging Face's [Trainer integration](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.report_to) for logging. To specify a single specific library to log with, set `report_to: <LIBRARY_NAME>` in the config YAML.

## Preparing your dataset

The above config (`llamafactory_configs/llama3_lora_sft_ray.yaml`) runs out of the box, because it assumes that the dataset and dataset information is located on the Hugging Face hub, and specifies it as below:
```yaml
...
### dataset
dataset: identity,alpaca_en_demo
dataset_dir: REMOTE:llamafactory/demo_data  # or use local absolute path
...
```

However, by default LLaMA-Factory reads dataset information from a local file `dataset_info.json`, which you can find in `LLaMA-Factory/data/`)

To specify new datasets that are accessible across Ray worker nodes, you must first add `dataset_info.json` and any data files to shared storage like `/mnt/cluster_storage`. 

For example, if you wanted to run pretraining on the `c4_demo` dataset, first go through the following setup steps:
```bash
# Create a copy of the data in /mnt/cluster_storage
cp LLaMA-Factory/data/c4_demo.json /mnt/cluster_storage/
```


```bash
# Create `/mnt/cluster_storage.dataset_info.json` containing:
# {
#     "c4_demo": {
#         "file_name": "/mnt/cluster_storage/c4_demo.json",
#         "columns": {
#         "prompt": "text"
#         }
#     },
# }
echo '{"c4_demo":{"file_name":"/mnt/cluster_storage/c4_demo.json","columns":{"prompt":"text"}}}' > /mnt/cluster_storage/dataset_info.json
```

In `llamafactory_configs/llama3_lora_pretrain_ray.yaml` the relevant `dataset_dir` argument is already specified.
```yaml
...
### dataset
dataset: c4_demo
dataset_dir: /mnt/cluster_storage  # or use local absolute path
...
```

To launch fine-tuning with this local dataset, you can run
```bash
USE_RAY=1 llamafactory-cli train llamafactory_configs/llama3_lora_pretrain_ray.yaml
```

## Running LLaMA-Factory in an Anyscale job
To run LLaMA-Factory as an Anyscale job, create a [custom container image](https://docs.anyscale.com/configuration/dependency-management/dependency-container-images/#customizing-a-container-image) that comes with LLaMA-Factory installed. You can specify the relevant packages in your dockerfile. For example, you could create an image using the `anyscale/ray-ml:2.42.0-py310-gpu` base image as follows.

```docker
# Start with an Anyscale base image.
FROM anyscale/ray-ml:2.42.0-py310-gpu
WORKDIR /app
RUN git clone --branch v0.9.3 --depth 1 https://github.com/hiyouga/LLaMA-Factory.git && \
    cd LLaMA-Factory && \
    pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir deepspeed==0.16.4 transformers==4.51.3 jieba nltk rouge-chinese && \
    pip install -e . 
```

You can then use this new image to run your job with the [Anyscale jobs CLI](https://docs.anyscale.com/platform/jobs/manage-jobs) locally or from a workspace. We provide an example job config in `sft_job_config.yaml` for running a job from this workspace, which contains the following:

```yaml
name: llama3-lora-sft-ray
image_uri: <your_image_uri>:<version>
requirements:
  - hf_transfer
env_vars:
  WANDB_API_KEY: <your_wandb_api_key>
  HF_HUB_ENABLE_HF_TRANSFER: '1'
  HF_TOKEN: <your_hf_token>
  USE_RAY: '1'
cloud: <your-cloud-name>
ray_version: 2.42.0
entrypoint: llamafactory-cli train llamafactory_configs/llama3_lora_sft_ray.yaml
max_retries: 1
```

Once you fill in the image uri of the image you created above, your WandB API key and HF token, and your cloud name, you can simply run the below command to start your training job! Note that the compute config and working directory for the job are inherited from the current workspace.
```bash
anyscale job submit --wait --config-file sft_job_config.yaml
```

The training job should take less than 15 minutes to start up and complete, after which you should see that the job status has been updated to "Succeeded"!

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/fine-tune-llm-oss/assets/completed.png" width=500px />



