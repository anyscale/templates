# Fine-tuning LLMs with LLaMA-Factory on Anyscale

**⏱️ Time to complete**: ~20 mins (includes the time for training the model)

Looking to get the most out of your LLM workloads? Fine-tuning pretrained LLMs can often be the most cost effective way to improve your model's performance on the tasks that you care about. This template will walk you through how you can use the popular open source library [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune LLMs on Anyscale. LLaMA-Factory comes with a [Ray](https://www.ray.io/) integration for scaling training to multiple GPUs and nodes.


## Getting started with LLaMA-Factory 
First, you need to install the LLaMA-Factory code. You can view the latest changes from the [LLaMA-Factory github](https://github.com/hiyouga/LLaMA-Factory.git). 

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
# install extras separately so Anyscale can track the dependencies on worker nodes
pip install torch jieba nltk rouge-chinese 
pip install -e .
```

Next, you need to set your `HF_TOKEN` and `WANDB_API_KEY` environment variables. It is recommended to do this via the environment variables section of the [dependencies tab](https://docs.anyscale.com/configuration/dependency-management/dependency-development/#environment-variables) in Workspaces. Simply running `export HF_TOKEN=<HF_TOKEN_HERE>`, or `huggingface-cli login` will work for single node compute configurations, but will not propagate the environment variables to worker nodes.

Additionally it is recommended to use `hf_transfer` for fast model downloading. You can do this by running `pip install hf_transfer`, and setting `HF_HUB_ENABLE_HF_TRANSFER=1` in the dependencies tab. An example of setting relevant environment variables in the dependencies tab is shown below.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/fine-tune-llm-oss/assets/env_vars.png" width=500px />

You can find some example configs that use Ray with LLaMA-Factory for pretraining and SFT (instruction tuning) in the `llamafactory_config` directory. A full set of examples with various models, tasks, and datasets can be found on the [LLaMA-Factory github](https://github.com/hiyouga/LLaMA-Factory/tree/main/examples). 

LLaMA-Factory provides a cli `llamafactory-cli` that allows you to launch training with a config yaml file. For example, you can launch a supervised finetuning job with ray by running the following command:
```bash
cd .. # return to top level directory
USE_RAY=1 llamafactory-cli train llamafactory_configs/llama3_lora_sft_ray.yaml
```
This will run an instruction tuning training job with `Meta-Llama-3-8B-Instruct` on an example subset of the alpaca dataset. Training statistics will by default be logged with all installed logging libraries (i.e. wandb, mlflow, comet, tensorboard), since LLaMA-Factory relies on HuggingFace's [Trainer integration](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.report_to) for logging. To specify a single specific library to log with, set `report_to: <LIBRARY_NAME>` in the config YAML.

## Preparing your dataset

The above config (`llamafactory_configs/llama3_lora_sft_ray.yaml`) runs out of the box, since it assumes that the dataset and dataset information is located on the huggingface hub, and specifies it as below
```yaml
...
### dataset
dataset: identity,alpaca_en_demo
dataset_dir: REMOTE:llamafactory/demo_data  # or use local absolute path
...
```

However, by default LLaMA-Factory reads dataset information from a local file `dataset_info.json` (which can be found in `LLaMA-Factory/data/`)

To specify new datasets and have them be accessible across Ray worker nodes, you must first add `dataset_info.json` and any data files to shared storage like `/mnt/cluster_storage`. 

For example, if you wanted to run pretraining on the `c4_demo` dataset, you would first go through the following setup steps:
```bash
# create a copy of the data in /mnt/cluster_storage
cp LLaMA-Factory/data/c4_demo.json /mnt/cluster_storage/
```


```bash
# create `/mnt/cluster_storage.dataset_info.json` containing
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

To launch finetuning with this local dataset, you can run
```bash
USE_RAY=1 llamafactory-cli train llamafactory_configs/llama3_lora_pretrain_ray.yaml
```

## Running LLaMA-Factory in an Anyscale job
In order to run LLaMA-Factory as an Anyscale job, you have to create a [custom container image](https://docs.anyscale.com/configuration/dependency-management/dependency-container-images/#customizing-a-container-image()) that comes with LLaMA-Factory installed. You can specify the relevant packages in your dockerfile. For example, you could create an image using the `anyscale/ray-ml:2.42.0-py310-gpu` base image as follows.

```dockerfile
# Start with an Anyscale base image.
FROM anyscale/ray-ml:2.42.0-py310-gpu
WORKDIR /app
RUN git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git && \
    cd LLaMA-Factory && \
    pip install --no-cache-dir torch jieba nltk rouge-chinese && \
    pip install -e .
```

You can then submit use this image to run your job via the [Anyscale jobs cli](https://docs.anyscale.com/platform/jobs/manage-jobs) locally or from a workspace. For example, from a workspace configured with the image you built above, you could run the following command to launch finetuning as a job.
```bash
anyscale job submit --wait --env USE_RAY=1 -- llamafactory-cli train llamafactory_configs/llama3_lora_sft_ray.yaml
```


