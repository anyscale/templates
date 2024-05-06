# Fine-tuning Llama-3, Mistral and Mixtral with Anyscale

**⏱️ Time to complete**: 2.5 hours for 7/8B models (9 hours for 13B, 25 hours for 70B)

The guide below walks you through the steps required for fine-tuning of LLM models. This template provides an easy to configure solution for ML Platform teams, Infrastructure engineers, and Developers to fine-tune LLMs.

### Popular base models to fine-tune

- meta-llama/Meta-Llama-3-8B-Instruct
- meta-llama/Meta-Llama-3-70B-Instruct
- mistralai/Mistral-7B-Instruct-v0.1
- mistralai/Mixtral-8x7b

A full list of supported models is in the FAQ section.

## Step 1 - Launch a fine-tuning job

We have provided different example configurations under the `training_configs` directory for different base models and accelerator types. You can use these as a starting point for your own fine-tuning jobs.

[Optional] you can get a WandB API key from [WandB](https://wandb.ai/authorize) to track the fine-tuning process.

Next, you can launch a fine-tuning job where the WandB API key is passed as an environment variable.

```python
# [Optional] You can set the WandB API key to track model performance
# !export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# Launch a full-param fine-tuning job for Llama 3 8B with 16 A10s
!python main.py training_configs/full_param/llama-3-8b.yaml

# Launch a LoRA fine-tuning job for Llama 3 8B with 16 A10s
# !python main.py training_configs/lora/llama-3-8b.yaml
```

As the command runs, you can monitor a number of built-in metrics out of the box in the `Metrics` tab under `Ray Dashboard`, such as the number of GPU nodes and GPU utilization.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/fine-tune-llm_v2/assets/gpu-usage.png" width=500px/>

Depending on whether you are running LoRA or full-param fine-tuning, you can continue with step 2(a) or step 2(b).

## Step 2(a) - Serving the LoRA fine-tuned model

Upon the job completion, you can see the LoRA weight storage location and model ID in the log, such as the one below:

```shell
Note: LoRA weights will also be stored in path {ANYSCALE_ARTIFACT_STORAGE}/lora_fine_tuning under meta-llama/Llama-2-8b-chat-hf:sql:12345 bucket.
```

You can specify this URI as the dynamic_lora_loading_path ([docs](https://docs.endpoints.anyscale.com/preview/examples/deploy-llms/#more-guides) in the llm serving template, and then query the endpoint.

Note: Such LoRA model IDs follow the format `{base_model_id}:{suffix}:{id}`

## Step 2(b) - Serving the full-parameter fine-tuned model

Once the fine-tuning job is complete, you can view the stored full-parameter fine-tuned checkpoint at the very end of the job logs. Here is an example fine-tuning job output:

```shell
Best checkpoint is stored in:
{ANYSCALE_ARTIFACT_STORAGE}/username/llmforge-finetuning/meta-llama/Llama-2-70b-hf/TorchTrainer_2024-01-25_18-07-48/TorchTrainer_b3de9_00000_0_2024-01-25_18-07-48/checkpoint_000000
```

Follow the [Learn how to bring your own models](https://docs.endpoints.anyscale.com/preview/examples/deploy-llms/#more-guides) section under the llm serving template to serve this fine-tuned model with the specified storage uri.

## Frequently asked questions

### Where can I view the bucket where my LoRA weights are stored?

All the LoRA weights are stored under the URI `${ANYSCALE_ARTIFACT_STORAGE}/lora_fine_tuning` where `ANYSCALE_ARTIFACT_STORAGE` is an environmental variable.

### How can I fine-tune using my own data?

You can open the file under `training_configs` and update `train_path` and `valid_path` to your training and evaluation file.

### How do I customize the fine-tuning job?

You can edit the values, such as `context_length`, `num_epoch`, `train_batch_size_per_device` and `eval_batch_size_per_device` to customize the fine-tuning job.

In addition, the deepspeed configs are provided in case you would
like to customize them.

### How can I get more control?

This template fine-tunes with Anyscale's library `llmforge`.
You can study main.py to find out how we call the `lmforge dev finetune` API with a YAML that specifies the fine-tuning workload.
You can call `lmforge dev finetune` yourself and gain control by modifying the training config YAMLs in this template.

### What's the full list of supported models?

The following models can be fine-tuned with `llmforge`.

- mistralai/Mistral-7B-Instruct-v0.1
- mistralai/Mixtral-8x7b
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-13b-hf
- meta-llama/Llama-2-13b-chat-hf
- meta-llama/Llama-2-70b-hf
- meta-llama/Llama-2-70b-chat-hf
- meta-llama/Meta-Llama-3-8B
- meta-llama/Meta-Llama-3-8B-Instruct
- meta-llama/Meta-Llama-3-70B
- meta-llama/Meta-Llama-3-70B-Instruct

### What if I want to use another dataset? 

The training configs provided in this template all train on the GSM8k which requires a context length of 512 tokens.
How to ensure the correct format is described in https://docs.endpoints.anyscale.com/fine-tuning/dataset-prep.
You can replace the s3 buckets in the training configs with paths to your own dataset.


### How can I use more or less compute?

The training configs provided in this template require 16 A10G GPUs by default.
The compute requirements depend on things such as the model size, batch sizes and context length that are set in the training configs.
You can change these paramters in the training configs and change the used compute within your workspace.

