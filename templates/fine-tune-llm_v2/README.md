# Fine-tuning Llama-3, Mistral and Mixtral with Anyscale

**⏱️ Time to complete**: 2.5 hours for 7/8B models (9 hours for 13B, 25 hours for 70B)

The guide below walks you through the steps required for fine-tuning of LLMs. This template provides an easy to configure solution for ML Platform teams, Infrastructure engineers, and Developers to fine-tune LLMs.

### Popular base models to fine-tune

- meta-llama/Meta-Llama-3-8B-Instruct
- meta-llama/Meta-Llama-3-70B-Instruct
- mistralai/Mistral-7B-Instruct-v0.1
- mistralai/Mixtral-8x7b

A full list of supported models is in the FAQ section.

## Step 1 - Launch a fine-tuning job

We provide example configurations under the `training_configs` directory for different base models and accelerator types. You can use these as a starting point for your own fine-tuning jobs.

[Optional] you can get a WandB API key from [WandB](https://wandb.ai/authorize) to track the fine-tuning process.

Next, you can launch a fine-tuning job with your WandB API key passed as an environment variable.


```python
# [Optional] You can set the WandB API key to track model performance
# !export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# Launch a LoRA fine-tuning job for Llama 3 8B with 16 A10s
!python main.py training_configs/lora/llama-3-8b.yaml

# Launch a full-param fine-tuning job for Llama 3 8B with 16 A10s
# !python main.py training_configs/full_param/llama-3-8b.yaml
```

As the command runs, you can monitor a number of built-in metrics in the `Metrics` tab under `Ray Dashboard`, such as the number of GPU nodes and GPU utilization.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/fine-tune-llm_v2/assets/gpu-usage.png" width=500px/>

Depending on whether you are running LoRA or full-param fine-tuning, you can continue with step 2(a) or step 2(b).
Find out more about full-paramter vs. LoRA fine-tuning in the FAQ section.

## Step 2(a) - Serving the LoRA fine-tuned model

Upon the job completion, you can see the LoRA weight storage location and model ID in the log, such as the one below:

```shell
Note: LoRA weights will also be stored in path {ANYSCALE_ARTIFACT_STORAGE}/lora_fine_tuning under meta-llama/Llama-2-8b-chat-hf:sql:12345 bucket.
```

You can specify this URI as the dynamic_lora_loading_path [docs](https://docs.anyscale.com/examples/deploy-llms#more-guides) in the llm serving template, and then query the endpoint.

Note: Such LoRA model IDs follow the format `{base_model_id}:{suffix}:{id}`

## Step 2(b) - Serving the full-parameter fine-tuned model

Once the fine-tuning job is complete, you can view the stored full-parameter fine-tuned checkpoint at the very end of the job logs. Here is an example fine-tuning job output:

```shell
Best checkpoint is stored in:
{ANYSCALE_ARTIFACT_STORAGE}/username/llmforge-finetuning/meta-llama/Llama-2-70b-hf/TorchTrainer_2024-01-25_18-07-48/TorchTrainer_b3de9_00000_0_2024-01-25_18-07-48/checkpoint_000000
```

Follow the [Learn how to bring your own models](https://docs.anyscale.com/examples/deploy-llms#more-guides) section under the llm serving template to serve this fine-tuned model with the specified storage uri.

## Finally

After you are with the above, you can find recipies that extend the functionality of this template under the cookbooks folder:

* (Continue fine-tuning from a previous checkpoint)[cookbooks/continue_from_checkpoint/README.md]

## Frequently asked questions

### Where can I view the bucket where my LoRA weights are stored?

All the LoRA weights are stored under the URI `${ANYSCALE_ARTIFACT_STORAGE}/lora_fine_tuning` where `ANYSCALE_ARTIFACT_STORAGE` is an environmental variable in your workspace.

### How can I fine-tune using my own data?

The training configs provided in this template all train on the [GSM8k dataset](https://huggingface.co/datasets/gsm8k) which requires a context length of 512 tokens. How to ensure the correct format for your own dataset is described in https://docs.endpoints.anyscale.com/fine-tuning/dataset-prep.

Open the file under `training_configs` and update `train_path` and `valid_path` to your training- and evaluation file.

### How do I customize the fine-tuning job?

You can edit the values, such as `context_length`, `num_epoch`, `train_batch_size_per_device` and `eval_batch_size_per_device` to customize the fine-tuning job. You may be able to reach higher model-quality if you tweak the learning rate but also possibly introduce learning instabilities that can be monitored in [WandB](https://wandb.ai/authorize). In addition, the deepspeed configs are provided within this template in case you want to customize them.

### What's the full list of supported models?

This is a growing list but it includes the following models:

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

In general, any model that is compatible with the architecture of these models can be fine-tuned using the same configs as the base models.

### Should I use LoRA or full-parameter fine-tuning?

There is no general answer to this but here are some things to consider:

- The quality of the fine-tuned models will, in most cases, be comparable if not the same
- LoRA shines if...
    - ... you want to serve many fine-tuned models at once yourself
    - ... you want to rapidly experiment (because fine-tuning, downloading and serving the model take less time)
- Full-parameter shines if...
    - ... you want to make sure that your fine-tuned model has the maximum quality
    - ... you want to serve only one fine-tuned version of the model

You can learn more about this in one of our [blogposts](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2).
There, you'll also find some guidance on the LoRA parameters and why, in most cases, you don't need to change them.

### I have the right model, context length and everything. Can I optimize compute cost?

Optimizing your fine-tuning runs for compute cost is a non-trivial problem.
The default configs in this template require the following compute:
Llama-3-8B and Mistral require 16 A10Gs. Llama-3-70B and Mixtral require 32 A10Gs.

Before optimizing for compute, make sure that you have selected a context length that is long enough for your dataset. If you have very few datapoints in your dataset that requires a much larger context than the others, consider removing them. The model of your choice and fine-tuning technique should also suit your data.

If you want different compute, we *suggest* the following workflow to find a suitable configuration:

* Start with a batch size of 1
* Choose a GPU instance type that you think will give you good flops/$. If you are not sure, here is a rough guideline:
    * g5 nodes for high availability
    * p4d/p4de nodes for lower availability but better flops/$
    * Anything higher-end if you have the means of acquiring them
* Do some iterations of trial and error on instance types and deepspeed settings to fit the workload while keeping other settings fixed
    * Use deepspeed stage 3 (all default configs in this template use stage 3)
    * Try to use deepspeed offloading only if it reduces the minimum number of instances you have to use
        * Deepspeed offloading slows down training but allows for larger batch sizes because of a more relaxed GRAM foot-print
    * Use as few instances as possible. Fine-tune on the same machine if possible.
        *  The GPU to GPU communication across machines is very expensive compared to the memory savings it could provide. You can use a cheap CPU-instance as a head-node for development and a GPU-instance that can scale down as a worker node for the heavy lifting.
        * Training single-node on A100s may end up cheaper than multi-node on A10s if availablity is not an issue
* Be aware that evaluation and checkpointing introduce their own memory-requirements
   * If things look good, run fine-tuning for a full epoch.
* After you have followed the steps above, increase batch size as much as possible without OOMing.

We do not guarantee that this will give you optimal settings, but have found this workflow to be helpful ourselves in the past.

### How can I get even more control?

This template fine-tunes with Anyscale's library `llmforge`, which uses [DeepSpeed](https://github.com/microsoft/DeepSpeed) and [Ray Train](https://docs.ray.io/en/latest/train/train.html) for distributed training.
You can study main.py to find out how we call the `lmforge dev finetune` API with a YAML that specifies the fine-tuning workload.
You can call `lmforge dev finetune` yourself and gain control by modifying the training config YAMLs in this template.
For anything that goes beyond using `llmforge`, you can build your own fine-tuning stack on Anyscale.

### What's with the `main` file that is created during fine-tuning?

It's an artifact of our fine-tuning libraries. Please ignore it.


