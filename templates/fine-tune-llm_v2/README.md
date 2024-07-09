# Fine-tuning Llama-3, Mistral and Mixtral with Anyscale

**⏱️ Time to complete**: 2.5 hours for 7/8B models (9 hours for 13B, 25 hours for 70B)

The guide below walks you through the steps required for fine-tuning of LLMs. This template provides an easy to configure solution for ML Platform teams, Infrastructure engineers, and Developers to fine-tune LLMs.

### Popular base models to fine-tune*

- meta-llama/Meta-Llama-3-8B (Full-param and LoRA)
- meta-llama/Meta-Llama-3-70B (Full-param and LoRA)
- mistralai/Mistral-7B (Full-param and LoRA)
- mistralai/Mixtral-8x7B (LoRA only)

*Any model that has the same architecture and parameter count as above can be finetuned. A subset of popular variants of these models are provided out of the box on this template. For this subset, the Huggingface model id is enough. But for models beyond this list, the location to the weights must be provided. 

A full list of out-of-the-box supported models is in the [FAQ](#faqs) section. In the end we provide more guides in form of [cookbooks](#cookbooks) and [end-to-end examples](#end-to-end-examples) that provide more detailed information about using this template.

# Quick start

## Step 1 - Launch a fine-tuning run in [workspaces](https://docs.anyscale.com/platform/workspaces/)

We provide example configurations under the `./training_configs` directory for different base models and accelerator types. You can use these as a starting point for your own fine-tuning jobs. The full-list of public configurations that are customizable see [Anyscale docs](https://docs.anyscale.com/reference/finetuning-config-api).

**Optional**: You can get a WandB API key from [WandB](https://wandb.ai/authorize) to track the fine-tuning process. If not provided, you can only track the experiments through the standard output logs.

Next, you can launch a fine-tuning job with your WandB API key passed as an environment variable.


```python
# [Optional] You can set the WandB API key to track model performance
# import os
# os.environ["WANDB_API_KEY"]="YOUR_WANDB_API_KEY"

# Launch a LoRA fine-tuning job for Llama 3 8B with 16 A10s
!llmforge anyscale finetune training_configs/lora/llama-3-8b.yaml

# Launch a full-param fine-tuning job for Llama 3 8B with 16 A10s
# !llmforge anyscale finetune  training_configs/full_param/llama-3-8b.yaml
```

`LLMForge` is an Anyscale CLI and library that is installed on this workspace so that you can quickly experiment and customize various LLM finetuning experiments by simply modifying a config file. For extensive documentation around what is supported through the config refer to [docs](https://docs.anyscale.com/reference/finetuning-config-api/). 


```python
# To get help on the CLI
!llmforge anyscale finetune --help
```

    [2024-06-28 14:38:55,193] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    Usage: llmforge anyscale finetune [OPTIONS] CONFIG
    
      Runs finetuning with LLMForge on a given configuration file.
    
      This is supposed to be used in the context of Anyscale platform either in
      Workspace or as entrypoint of a job.
    
      Args:
    
          CONFIG: Path to the YAML configuration. See docs for more info.
    
    Options:
      --help  Show this message and exit.


As the command runs, you can monitor a number of built-in metrics in the `Metrics` tab under `Ray Dashboard`, such as the number of GPU nodes and GPU utilization.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/fine-tune-llm_v2/assets/gpu-usage.png" width=500px/>

Depending on whether you are running LoRA or full-param fine-tuning, you can continue with step 2(a) or step 2(b). To learn more about LoRA vs. full-parameter, see the cookbooks.



## Step 2(a) - Serving the LoRA fine-tuned model

Upon the job completion, you can see the LoRA weight storage location and model ID in the log, such as the one below:

```shell
Note: LoRA weights will also be stored in path {ANYSCALE_ARTIFACT_STORAGE}/lora_fine_tuning under meta-llama/Llama-2-8b-chat-hf:sql:12345 bucket.
```

You can specify this URI as the dynamic_lora_loading_path [docs](https://docs.anyscale.com/examples/deploy-llms#more-guides) in the llm serving template, and then query the endpoint.

> Note: Such LoRA model IDs follow the format `{base_model_id}:{suffix}:{id}`


## Step 2(b) - Serving the full-parameter fine-tuned model

Once the fine-tuning job is complete, you can view the stored full-parameter fine-tuned checkpoint at the very end of the job logs. Here is an example fine-tuning job output:

```shell
Best checkpoint is stored in:
{ANYSCALE_ARTIFACT_STORAGE}/username/llmforge-finetuning/meta-llama/Llama-2-70b-hf/TorchTrainer_2024-01-25_18-07-48/TorchTrainer_b3de9_00000_0_2024-01-25_18-07-48/checkpoint_000000
```

Follow the [Learn how to bring your own models](https://docs.anyscale.com/examples/deploy-llms#more-guides) section under the llm serving template to serve this fine-tuned model with the specified storage uri.

## Cookbooks

After you are with the above, you can find recipies that extend the functionality of this template under the cookbooks folder:

* [Bring your own data](cookbooks/bring_your_own_data/README.md): Everything you need to know about using custom datasets for fine-tuning.
* [Customize initial weights and prompt format](cookbooks/customize_initial_weights_and_prompt_format/README.md): Learn how you can finetune a model with a similar architecture to the Llama or Mistral family and customize the chat template/ prompt format. 
* [Continue fine-tuning from a previous checkpoint](cookbooks/continue_from_checkpoint/README.md): A detailed guide on how you can use a previous checkpoint for another round of fine-tuning.
* [LoRA vs. full-parameter training](cookbooks/continue_from_checkpoint/README.md): Learn the differences between LoRA and full-parameter training and how to configure both.
* [Modifying hyperparameters](cookbooks/modifying_hyperparameters/README.md): A brief guide on customization of your fine-tuning job.
* [Optimizing Cost and Performance for Finetuning](cookbooks/optimize_cost/README.md): A detailed guide on default performance-related parameters and how you can optimize throughput for training on your own data.
* [Run finetuning as Anyscale Job](cookbooks/launch_as_anyscale_job/README.md): A detailed guide on how to submit a finetuning workflow as a job (outside the context of workspaces.)

## End-to-end Examples

Here is a list of end-to-end examples that involve more steps such as data preprocessing, evaluation, etc but with a main focus on improving model quality via fine-tuning.

* [Fine-tuning for Function calling on custom data](end-to-end-examples/fine-tune-function-calling/README.md)

## LLMForge Versions

Here is a list of LLMForge image versions:

| version | image_uri |
|---------|-----------|
| `0.5.0.1`  | `localhost:5555/anyscale/llm-forge:0.5.0.1-ngmM6BdcEdhWo0nvedP7janPLKS9Cdz2` |


## FAQs

### Where can I view the bucket where my LoRA weights are stored?

All the LoRA weights are stored under the URI `${ANYSCALE_ARTIFACT_STORAGE}/lora_fine_tuning` where `ANYSCALE_ARTIFACT_STORAGE` is an environmental variable in your workspace.

### What's the full list of supported models?

This is a growing list but it includes the following models:

- meta-llama/Meta-Llama-3-8B
- meta-llama/Meta-Llama-3-8B-Instruct
- meta-llama/Meta-Llama-3-70B
- meta-llama/Meta-Llama-3-70B-Instruct
- meta-llama/Llama-2-7b-hf
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-13b-hf
- meta-llama/Llama-2-13b-chat-hf
- meta-llama/Llama-2-70b-hf
- meta-llama/Llama-2-70b-chat-hf
- codellama/CodeLlama-34b-Instruct-hf
- mistralai/Mistral-7B-Instruct-v0.1
- mistralai/Mixtral-8x7B-Instruct-v0.1

In general, any model that is compatible with the architecture of these models can be fine-tuned using the same configs as the base models.

NOTE: currently mixture of expert models (such as `mistralai/Mixtral-8x7B)` only support LoRA fine-tuning

### What's with the `main` file that is created during fine-tuning?

It's an artifact of our fine-tuning libraries. Please ignore it.


