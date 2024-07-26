# Intro to Fine-tuning Open-weight LLMs with Anyscale

**⏱️ Time to complete**: ~3 hours (includes the time for training the model)


This template comes with a installed library for training LLMs on Anyscale called LLMForge. It provides the fastest way to try out training LLMs with Ray on Anyscale. You can read more about this library and its features in the [docs](https://docs.anyscale.com/latest/llms/finetuning/intro). For learning on how to serve the model online or offline for doing batch inference you can refer to the [serving template](https://console.anyscale.com/v2/template-preview/endpoints_v2) or the [offline batch inference template](https://console.anyscale.com/v2/template-preview/batch-llm), respecitvely.


## Getting Started

You can find some tested config files examples in the `training_configs` directoy. LLMForge comes with a CLI that lets you pass in a config YAML file to start your training.


```bash
WANDB_API_KEY=<PUT_YOUR_WANDB_KEY_HERE> llmforge anyscale finetune training_configs/custom/meta-llama--Meta-Llama-3-8B-Instruct/lora/16xA10-512.yaml
```

This code will run LoRA fine-tuning on the Meta-Llama-3-8B-Instruct model with 16xA10-512 configuration on a GSM-8k math dataset.

When the training is done, you will see a message like this:

```bash
Note: LoRA weights will also be stored in path <path>
````

This is the path where the adapted weights are stored, you can use them fore inference. You can also see the list of your fine-tuned models in the `serving` tab in the Anyscale console.

# What is Next?

* Make sure to checkout the [LLMForge documentation](https://docs.anyscale.com/latest/llms/finetuning/intro) and [user guides](https://docs.anyscale.com/latest/llms/finetuning/user-guides) for more information on how to use the library and the features it supports.
* You can follow the [serving template](https://console.anyscale.com/v2/template-preview/endpoints_v2) to learn how to serve the model online.
* You can follow the [offline batch inference template](https://console.anyscale.com/v2/template-preview/batch-llm) to learn how to do batch inference.




--------- 
**Task:** 

Fine-tune llama-3-8b-instruct in default mode (LoRA rank 8). Just giving the dataset.

**Command:**
```bash
llmforge anyscale finetune training_configs/default/meta-llama/Meta-Llama-3-8B-Instruct-simple.yaml --default
```

**Config:**

```yaml
model_id: meta-llama/Meta-Llama-3-8B-Instruct
train_path: s3://...
```


--------- 

**Task:** 

Fine-tune llama-3-8b-instruct in default mode but also control parameters like `learning_rate` and `num_epochs`. 

**Command:**
```bash
llmforge anyscale finetune training_configs/default/meta-llama/Meta-Llama-3-8B-Instruct-custom.yaml --default
```

**Config:**

```yaml
model_id: meta-llama/Meta-Llama-3-8B-Instruct
train_path: s3://...
valid_path: s3://...
num_epochs: 3
learning_rate: 1e-4         
```


### Custom

---------
**Task:** 

Fine-tune llama-3-8b-instruct (a "core" model) in custom mode on 16xA10s (auto mode uses 8xA100-80G) with context length of 512.


**Command:** 

```bash
llmforge anyscale finetune training_configs/custom/meta-llama--Meta-Llama-3-8B-Instruct/lora/16xA10-512.yaml 
```

**Config:**

```yaml
model_id: meta-llama/Meta-Llama-3-8B-Instruct
train_path: s3://...
valid_path: s3://...
context_length: 512
deepspeed:
  config_path: deepspeed_configs/zero_3_offload_optim+param.json
worker_resources:
  accelerator_type:A10G: 0.001
```


---------
**Task:** 

Fine-tune gemma-2-27b in custom mode on 8xA100-80G.


**Command:** 

```bash
llmforge anyscale finetune training_configs/custom/google--gemma-2-27b-it/lora/8xA100-80G-512.yaml 
```

**Config:**

```yaml
model_id: google/gemma-2-27b-it
train_path: s3://...
valid_path: s3://...
num_devices: 8
worker_resources:
  accelerator_type:A100-80G: 0.001
generation_config:
  prompt_format:
    system: "{instruction} + "
    assistant: "<start_of_turn>model\n{instruction}<end_of_turn>\n"
    trailing_assistant: "<start_of_turn>model\n"
    user: "<start_of_turn>user\n{system}{instruction}<end_of_turn>\n"
    system_in_user: True
    bos: "<bos>"
    default_system_message: ""
  stopping_sequences: ["<end_of_turn>"]
```

More examples can be found in `./training_configs`. For specific features read [cookbooks](#cookbooks) and [end-to-end examples](#end-to-end-examples).

## Cookbooks

After you are with the above, you can find recipies that extend the functionality of this template under the cookbooks folder:

* [Bring your own data](cookbooks/bring_your_own_data/README.md): Everything you need to know about using custom datasets for fine-tuning.
* [Bring any huggingface model and prompt format](cookbooks/bring_any_hf_model/README.md): Learn how you can finetune any 🤗Hugging Face model with a custom prompt format (chat template). 
* [LoRA vs. full-parameter training](cookbooks/continue_from_checkpoint/README.md): Learn the differences between LoRA and full-parameter training and how to configure both.
* [Continue fine-tuning from a previous checkpoint](cookbooks/continue_from_checkpoint/README.md): A detailed guide on how you can use a previous checkpoint for another round of fine-tuning.
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
| `0.5.2`  | `localhost:5555/anyscale/llm-forge:0.5.2` |
| `0.5.1`  | `localhost:5555/anyscale/llm-forge:0.5.1` |
| `0.5.0.1`  | `localhost:5555/anyscale/llm-forge:0.5.0.1-ngmM6BdcEdhWo0nvedP7janPLKS9Cdz2` |


