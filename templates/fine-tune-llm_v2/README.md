# Fine-tuning Open-weight LLMs with Anyscale

**⏱️ Time to complete**: N/A

Fine-tuning LLMs is an easy and cost-effective way to tailor their capabilities towards niche applications with high-acccuracy. While Ray and RayTrain offer generic primitives for building such workloads, at Anyscale we have created a higher-level library called _LLMForge_ that builds on top of Ray and other open-source libraries to provide an easy to work with interface for fine-tuning and training LLMs. 

This template is a guide on how to use LLMForge for fine-tuning LLMs.


### Table of contents

- [What is LLMForge?](#what-is-llmforge)
  - [Configurations](#configurations)
    - [Default Mode](#default-mode)
    - [Custom Mode](#custom-mode)
  - [Models Supported in default Mode](#models-supported-in-default-mode)
- [Summary of Features in Custom Mode](#summary-of-features-in-custom-mode)
- [Examples](#examples)
  - [Default](#default)
  - [Custom](#custom)
- [Cookbooks](#cookbooks)
- [End-to-end Examples](#end-to-end-examples)
- [LLMForge Versions](#llmforge-versions)

## What is LLMForge?

LLMForge is a library that implements a collection of design patterns that use Ray, RayTrain, and RayData in combination with other open-source libraries (e.g. Deepspeed, 🤗 Huggingface accelerate, transformers, etc.) to provide an easy to use library for fine-tuning LLMs. In addition to these design patterns, it offers tight integrations with the Anyscale platform, such as model registery, streamlined deployment, observability, Anyscale's job submission, etc.

### Configurations

LLMForge workloads are specified using YAML configurations ([documentation here](https://docs.anyscale.com/reference/finetuning-config-api)). The library offers two main modes: `default` and `custom`.

#### Default Mode
Similar to OpenAI's finetuning experience, the `default` mode provides a minimal and efficient setup. It allows you to quickly start a finetuning job by setting just a few parameters (`model_id` and `train_path`). All other settings are optional and will be automatically selected based on dataset statistics and predefined configurations.

#### Custom Mode
The `custom` mode offers more flexibility and control over the finetuning process, allowing for advanced optimizations and customizations. You need to provide more configurations to setup this mode (e.g. prompt format, hardware, batch size, etc.)

Here's a comparison of the two modes:

| Feature | Default Mode | Custom Mode |
|---------|-----------|-------------|
| Ideal For | Prototyping what's possible, focusing on dataset cleaning, finetuning, and evaluation pipeline | Optimizing model quality by controlling more parameters, hardware control |
| Command | `llmforge anyscale finetune config.yaml --default` | `llmforge anyscale finetune config.yaml` |
| Model Support | Popular models with their prompt format (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`)* | Any HuggingFace model, any prompt format (e.g., `meta-llama/Meta-Llama-Guard-2-8B`) |
| Task Support | Instruction tuning for multi-turn chat | Causal language modeling, Instruction tuning, Classification|
| Data Format | Supports chat-style datasets, with fixed prompt formats per model | Supports chat-style datasets, with flexible prompt format |
| Hardware | Automatically selected (limited by availability) | User-configurable |
| Fine-tuning type| Only supports LoRA (Rank-8, all linear layers) | User-defined LoRA and Full-parameter |

*NOTE: old models will get deprecated

Choose the mode that best fits your project requirements and level of customization needed.

### Models Supported in Default Mode

Default mode supports a select list of models, with a fixed cluster type of 8xA100-80G. For each model we only support context lengths of 512 up to Max. context length in increments of 2x (i.e. 512, 1024, ...). Here are the supported models and their configurations:

Model family | model_id(s) | Max. context lengths |
|------------|----------|----------------------|
|Llama-3.1| `meta-llama/Meta-Llama-3.1-8B-Instruct` | 4096 |
|Llama-3.1| `meta-llama/Meta-Llama-3.1-70B-Instruct`  | 4096 |
|Llama-3| `meta-llama/Meta-Llama-3-8B-Instruct` | 4096 |
|Llama-3| `meta-llama/Meta-Llama-3-70B-Instruct`| 4096 |
|Mistral| `mistralai/Mistral-Nemo-Instruct-2407`  | 4096 |
|Mistral| `mistralai/Mistral-7B-Instruct-v0.3` | 4096 |
|Mixtral| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 4096 |


Note: 
- Cluster type for all models: 8xA100-80G
- Supported context length for models: 512 up to max. context length of each model in powers of 2.

## Summary of Features in Custom Mode

### ✅ Support both Full parameter and LoRA

* LoRA with different configurations, ranks, layers, etc. (Anything supported by huggingface transformers)
* Full-parameter with multi-node training support
    
### ✅ State of the art performance related features:

* Gradient checkpointing
* Mixed precision training
* Flash attention v2
* Deepspeed support (zero-DDP sharding)

### ✅ Unified chat data format with flexible prompt format support enabling finetuning for:


#### Use-case: Multi-turn chat, Instruction tuning, Classification:

Example data format (JSON):
```json
{
    "messages: [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Howdy!"},
        {"role": "user", "content": "What is the type of this model?"},
        # For classification we can define special tokens in the assistant message
        {"role": "assistant", "content": "[[1]]"},
        ...
    ]
}
```

Prompt Format for llama-3-instruct (YAML):

```yaml
system: "<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
user: "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
assistant: "<|start_header_id|>assistant<|end_header_id|>\n\n{instruction}<|eot_id|>"
system_in_user: False
```

#### Use-case: Casual language modeling (aka continued pre-training), custom prompt formats (e.g. Llama-guard):

Example Continued pre-training (JSON):
```json
{
    "messages": [
        # We don't do any formatting, just chunks of text
        {"role": "user", "content": "Once upon a time ..."},
    ],
},
{
    "messages": [
        {"role": "user", "content": "..."},
    ],
}
```

Prompt Format for doing nothing except concatenation:

```yaml
system: "{instruction}"
user: "{instruction}"
assistant: "{instruction}"
system_in_user: False
```

### ✅ Flexible task support: 

* Causal language modeling: Each token predicted based on all past tokens.
* Instruction tuning: Only assistant tokens are predicted based on past tokens.
* Classification: Only special tokens in the assistant message are predicted based on past tokens.
* (Coming soon) Preference tuning: Use the contrast between chosen and rejected messages to improve the model.

### ✅ Support for multi-stage continuous fine-tuning

* Fine-tune on one dataset, then continue fine-tuning on another dataset, for iterative improvements.
* Do continued pre-training on one dataset, then chat-style fine-tuning on another dataset.
* (Coming soon) Do continued pre-training on one dataset followed by iterations of supervised-finetuning and preference tuning on independent datasets.

### ✅ Support for context length extension

* Extend the context length of the model via methods like RoPE scaling.

### ✅ Configurability of hyper-parameters

* Full control over learning hyper-parameters such as learning rate, n_epochs, batch size, etc.

### ✅ Anyscale and third-party integrations

* (Coming soon) Model registery: 
    * SDK for accessing finetuned models for creating automated pipelines 
    * More streamlined deployment flow when finetuned on Anyscale
* Monitoring and observability:
    * Take advantage of standard logging frameworks such as Weights and Biases
    * Use of ray dashboard and anyscale loggers for debugging and monitoring the training process
* Anyscale jobs integration: Use Anyscale's job submission API to programitically submit long-running jobs through LLMForge


## Examples

Here are some examples for default mode and custom mode:

### Default Mode


--------- 
**Task:** 

Fine-tune llama-3-8b-instruct in default mode (LoRA rank 8). Just giving the dataset.

**Command:**
```bash
llmforge anyscale finetune training_configs/default/llama-3-8b/simple.yaml --default
```

**Config:**

```yaml
model_id: meta-llama/Meta-Llama-3-8B-Instruct
train_path: s3://...
valid_path: s3://...
num_epochs: 3
learning_rate: 1e-4    
```


--------- 

**Task:** 

Fine-tune llama-3-8b-instruct in default mode but also control parameters like `learning_rate` and `num_epochs`. 

**Command:**
```bash
llmforge anyscale finetune training_configs/default/llama-3-8b/custom.yaml --default
```

**Config:**

```yaml
model_id: meta-llama/Meta-Llama-3-8B-Instruct
train_path: s3://...
valid_path: s3://...      
```


### Custom

---------
**Task:** 

Fine-tune llama-3-8b-instruct in custom mode (model is supported in default-mode) on 32xA10s (auto mode uses 8xA100-80G).


**Command:** 

```bash
llmforge anyscale finetune training_configs/custom/meta-llama--Meta-Llama-3-8B-Instruct/lora/32xA10.yaml 
```

**Config:**

```yaml
model_id: meta-llama/Meta-Llama-3-8B-Instruct
train_path: s3://...
valid_path: s3://...
num_epochs: 3
learning_rate: 1e-4
deepspeed:
  config_path: configs/deepspeed/zero_3_llama_2_7b.json
worker_resources:
    accelerator: ...
```


---------
**Task:** 

Fine-tune gemma-2-27b in custom mode (model is not supported in default-mode) on 8xA100-80G.


**Command:** 

```bash
llmforge anyscale finetune training_configs/custom/google--gemma-2-27b-it/lora/8xA100-80G.yaml 
```

**Config:**

```yaml
model_id: google/gemma-2-27b-it
train_path: s3://...
valid_path: s3://...
num_epochs: 3
learning_rate: 1e-4
deepspeed:
  config_path: configs/deepspeed/zero_3_llama_2_7b.json
worker_resources:
    accelerator: ...
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


