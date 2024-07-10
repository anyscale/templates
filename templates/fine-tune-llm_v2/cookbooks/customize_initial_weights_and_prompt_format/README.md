# Customize initial weights and prompt format
**⏱️ Time to complete**: 60 minutes

This guide will showcase how you can finetune a model with a similar architecture to the Llama or Mistral family and customize the chat template or prompt format. We will focus on fine-tuning the [Meta Llama Guard 2 model](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-guard-2/) throughput this cookbook. 

The two capabilities showcased here are
1. Bringing your own weights - (1) weights of other models similar in architecture to the Llama or Mistral family of models or (2) weights from a previous finetuning run. While we focus on (1) here, (2) is an important use-case of multi-step fine-tuning covered in depth in the cookbook [here](../continue_from_checkpoint/).
2. Customizing the chat template or prompt format - Specify a custom prompt format for formatting input messages to easily fine-tune on _any_ data format.

The Anyscale platform is uniquely suited to support both of these use-cases. This guide assumes you have familiarized yourself with the [basic fine-tuning guide](../../README.md).


# Table of Contents
1. [Bring your own weights](#bring-your-own-weights)
    - [Bring models of the same architecture](#bring-models-of-the-same-architecture)
        - [Example YAML](#example-yaml)
        - [How do I configure access to my weights in remote storage??](#)
            - [How do I bring my weights to Anyscale?](#how-do-I-bring-my-weights-to-Anyscale-?)
    - [Bring checkpoints from a previous finetuning run](#bring-checkpoints-from-a-previous-finetuning-run)
2. [Customizing the prompt format (chat template)](#customizing-the-prompt-format)
    - [How prompt formatting works in `llmforge`](#how-prompt-formatting-works-in-llmforge)
    - [Customizing data preprocessing and the prompt format](#customizing-data-preprocessing-and-the-prompt-format)

# Bring your own weights

In general, you can customize the initial weights in your fine-tuning run through two options in the YAML:
- `initial_base_model_ckpt_path` : Path to the base model weights you wish to start with
- `initial_adapter_model_ckpt_path`: Path to the adapter (LoRA) weights you wish to start with

Note that you can use the above parameters independent of one another. 

## Bring models of the same architecture
You can fine-tune a model similar in architecture to the Llama or Mistral family of models to fine-tune on the Anyscale Platform. For example, [Llama-Guard-2](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-guard-2/) is a model that is based on Llama-3-8B architecture that has been finetuned on a specific task of classifying human-AI conversations. To fine-tune Llama Guard 2, you can specify the model ID and `initial_base_model_ckpt_path` as below:

### Example YAML

```yaml
model_id: meta-llama/Meta-Llama-3-8B
initial_base_model_ckpt_path: s3://my-bucket/llama-guard-2
...
```


The overarching idea is that specifying a model ID will provide context to the architecture of the LLM, which will guide how the model will be further trained. It won't necessarily mean that the model that is specified here is the model that will be fine-tuned. For that, we will rely on the weights that are provided. Knowing that Llama 3 8B model (or any other Llama model for that matter) shares the same architecture as the Llama Guard 2 model makes it a suitable choice for the model ID. However, note that this would still use the same chat-templating / prompt formatting as Llama-3 while starting to fine-tune with Llama Guard 2 weights. For the specific case of Llama Guard 2, we need customization even in the prompt format which will be outlined below. 

### How do I configure access to my weights in remote storage?

For models configured for public access, you simply need to add the URI of the location of the model weights in your training YAML. We support loading models stored on S3 (with GCS support coming soon). For private models, you could configure the read permissions for your Workspace to pull from the bucket holding your model weights. Alternatively, you could sync your model weights to your Anyscale-provided artifact storage, or even just keep it on disk in shared storage in your workspace. The [Bring your own data](../../cookbooks/bring_your_own_data/README.md) cookbook, while focusing on datasets, provides in-depth detail on these options.


## Bring checkpoints from a previous finetuning run

This is the use case  of multi-step fine-tuning where you want to customize the base model weights to start with or the adapter weights to continue fine-tuning on the Anyscale Platform. Being able to provide the custom checkpoints for the second (or later) stage of fine-tuning is just one part of the equation. There are a number of other considerations here (What's the right order of datasets in a 2-stage fine-tuning run? How do differences in context length fit in? etc) all of which are covered in our [continue from checkpoint](../continue_from_checkpoint/) cookbook.

# Customizing the prompt format

## How prompt formatting works in `llmforge`

Here's a quick rundown of how prompt formatting or chat templating works: the training or validation data needs to be formatted in the OpenAI messages format. Each example has a "messages" entry consisting a conversation with "system", "user" and "assistant" roles. For example:

```json 
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"}, 
    {"role": "user", "content": "What's the value of 1+1?"}, 
    {"role": "assistant", "content": "The value is 2"}
    ]
  }
```

For each role, depending on the model, we add certain tokens as headers or footers along with a BOS token at the start of the conversation and an EOS token at the end of each assistant response. This templating/ formatting is a crucial preprocessing step in bringing the conversation format into a plain text input - which later tokenized and fed into the model. For Llama-3-8B, the above example would be formatted as follows:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat's the value of 1+1?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe value is 2<|eot_id|>
```

The prompt format can be specified in our YAML as a part of the `generation_config` for the model (the same format is used in our inference code):

```yaml
generation_config:
  prompt_format:
    system: 
    user: 
    assistant:
    trailing_assistant:  # inference-only
    bos: # optional
    system_in_user: # optional
    default_system_message: # optional
```

For the models in the [list of supported models](../../README.md#faqs), we have default generation config parameters. This means that `generation_config` need not be specified when you just want to finetune a model like `meta-llama/Meta-Llama-3-8B-Instruct` directly.  

### Examples
For `meta-llama/Meta-Llama-3-8B`, we use the following prompt format:
```yaml
generation_config:
  prompt_format:
    system: "<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
    user: "<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
    assistant: """<|start_header_id|>assistant<|end_header_id|>\n\n{instruction}<|eot_id|>"
    trailing_assistant: "<|start_header_id|>assistant<|end_header_id|>\n\n" # inference-only 
    bos: "<|begin_of_text|>"
    system_in_user: False
    default_system_message: ""
```

For `mistralai/Mistral-7B`, we the below prompt format:
```yaml
generation_config:
  prompt_format:
    system: "{instruction} + "
    user: "[INST] {system}{instruction} [/INST]"
    assistant: " {instruction}</s>"
    trailing_assistant: "" # inference-only 
    bos: "<s>"
    system_in_user: True
    default_system_message: ""
```

You can find more information on each entry in the [config API reference](https://docs.anyscale.com/reference/finetuning-config-api). Each `system`, `user` and `asistant` entry must contain the format specifier `{instruction}` which will actually format the corresponding entry in `messages`.  

## Customizing data preprocessing and the prompt format

Different LLMs are trained with different prompt formats (or chat templates). In order to figure out the right way to format your data and choose the config parameters, we need to work backwards from the text input to the model. For Llama Guard 2, for instance, the format that the model is trained with is the following:

```text
<|begin_of_text|>[INST] Task: Check if there is unsafe content in '{{ role }}' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{{ unsafe_categories }}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: {{ user_message_1 }}

Agent: {{ model_answer_1 }}

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST {{ role }} in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]
```

This is wildly different from Llama-3's prompt format. Fortunately, the OpenAI conversation format is highly flexible and adaptable to almost any input. In a case like Llama Guard 2, it is beneficial to take control of the templating yourself and format your inputs beforehand in the data preprocessing stage. For example, here's one way your data can look:

```json
{
  "messages": [
    {"role": "system", 
    "content": 
    "<|begin_of_text|>[INST] Task: Check if there is unsafe content in '{{ role }}' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
{{ unsafe_categories }}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: {{ user_message_1 }}

Agent: {{ model_answer_1 }}

<END CONVERSATION>

",
  }, 
    {"role": "user", "content":  "Provide your safety assessment for ONLY THE LAST {{ role }} in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"}, 
    {"role": "assistant", "content": "{expected_response}"}
    ]
  }
```

Note: All the entries in the messages list need to have non-empty content, and at a minimum we expect one user and one assistant message. 

Since we've taken care of the full templating ourselves, we just need the prompt formatter to verbatim concatenate the content in different roles. Thus, the generation config can look like:

```yaml
generation_config:
  prompt_format:
    system: "{instruction}"
    user: "{instruction}"
    assistant: "{instruction}<|end_of_text|>"
    trailing_assistant: ""
    bos: "" # optional, empty string by default
```

You can see how we make use of format specifiers to format the input chat. For the above example, the "instruction" (format specifier) passed in to the `system` template is almost the entire prompt (mainly problem context), the "instruction" passed in to the `user` template contains the specific instructions for the assistant, and the "instruction" passed in to the `assistant` template is the expected response ('safe' or 'unsafe'). Also note that this is only one of the many possibilites of `prompt_format` you can specify (with your data preprocessing changing accordingly). 


With the change in the base model weights (`initial_base_model_ckpt_path`) and the change in `prompt_format`, you should be able to fine-tune a model like Llama Guard-2. An example YAML is provided in [llama-guard-2.yaml](./llama-guard-2.yaml). We've preprocessed [nvidia/Aegis-AI-Content-Safety-Dataset-1.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0?row=0) to fine-tune Llama Guard 2. To get started, run 

```python
llmforge anyscale finetune cookbooks/customize_initial_weights_and_prompt_format/llama-3-8b.yaml 
```


## Inference time behaviour

After customizing the prompt format during fine-tuning, you need to make sure that the same format is being used at inference. You can use the [inference template](https://docs.anyscale.com/examples/deploy-llms) to deploy your fine-tuned model and specify the same prompt format parameters under  the `generation` entry in the YAML. 
