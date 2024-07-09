# Customize initial weights and prompt format
**⏱️ Time to complete**: 60 minutes

This guide extends the use-case of finetuning base models to showcase how to
1. Bring your own weights - (1) weights of other models similar in architecture to the Llama or Mistral family of models or (2) weights from a previous finetuning run.
2. Customize the chat template/ prompt format - Specify a custom prompt format for formatting input messages to easily fine-tune on _any_ data format.

The Anyscale platform is uniquely suited to support both of these use-cases. This guide assumes you have familiarized yourself with the [basic fine-tuning guide](../../README.md).
We will focus on fine-tuning the [Meta Llama Guard 2 model](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-guard-2/) throughput this cookbook. 


# Table of Contents
1. [Bring your own weights](#bring-your-own-weights)
    - [Bring models of the same architecture](#bring-models-of-the-same-architecture)
        - [Example YAML](#example-yaml)
        - [How do I configure access to my weights in remote storage??](#)
            - [How do I bring my weights to Anyscale?](#how-do-I-bring-my-weights-to-Anyscale-?)
2. [Customizing the prompt format (chat template)](#customizing-the-prompt-format)

# Bring your own weights

In general, you can customize the initial weights in your fine-tuning run through two options in the YAML:
- `initial_base_model_ckpt_path` : Path to the base model weights you wish to start with
- `initial_adapter_model_ckpt_path`: Path to the adapter (LoRA) weights you wish to start with

## Bring models of the same architecture
You can fine-tune a model similar in architecture to the Llama or Mistral family of models to fine-tune on the Anyscale Platform. For example, to fine-tune Llama Guard 2, you can 
specify the model ID and `initial_base_model_ckpt_path` as below:
### Example YAML

```yaml
model_id: meta-llama/Meta-Llama-3-8B
initial_base_model_ckpt_path: s3://my-bucket/llama-guard-2
train_path: s3://air-example-data/gsm8k/train.jsonl
valid_path: s3://air-example-data/gsm8k/test.jsonl
...
```


The overarching idea is that specifying a model ID will provide context to the architecture of the LLM, which will guide how the model will be further trained. It won't necessarily mean that the model that is specified here is the model that will be fine-tuned. For that, we will rely on the weights that are provided. Knowing that Llama 3 8B model (or any other Llama model for that matter) shares the same architecture as the Llama Guard 2 model makes it a suitable choice for the model ID. However, note that this would still use the same chat-templating / prompt formatting as Llama-3 while starting to fine-tune with Llama Guard 2 weights. For the specific case of Llama Guard 2, we need customization even in the prompt format which will be outlined below. 

### How do I configure access to my weights in remote storage?

For models configured for public access, you simply need to add the URI of the location of the model weights in your training YAML. We support loading models stored on S3 (with GCS support coming soon). For private models, you could configure the read permissions for your Workspace to pull from the bucket holding your model weights. Alternatively, you could sync your model weights to your Anyscale-provided artifact storage, or even just keep it in shared storage in your workspace. The [Bring your own data](../../cookbooks/bring_your_own_data/README.md) cookbook, while focusing on datasets, provides in-depth detail on these options.


## Bring checkpoints from a previous finetuning run

This is a similar use case where you want to customize the base model weights to start with or the adapter weights to continue fine-tuning on the Anyscale Platform. 

Further, there are a number of other considerations here (What's the right order of datasets in a 2-stage fine-tuning run? How do differences in context length fit in? etc) all of which are covered in our [continue_from_checkpoint](../continue_from_checkpoint/) cookbook.

# Customizing the prompt format

## How prompt formatting works in `llmforge`

Here's a quick rundown of how prompt formatting/ chat templating works right now: the training/validation data needs to be formatted in the OpenAI messages format. Each example has a "messages" entry consisting a conversation with "system", "user" and "assistant" roles. For each role, depending on the model, we add certain tokens as headers/footers along with a BOS token at the start of the conversation and an EOS token at the end of each assistant response. The prompt format can be specified in our YAML as a part of the `generation_config` for the model (the same format is used in our inference code):

```yaml
generation_config:
  prompt_format:
    system: 
    user: 
    assistant:
    bos: # optional
    system_in_user: # optional
    trailing_assistant:  # optional, inference-only
    default_system_message: # optional
```

### Examples
For `meta-llama/Meta-Llama-3-8B`, we use the following prompt format:
```yaml
generation_config:
  prompt_format:
    system: """"<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>""""
    user: """<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"""
    assistant: """<|start_header_id|>assistant<|end_header_id|>\n\n{instruction}<|eot_id|>"""
    bos: """<|begin_of_text|>""" 
    system_in_user: False
    trailing_assistant: """<|start_header_id|>assistant<|end_header_id|>\n\n""" # inference-only 
    default_system_message: ""
```

For `mistralai/Mistral-7B`, we the below prompt format:
```yaml
generation_config:
  prompt_format:
    system: """"{instruction} + """"
    user: """[INST] {system}{instruction} [/INST]"""
    assistant: """ {instruction}</s>"""
    bos: """<s>""" 
    system_in_user: True
    trailing_assistant: "" # inference-only 
    default_system_message: ""
```

You can find more information on each entry in the [config API reference](https://docs.anyscale.com/reference/finetuning-config-api). Each `system`, `user` and `asistant` entry must contain the format specifier `{instruction}` which will actually format the corresponding entry in `messages`. For example, if a `messages` entry was as follows while training Llama-3-8B:

```json 
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"}, 
    {"role": "user", "content": "What's the value of 1+1?"}, 
    {"role": "assistant", "content": "The value is 2"}
    ]
  }
```

The formatted text input to the model is:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat's the value of 1+1?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe value is 2<|eot_id|>
```

## Customizing data preprocessing and the prompt format

Different LLMs are trained with different prompt formats (or chat templates). In order to figure out the right way to format your data and choose the config parameters, we need to work backwords from the text input to the model. For Llama Guard 2, for instance, the format that the model is trained with is the following:

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
{"system": "<|begin_of_text|>[INST] Task: Check if there is unsafe content in '{{ role }}' messages in conversations according our safety policy with the below categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
{{ unsafe_categories }}
<END UNSAFE CONTENT CATEGORIES>
<BEGIN CONVERSATION>
User: {{ user_message_1 }}
Agent: {{ model_answer_1 }}
<END CONVERSATION>
Provide your safety assessment for ONLY THE LAST {{ role }} in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]", 
"user": "", 
"assistant": "{expected_response}"}
```

Since we've taken care of the full templating ourselves, we just need the prompt formatter to verbatim concatenate the content in different roles. Thus, the generation config can look like:

```yaml
generation_config:
  prompt_format:
    system: """{instruction}"""
    user: """{instruction}"""
    assistant: """{instruction}"""
    bos: "" # optional, empty string by default
    trailing_assistant: ""
```

For the above example, the "instruction" passed in to the `system` template is simply the entire prompt, the "instruction" passed in to the `user` template is empty, and the "instruction" passed in to the `assistant` is the expected response ('safe' or 'unsafe'). Also note that we've specified only one of the many possibilites of `prompt_format` you can specify (with your data preprocessing changing accordingly).


With the change in the base model weights (`initial_base_model_ckpt_path`) and the change in `prompt_format`, you should be able to fine-tune a model like Llama Guard-2.
