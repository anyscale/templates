# Bring your own weights 
**⏱️ Time to complete**: 60 minutes

This guide extends the use-case of finetuning base models to showcase how you can bring (1) weights of other models similar in architecture to the Llama or Mistral family of models or (2) weights of models that have already been finetuned. The Anyscale platform is uniquely suited to support both of these use-cases. This guide assumes you have familiarized yourself with the [basic fine-tuning guide](../../README.md).

The underlying principle of these use-cases is similar - using existing model weights or checkpoints for finetuning. This branches out from the basic fine-tuning guide in that it no longer demands the base model as a starting point in our templates.

# Table of Contents
1. [Bring models of the same architecture](#bring-models-of-the-same-architecture)
    - [Exampe YAML](#example-YAML)
    - [What do I need to change?](#what-do-I-need-to-change-?)
        - [Specifying the right model ID and prompt format](#specifying-the-right-model-ID-and-prompt-format)
        - [How do I bring my weights to Anyscale?](#how-do-I-bring-my-weights-to-Anyscale-?)
2. [Bring models that have already been finetuned](#bring-models-that-have-already-been-finetuned)
    - [How to fine-tune from a previous checkpoint](#how-to-fine-tune-from-a-previous-checkpoint)
    - [What and how are we fine-tuning?](#what-and-how-are-we-fine-tuning-?)
    - [Things to Notice](#things-to-notice)
    - [FAQs](#FAQs)


# Bring models of the same architecture

This guide focuses on how you can bring weights of a model similar in architecture to the Llama or Mistral family of models to fine-tune on the Anyscale Platform. Specifically, we will fine-tune the [Meta Llama Guard 2 model](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-guard-2/).

## Example YAML

```yaml
model_id: meta-llama/Meta-Llama-3-8B
initial_base_model_ckpt_path: s3://my-bucket/llama-guard-2
train_path: s3://air-example-data/gsm8k/train.jsonl
valid_path: s3://air-example-data/gsm8k/test.jsonl
generation_config:
  prompt_format:
    system: """{instruction}"""
    user: """{instruction}"""
    assistant: """{instruction}"""
...
```

## What do I need to change?

Along with the training and validation file paths, we specify the model that has the same architecture `model_id` and point to the location of the model weights in an S3 bucket `initial_base_model_ckpt_path`. Additionally, given that this model could have a different prompt format, we add it to the configuration YAML as `prompt_format `.

### Specifying the right model ID and prompt format

The overarching idea is that specifying a model ID will provide context to the architecture of the LLM, which will guide how the model will be further trained. It won't necessarily mean that the model that is specified here is the model that will be fine-tuned. For that, we will rely on the weights that are provided. Knowing that Llama 3 8B model (or any other Llama model for that matter) shares the same architecture as the Llama Guard 2 model makes it a suitable choice for the model ID.

Different LLMs are trained with different prompt formats. Leveraging this format and continuing with it make the finetuning more effective. For Llama Guard 2, for instance, the format that data is trained with is the following:

```<|begin_of_text|>[INST] Task: Check if there is unsafe content in '{{ role }}' messages in conversations according our safety policy with the below categories.
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

The format of your training data needs to map to the one above in order to maintain consistency and yield the best results from the finetuning run. Meaning, the data format that Anyscale's fine-tuning template demands (documented [here](https://docs.anyscale.com/endpoints/fine-tuning/dataset-prep/)) needs to be converted to this format. Fortunately, the syster/user/assistant format is highly flexible and adaptable to almost any prompt format.

If the starting point is the Anyscale data format and the ending point is the Llama Guard 2 format, the following is one of the conversion schemas that you could apply:
```
  system: """{instruction}"""
  user: """{instruction}"""
  assistant: """{instruction}"""
```
where the instruction passed in to the system is simply the entire prompt, the instruction passed in to the user is empty, and the instruction passed in to the assistant is 'safe' or 'unsafe'.

The idea of this prompt formatter is to simply map the input data format of Anyscale's fine-tuning template to the format corresponding to the model needing to be fine-tuned.

### How do I bring my weights to Anyscale?

For models configured for public access, you simply need to add the URI of the location of the model weights in your training YAML. We support loading models stored on S3 and GCS. For private models, you could configure the read permissions for your Workspace to pull from the bucket holding your model weights. Alternatively, you could sync your model weights to your Anyscale-provided storage. The [Bring your own data](../../cookbooks/bring_your_own_data/README.md) cookbook provides in-depth detail on these options.

# Bring models that have already been finetuned

This guide showcases how a checkpoint that was created earlier can be used as initialization for another round of fine-tuning. Specifically, we will fine-tune a fine-tuned [Meta Llama 3 8B model](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3). This allows us to sequentially combine fine-tuning on multiple datasets in order to get a performance boost on the final task that we care about.

## How to fine-tune from a previous checkpoint

To get started, we can run the following illustrative example. Run this command from where `main.py` is located.

```
python main.py cookbooks/continue_from_checkpoint/llama-3-8b.yaml
```

Running the above command will fine-tune on the [GSM8k dataset](https://huggingface.co/datasets/gsm8k). 
In this example, we have splited the dataset into two halves, each consisting of approximately 4,000 samples.
The provided initial checkpoint has been trained on the first half and is already good at solving GSM8k. By running the above command, you continue fine-tuning from the provided checkpoint with the second half.

Note the following evaluation losses. The first three epochs of training where run on the first half of the GSM8k dataset. The second three epochs of training where run on the second half.

<img src="./assets/continue_ft.png" alt="evaluation losses" width="600"/>

Note that on the first iteration of the second training (epoch 4), the evaluation loss starts off much lower than in the first training.


## What and how are we fine-tuning?

The following is a snippet from the `llama-3-8b.yaml` file we use above. 

```yaml
# ...
model_id: meta-llama/Meta-Llama-3-8B-Instruct
# initial_base_model_ckpt_path: ...
initial_adapter_model_ckpt_path: s3://large-dl-models-mirror/finetuning_template/continued_ft_gsm8k_checkpoint
train_path: s3://large-dl-models-mirror/finetuning_template/train_2.jsonl
# ...
```

We fine-tune Llama 3 8B Instruct, but the initial weights of the LoRA adapter are loaded from our s3 mirror.
It makes sense to keep those weights in a bucket so that they can be accessed from all nodes of your cluster.
The train path `(.../train_2.jsonl)` points to the second part of the GSM8k dataset that we fine-tune on.
If we wanted to continue the finetuning of a full-parameter checkpoint, we should configure `initial_base_model_ckpt_path` instead of `initial_adapter_model_ckpt_path`. 

## Things to Notice

When comparing the training and evaluation loss of the second (continued) fine-tuning with the first run, you'll notice that the values are lower.
For instance, the checkpoint in the llama-3-8b.yaml has an evaluation loss of 0.8886.
After continued fine-tuning, we achieve a checkpoint with an evaluation loss of 0.8668.
It's important to note that the significance of such loss values varies greatly depending on the task at hand. A difference of 0.0218 may represent a substantial improvement for some tasks, while it may only be a minor improvement for others.

To determine whether continued fine-tuning is beneficial for your specific task, we recommend monitoring the training and evaluation loss during the fine-tuning process.
This will help you assess the impact of the additional fine-tuning on your model's performance.


## FAQs

### In what order should I fine-tune?

In general: Finish with the dataset that is closest to what you want during inference.
If you are extending the context of the model beyond its native context length, you should start with the smallest context length end with the largest.

### Should I extend the dataset samples or replace them with new ones when I continue fine-tuning?

This depends on your task and how many epochs have already been trained. If in doubt, you can always watch the training and evaluation loss to see if you are overfitting.

### How can I fine-tune a model that I fine-tuned on Anyscale Endpoints?

You have to download the model weights through the `Serving` page, upload them to a bucket of your choice and reference the bucket as an initial checkpoint in the training config yaml.

<img src="./assets/download.png" alt="downloading the model weights" width="500"/>

### Can I combine a full-parameter finetuned model with PEFT, or vice-versa?

We support both Full-parameter checkpoints and LoRA-adapter checkpoints. However, we recommend not to combine the two by training a full-parameter model followed by a LoRA adaptation. Serving the resulting LoRA adapter will require the base full-parameter checkpoint. Unless you are fine-tuning many such LoRA adaptors for different tasks, this serving architecture does not have the neither the economical benefits of LoRA nor the quality benefits of full-parameter.