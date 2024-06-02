# Continue fine-tuning from a previous checkpointing

**⏱️ Time to complete**: 40 minutes

This guide assumes that you have familiarized yourself with the main fine-tuning guide of this template.
In this cookbook tutorial, we showcase how a checkpoint that was created earlier can be used as initialization for another round of fine-tuning.
This allows us to sequentially combine fine-tuning on multiple datasets in order to get performance boost on the final task that we care about. 

We support both Full-parameter checkpoints, and LoRA-adapter checkpoints. However, we recommend not combining the two by training a full-parameter model followed by a LoRA adaptation, because serving the resulting LoRA adapter will require the full-parameter checkpoint. Unless you are fine-tuning many such LoRA adaptors for different tasks this serving architecture does not have the neither the economical benefits of LoRA nor the quality benefits of full-parameter.

## How to fine-tune from a previous checkpointing
To get started, we can run the following illustrative example:

```python
# [Optional] You can set the WandB API key to track model performance
# !export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# Continue LoRA fine-tuning on the GSM8k dataset with Llama 3 8B
!llmforge dev finetune llama-3-8b.yaml
```

## What we are fine-tuning on

Running the above command will fine-tune on the [GSM8k dataset](https://huggingface.co/datasets/gsm8k). 
In this example, we split the dataset into two halfs, each consisting of approximately 4.000 samples.
The provided initial checkpoint has been trained on the first half and is already good at solving GSM8k. By running the above command, you continue fine-tuning from the provided checkpoint with the second half.

Note the following evaluation losses. The first graph shows the evaluation loss on three epochs of training on the first half of the GSM8k dataset.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/fine-tune-llm_v2/cookbooks/continue_from_checkpoint/../../assets/3epochs_1st_dataset.png" alt="evaluation loss of 1st training" width="700"/>

The second graph shows the evaluation loss on three epochs of training on the second half, starting with the fine-tuned weights of the first training.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/fine-tune-llm_v2/cookbooks/continue_from_checkpoint/../../assets/3epochs_2nd_dataset.png" alt="evaluation loss of 2nd training" width="700"/>

Note that the evaluation loss starts way lower than where it starts or finishes in the first training.

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
If we were not fine-tuning with LoRA, we would not configure `initial_adapter_model_ckpt_path`, but `initial_base_model_ckpt_path` instead.

# How to use this for your own purpose

The training and evaluation loss of the second, the continued, fine-tuning are lower than what we saw in the first run.
For example, the checkpoint that you find in the llama-3-8b.yaml has an evaluation loss of 0.8886.
After continued fine-tuning, we reach a checkpoint with an evaluation loss of 0.8668.
Such loss values depend greatly on the task at hand - a difference of 0.0218 may be a big improvement on some tasks and a minor improvement on others.

We advise to monitor training loss and evaluation loss of fine-tunes to find out if you are improving through the continued fine-tuning.

## FAQ

### In what order should I fine-tune?

In general: Finish with the dataset that is closest to what you want during inference.
If you are extending the context of the model beyond it's native context length, you should start with the smallest context length end with the largest.

### Should I extend the dataset samples or replace them with new ones when I continue fine-tuning

This depends on your task and how many epochs have already been trained. If in doubt, you can always watch the training and evaluation loss to see if you are overfitting.

### How can I fine-tune a model that I fine-tuned on Anyscale Endpoints?

You have to download the model weights through Anyscale Endpoints, upload them to a bucket of your choice and reference the bucket as an initial checkpoint in the training config yaml.


