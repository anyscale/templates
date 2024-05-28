# Continue fine-tuning from a previous checkpointing

This document assumes that you have familiarized yourself with the main fine-tuning guide of this template.
In this folder of the template, we showcase how a checkpoint that was created earlier can be used to start a training from.
We case use this, for example, if we think that starting from a given checkpoint will give us a performance advantage.

There are two types of checkpoints to considere here: Full-parameter checkpoints, and LoRA-adapter checkpoints.
For starters, we advise against combining the two (by training a LoRA adapter ontop of a full-parameter checkpoint), because serving the resulting LoRA adapter will require the full-parameter checkpoint.

## How to fine-tune from a previous checkpointing

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