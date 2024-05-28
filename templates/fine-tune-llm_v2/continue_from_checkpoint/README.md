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
!python llmforge dev finetune llama-3-8b.yaml
```

## FAQ

### In what order should I fine-tune?

In general: Finish with the dataset that is closest to what you want during inference.
If you are extending the context of the model beyond it's native context length, you should start with the smallest context length end with the largest.