# Fine-Tuning with LoRA vs full-parameter

**⏱️ Time to complete**: 60min for LoRA fine-tuning, 2.5h for full-parameter fine-tuning

This guide assumes that you have familiarized yourself with the [main fine-tuning guide](../../README.md) of this template.
In this cookbook, we explain the nuances of fine-tuning with [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) versus full-parameter fine-tuning.

## Quick theoretical comparison

Full-parameter fine-tuning takes the LLM "as is" and trains it on the given dataset. In principle, this is regular supervised training like in the pretraining stage of the LLM. You can expect full-parameter fine-tuning to result in slightly higher model quality.

[LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) is a fine-tuning technique that freezes all the weights of your LLM and adds a few parameters to it that get fine-tuned instead. These additional parameters make up a LoRA checkpoint. There are three important things to take away from this:
1. Since all the original weights are frozen, they don't have to be optimized and therefore don't take up as many resources during fine-tuning. In practice, you can fine-tune on a smaller cluster.
2. Since the checkpoint only consists of the few additional parameters, it is very small. If we load the original model into memory, we can swap out the fine-tuned weights quickly. Therefore, it makes for an efficient scheme for serving many fine-tuned models alongside each other.
3. Optimizing few parameters has a regularization effect - "[it learns less and forgets less](https://arxiv.org/abs/2405.09673)"

You can find a more in-depth analysis of this topic [here](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2).
The domain also has an effect on LoRA's performance. Depending on the domain, it may perform the same or slightly worse than full-parameter fine-tuning.

## How to configure LoRA vs full-parameter fine-tuning jobs

Both fine-tuning techniques require the same dataset format and result in a checkpoint that you can serve with Anyscale's serving template.
Next, we illustrate this by showing you two commands that fine-tune on the same data - once with LoRA, once with full-parameter fine-tuning.
You can look at the respective yaml files to see how they differ in their configuration.

### How to launch a LoRA fine-tuning job


```python
# [Optional] You can set the WandB API key to track model performance
# import os
# os.environ["WANDB_API_KEY"]="YOUR_WANDB_API_KEY"

# Run this command from the base directory of this template
# Fine-tune Llama 3 8B with LoRA
!llmforge anyscale finetune training_configs/lora/llama-3-8b.yaml
```

### How to launch a full-parameter fine-tuning job

We advise against running this end-to-end, because it will take many hours on a default cluster.
If you want to run this full-parameter fine-tuning end-to-end, consider configuring your workspace to use more capable GPUs like A100s or better.


```python
# [Optional] You can set the WandB API key to track model performance
# !export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# Run this command from the base directory of this template
# Fine-tune Llama 3 8B with full-parameter fine-tuning
!llmforge anyscale finetune training_configs/full_param/llama-3-8b.yaml
```

## Comparison of configurable parameters

### LoRA fine-tuninng

In [our blogpost](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2), you'll find information on what the parameters mean. Here is a snippet from the yaml used above with explanatory comments for the most interesting parameters.

```yaml
lora_config:
  # Determines the rank of the matrices that we fine-tune. Higher rank means more parameters to fine-tune. Increasing the rank gives you diminishing returns.
  r: 8
  # Scales the learnt LoRA weights. A value 16 is common practice and is not advised to be fine-tuned.
  lora_alpha: 16
  # Rate at which LoRA weights are dropped out. Can act as a regularizer.
  lora_dropout: 0.05
  # The modules of the LLM that we want to fine-tune with LoRA.
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
    - embed_tokens
    - lm_head
```

You generally don't need to change any of these parameters. In our experience, there is little or nothing to be gained from that.
We advise to fine-tune LoRA with a learning rate of about 1e-4. You can increase it slightly if training is stable enough.
LoRA is rather sensitive to the learning rate. For optimal performance, it's important to target all possible layers with LoRA, while choosing a higher rank gives very minor improvements ([link to paper](https://arxiv.org/abs/2405.09673)).

### Full-parameter fine-tuning

Full-parameter fine-tuning requires the same config as LoRA fine-tuning, but the `lora_config` part should be omited.
We advise to use a learning rate of about 1e-5 here. You can increase it slightly if training is stable enough.

## FAQ

### Should I use LoRA or full-parameter fine-tuning?

There is no general answer to this but here are some things to consider:

- The quality of the fine-tuned models will, in most cases, be comparable if not the same
- LoRA shines if:
    - You want to serve many fine-tuned models at once yourself
    - You want to rapidly experiment (because fine-tuning, downloading and serving the model take less time)
- Full-parameter shines if:
    - You want to make sure that your fine-tuned model has the maximum quality
    - You want to serve only one fine-tuned version of the model

There, you'll also find some guidance on the LoRA parameters and why, in most cases, you don't need to change them.
