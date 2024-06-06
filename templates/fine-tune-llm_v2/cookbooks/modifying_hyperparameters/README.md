# Modifying hyperparameters

**⏱️ Time to complete**: 10 minutes

This guide will focus on how you can customize your fine-tuning run by modifying the various hyperparameters configurable. Make sure you've read the [basic fine-tuning guide](../../README.md) for better context. 

We provide a number of options to configure via the training YAML.


## GPU resources

Configuring GPU resources to be used is one of the most important pre-requisities for training. There are two fields in our YAML that are relevant:

```yaml
num_devices: 16 # number of GPUs 
worker_resources:
  accelerator_type:A10G: 0.001 # specifies GPU type available, and a minimum allocation per worker
```

Internally, our fine-tuning code will launch Ray workers with each being allocated one GPU. The cluster will be auto-scaled if needed to meet the requirements. The different GPU types you can specify can depend on the specific Anyscale Cloud. The value you specify for the accelerator type does not matter much, as long as it's non-zero (so that each worker is allocated a GPU) and less than or equal to 1 (so that the requested number of GPUs is the same as `num_devices`).

## Learning rate

There are two entities of interest here: the actual learning rate value itself and the particular learning rate scheduler you use. The parameters you can control in the YAML are below: 

```yaml
learning_rate: 1e-4
lr_scheduler_type: cosine
num_warmup_steps: 10
```

In the above config, the training run would use a cosine learning rate schedule (the default) with an initial warmup of 10 steps (the default). The peak learning rate would be 1e-4 (the value specified). 

We support both `'linear'` and `'cosine'` schedules. 

## Batch size

The batch size for training and validation depends on the below parameters:

```yaml
num_devices: 8
train_batch_size_per_device: 16
eval_batch_size_per_device: 16
```
The effective batch size for training would be `train_batch_size_per_device * num_devices`. For the hardware you specify, the amount you can push `train_batch_size_per_device` / `eval_batch_size_per_device` depends on dataset statistics (average sequence length) and the context length used. For a context length of 512 and the default NVIDIA A10 GPUs, the per-device batch size of 16 is a good default. 

## LoRA configs
We support all the LoRA parameters you can configure in [🤗PEFT](https://huggingface.co/docs/peft/en/package_reference/lora#peft.LoraConfig):

```yaml
lora_config:
  r: 8
  lora_alpha: 16
  lora_dropout: 0.05
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
  task_type: "CAUSAL_LM"
  modules_to_save: []
  bias: "none"
  fan_in_fan_out: false
  init_lora_weights: true
```


