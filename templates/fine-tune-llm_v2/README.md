# Fine-tuning Llama-2/Mistral models with Anyscale

**⏱️ Time to complete**: 2.5 hours for 7b models (9 hours for 13b, 25 hours for 70b)

The guide below walks you through the steps required for fine-tuning of LLM models. This template provides an easy to configure solution for ML Platform teams, Infrastructure engineers, and Developers to fine-tune LLMs.

### Supported base models

- mistralai/Mistral-7B-Instruct-v0.1
- mistralai/Mixtral-8x7b
- meta-llama/Llama-2-7b-hf
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-13b-hf
- meta-llama/Llama-2-13b-chat-hf
- meta-llama/Llama-2-70b-hf
- meta-llama/Llama-2-70b-chat-hf

## Step 1 - Launch a fine-tuning job

We have provided different example configurations under the `training_configs`
directory for different base models and accelerator types. You can use these as a
starting point for your own fine-tuning jobs.

[Optional] you can get a WandB API key from [WandB](https://wandb.ai/authorize) to track the finetuning process.

Next, you can launch a fine-tuning job where the WandB API key is passed as an environment variable.




```python
# From the VSCode terminal (press [**Ctrl + `**] in VSCode), run the command to trigger a fine-tuning job. Generally, a fine-tuning job will take a few hours.

# [Optional] You can set the WandB API key to track model performance
# !export WANDB_API_KEY={YOUR_WANDB_API_KEY}

# Launch a full-param fine-tuning job for Llama 7b with 16 A10s
!python main.py training_configs/full_param/llama-2-7b-512-16xa10.yaml

# Launch a LoRA fine-tuning job for Llama 7b with 16 A10s
# !python main.py training_configs/lora/llama-2-7b-512-16xa10.yaml
```

    [2024-03-13 15:34:50,020] [INFO] [real_accelerator.py:158:get_accelerator] Setting ds_accelerator to cuda (auto detect)
    Downloading the tokenizer ...
    RUN(['awsv2', 'configure', 'set', 's3.max_concurrent_requests', '32'])
    RUN(['awsv2', 'configure', 'set', 'default.s3.preferred_transfer_client', 'crt'])
    RUN(['awsv2', 'configure', 'set', 'default.s3.target_bandwidth', '100Gb/s'])
    RUN(['awsv2', 'configure', 'set', 'default.s3.multipart_chunksize', '8MB'])
    RUN(['awsv2', 's3', 'sync', '--no-sign-request', '--region', 'us-west-2', '--exclude', '*', '--include', '*token*', 's3://llama-2-weights/models--meta-llama--Llama-2-7b-chat-hf', '/home/ray/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf'])
    download: s3://llama-2-weights/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1d3cabadba7ec7f1a9ef2ba5467ad31b3b84ff0/tokenizer_config.json to ../../../../.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1d3cabadba7ec7f1a9ef2ba5467ad31b3b84ff0/tokenizer_config.json
    download: s3://llama-2-weights/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1d3cabadba7ec7f1a9ef2ba5467ad31b3b84ff0/special_tokens_map.json to ../../../../.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1d3cabadba7ec7f1a9ef2ba5467ad31b3b84ff0/special_tokens_map.json
    download: s3://llama-2-weights/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1d3cabadba7ec7f1a9ef2ba5467ad31b3b84ff0/tokenizer.json to ../../../../.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1d3cabadba7ec7f1a9ef2ba5467ad31b3b84ff0/tokenizer.json
    download: s3://llama-2-weights/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1d3cabadba7ec7f1a9ef2ba5467ad31b3b84ff0/tokenizer.model to ../../../../.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1d3cabadba7ec7f1a9ef2ba5467ad31b3b84ff0/tokenizer.model
    done
    Tokenizer init done.
    ^C
    
    Aborted!


Depending on whether you are running LoRA or full-param fine-tuning, you can continue with step 2(a) or step 2(b).

## Step 2(a) - Serving the LoRA finetuned model

Upon the job completion, you can see the LoRA weight storage location and model ID in the log, such as the below:

```shell
Note: LoRA weights will also be stored in path s3://anyscale-data-cld-id/org_id/cloud_id/artifact_storage/lora_fine_tuning under meta-llama/Llama-2-7b-chat-hf:sql:12345 bucket.
```

You can specify this URI as the [dynamic_lora_loading_path](../endpoints_v2/examples/lora/DeployLora.ipynb#setup-lora-model-deployment) in the llm serving template, and then query the endpoint.

Note: Model IDs follow the format `{base_model_id}:{suffix}:{id}`

## Step 2(b) - Serving the full-param finetuned model

Once the fine-tuning job is complete, you can view the stored full-param fine-tuned model weight at the very end of the job logs. Here is an example finetuning job output:

```shell
Best checkpoint is stored in:
anyscale-data-cld-id/org_id/cloud_id/artifact_storage/username/llmforge-finetuning/meta-llama/Llama-2-70b-hf/TorchTrainer_2024-01-25_18-07-48/TorchTrainer_b3de9_00000_0_2024-01-25_18-07-48/checkpoint_000000
```

You can follow the [Learn how to bring your own models](../endpoints_v2/examples/CustomModels.ipynb#adding-a-private-model) section under the llm serving template to serve this finetuned model with the specified storage uri.

## Frequently asked questions

### Where can I view the bucket where my LoRA weights are stored?

All the LoRA weights are stored under the URI `${ANYSCALE_ARTIFACT_STORAGE}/lora_fine_tuning` where `ANYSCALE_ARTIFACT_STORAGE` is an environmental variable.

### How can I fine-tune using my own data?

You can open the file under `training_configs` and update `train_path` and `valid_path` to your training and evaluation file.

### How do I customize the fine-tuning job?

You can edit the values, such as `context_length`, `num_epoch`, `train_batch_size_per_device` and `eval_batch_size_per_device` to customize the fine-tuning job.

In addition, the deepspeed configs are provided in case you would
like to customize them.


