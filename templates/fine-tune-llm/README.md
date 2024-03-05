# Fine-tuning Llama-2/Mistral models with Anyscale

Anyscale currently offers a simple CLI command to fine-tune LLM models via `anyscale fine-tuning submit`
and you can take a look at the documentation [here](https://docs.anyscale.com/endpoints/fine-tuning/get-started).
This guide provides starter configurations if you would like to further customize the fine-tuning process.

### Supported base models

- mistralai/Mistral-7B-Instruct-v0.1
- mistralai/Mixtral-8x7b
- meta-llama/Llama-2-7b-hf
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-13b-hf
- meta-llama/Llama-2-13b-chat-hf
- meta-llama/Llama-2-70b-hf
- meta-llama/Llama-2-70b-chat-hf

# Step 1 - Launch a fine-tuning job

We have provided different example configurations under the `training_configs`
directory for different base models and instance types. You can use these as a
starting point for your own fine-tuning jobs.

[Optional] you can get a WandB API key from [WandB](https://wandb.ai/authorize) to track the finetuning process.

Next, you can launch a fine-tuning job where the WandB API key is passed as an environment variable.

```shell
# Launch a full-param fine-tuning job for Llama 7b with 16 g5.4xlarge instances
WANDB_API_KEY={YOUR_WANDB_API_KEY} python train.py job_compute_configs/aws.yaml training_configs/full_param/llama-2-7b-512-16xg5_4xlarge.yaml

# Launch a LoRA fine-tuning job for Llama 7b with 16 g5.4xlarge instances
WANDB_API_KEY={YOUR_WANDB_API_KEY} python train.py job_compute_configs/aws.yaml training_configs/lora/llama-2-7b-512-16xg5_4xlarge.yaml
```

Once you submit the command, in the terminal you will see the link to your finetuning job. Ex:

```
View the job in the UI at https://console.anyscale.com/jobs/prodjob_62is21vur3fwl5y5xkc9u1t3ll
```

You can now monitor the progress of the job in the provided job link. Generally a full-param fine-tuning job will take a few hours.

Depending on whether you are running LoRA or full-param fine-tuning, you can continue
with step 2(a) or step 2(b).

# Step 2(a) - Serve the LoRA finetuned model

Upon the job submission, you can see the LoRA weight storage location and model ID in the log, such as the below:

```shell
Note: Lora weights will also be stored in path s3://anyscale-data-cld-id/org_id/cloud_id/artifact_storage/lora_fine_tuning under meta-llama/Llama-2-7b-chat-hf:sql:12345 bucket.
```

Once the job is complete, you can deploy the _base model_ on Endpoints by creating a
new endpoint or adding it to an existing endpoint
([docs](https://docs.anyscale.com/endpoints/model-serving/get-started#deploy-an-anyscale-private-endpoint)).
Then you can enable LoRA serving with the step
[here](https://docs.anyscale.com/endpoints/model-serving/model-configuration#serving-lora-weights).
Therefore, when you query the endpoint, the endpoint will load the LoRA adapter from the
stored bucket dynamically.

You can follow the endpoints page guide to query the endpoint
([docs](https://docs.anyscale.com/endpoints/model-serving/get-started#step-4-query-the-model)).
By updating the model ID to the LoRA model ID, you can query the LoRA model.

Note: Model IDs follow the format `{base_model_id}:{suffix}:{id}`

# Step 2(b) - Serve the full-param finetuned model. Import the model

Once the fine-tuning job is complete, you can view the stored full-param fine-tuned model weight at the very end of the job logs. Here is an example finetuning job output:

```shell
Best checkpoint is stored in:
anyscale-data-cld-id/org_id/cloud_id/artifact_storage/username/llmforge-finetuning/meta-llama/Llama-2-70b-hf/TorchTrainer_2024-01-25_18-07-48/TorchTrainer_b3de9_00000_0_2024-01-25_18-07-48/checkpoint_000000
```

You can go to models page and import this model by clicking the `Import` button.
The checkpoint uri points to a remote bucket location where the model
configs, weights and tokenizers are stored.
When entering the remote uri, please make sure to add the
prefix `s3://` or `gs://`.

For the generation config, you can reference example configs
[here](https://docs.anyscale.com/endpoints/model-serving/import-model#generation-configuration-examples).

# Step 3 - Deploy the full-param finetuned model on Endpoints

Once the model is imported, you can deploy it on Endpoints by creating a
new endpoint or adding it to an existing endpoint. You can follow the
endpoints page guide to query the endpoint ([docs](https://docs.anyscale.com/endpoints/model-serving/get-started#step-4-query-the-model)).

# Frequently asked questions

### Where can I view the bucket where my LoRA weights are stored?

All the lora weights are stored under the uri `${ANYSCALE_ARTIFACT_STORAGE}/lora_fine_tuning`
where `ANYSCALE_ARTIFACT_STORAGE` is an environmental variable.

### How can I serve the LoRA weights via the serving template?

When you run the fine-tuning job, you should get a checkpoint uri
from the log `Note: Lora weights will also be stored in path s3://anyscale-data-cld-id/org_id/cloud_id/artifact_storage/lora_fine_tuning under meta-llama/Llama-2-7b-chat-hf:sql:12345 bucket.`.

Upon job success, you can specify this URI as the `dynamic_lora_loading_path` in the serving
workspace template.

### How can I fine-tune using my own data?

You can open the file under `training_configs` and update
`train_path` and `valid_path` to your training and evaluation file.

### How do I customize the fine-tuning job?

You can edit the values, such as `context_length`, `num_epoch`,
`train_batch_size_per_device` and `eval_batch_size_per_device`
to customize the fine-tuning job.

In addition, the deepspeed configs are provided in case you would
like to customize them.

### What if I want to use a different instance type?

You can edit both job and training configs to use
a different instance type. Note that the `num_devices` field
under the `training_configs` file would need
to be updated to be the total of GPUs that you expect to use.
For instance, if you expect to fine-tune a model with 4 g5.12xlarge,
the `num_devices` should be 16.
