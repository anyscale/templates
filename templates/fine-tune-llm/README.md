# Fine-tuning Llama-2/Mistral models with Anyscale

Anyscale currently offers a simple CLI command to fine-tune LLM models via `anyscale fine-tuning submit`
and you can take a look at the documentation [here](https://docs.anyscale.com/endpoints/fine-tuning/get-started).
This guide provides starter configurations if you would like to further customize the fine-tuning process.

### Supported base models

- mistralai/Mistral-7B-Instruct-v0.1
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

```shell
# Launch a fine-tuning job for Llama 7b with 16 g5.4xlarge instances

llmforge dev launch job_configs/aws.yaml training_configs/llama-2-7b-512-16xg5_4xlarge.yaml.yaml
```

Once you submit the command, you can monitor the progress of the job in
the provided job link. Generally a full-param fine-tuning job will take a few hours.

# Step 2 - Import the model

Once the fine-tuning job is complete, you can view the stored model weight at the very end of the job logs. Here is an example finetuning job output:

```shell

Best checkpoint is stored in:
anyscale-data-cld-id/org_id/cloud_id/artifact_storage/username/llmforge-finetuning/meta-llama/Llama-2-70b-hf/TorchTrainer_2024-01-25_18-07-48/TorchTrainer_b3de9_00000_0_2024-01-25_18-07-48/checkpoint_000000
```

You can go to models page and import this model by clicking the `Import` button.
When entering the remote uri, please make sure to add the
prefix `s3://` or `gs://`.

For the generation config, you can reference example configs
[here](https://docs.anyscale.com/endpoints/model-serving/import-model#generation-configuration-examples).

# Step 3 - Deploy the model on Endpoints

Once the model is imported, you can deploy it on Endpoints by creating a
new endpoint or adding it to an existing endpoint. You can follow the
endpoints page guide to query the endpoint.
