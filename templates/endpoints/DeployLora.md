# Serving LoRA Models

We support serving multiple LoRA adapters with a common base model in the same request batch which allows you to serve a wide variety of use-cases without increasing hardware spend. In addition, we multiplex LoRA adapters to reduce the number of swaps. There is a slight latency overhead to serving a LoRA model compared to the base model, typically 10-20%.

# Quick Start

The files `lora-serve.yaml` and `lora-query.py` are provided for you in this template. You can run our example that we fine-tuned on the viggo dataset as follows:

> Before you start: Open `lora-serve.yaml` and modify `HUGGING_FACE_HUB_TOKEN` with your own token.

To deploy the model, run:
```shell
serve run lora-serve.yaml
# Leave this tab open
```
To dynamically load a LoRA adapter and query the LoRA model, run:
```shell
python lora-query.py
```

This should give you an output such as:

```shell
inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+ (for Everyone 10 and Older)], genres[driving/racing, sport], platforms[PlayStation, Xbox, PC], available_on_steam[no], has_linux_release[no], has_mac_release[no])
```

# Modifying your LoRA Model Deployment

In order to deploy your own LoRA adapters you need to update the following fields in `lora-serve.yaml`:

1. `dynamic_lora_loading_path`
    - The base artifact storage path of your LoRA checkpoints.
    - The path to the checkpoints must be in the following format: `{dynamic_lora_loading_path}/{base_model_id}:{suffix}:{id}`, e.g. `s3://my-bucket/my-lora-checkouts/meta-llama/Llama-2-7b-chat-hf:lora-model:1234`.
    - The models can be loaded from any accessible AWS S3 or Google Cloud Storage bucket. You can use an existing bucket where you have the LoRA models or can upload the models to the `$ANYSCALE_ARTIFACT_STORAGE` already provided by Anyscale Workspace.
    - New models can be uploaded to the `dynamic_lora_loading_path` dynamically before or after the Serve application is launched.
2. `multiplex_models
    - The path to the YAML for the base model deployment configuration.
3. `HUGGING_FACE_HUB_TOKEN`
    - This is needed for Llama 2 models


# Querying the default query

You can update `lora-query.py` with your own query or query other models. To query other models, modify the `MODEL` field used in `lora-query.py`.

> Note: LoRA model IDs must always follow this format: `{base_model_id}:{suffix}:{id}`. (e.g. `meta-llama/Llama-2-7b-chat-hf:lora-model:1234`)

You can also directly query the base model by changing the `MODEL` variable to the base model id.
