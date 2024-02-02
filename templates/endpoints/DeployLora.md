# Serving LoRA Models

We support serving multiple LoRA adapters with a common base model in the same request batch which allows you to serve a wide variety of use-cases without increasing hardware spend. In addition, we use Serve multiplexing to reduce the number of swaps for LoRA adapters. There is a slight latency overhead to serving a LoRA model compared to the base model, typically 10-20%. 

# Setup LoRA Model Deployment

`lora-serve.yaml` and `lora-query.py` are provided for you in this template. 

In order to deploy LoRA adapters you would need to update `lora-serve.yaml`:
1. `dynamic_lora_loading_path` - The LoRA checkpoints are loaded from the artifact storage path specified in `dynamic_lora_loading_path`. The path to the checkpoints must be in the following format: `{dynamic_lora_loading_path}/{base_model_id}:{suffix}:{id}`, e.g. `s3://my-bucket/my-lora-checkouts/meta-llama/Llama-2-7b-chat-hf:lora-model:1234`. The models can be loaded from any accessible AWS S3 or Google Cloud Storage bucket. You can use an existing bucket where you have the LoRA models or can upload the models to `$ANYSCALE_ARTIFACT_STORAGE` already provided by Anyscale Workspace. New models can be uploaded to the `dynamic_lora_loading_path` dynamically before or after the Serve application is launched.
2. The config YAML for the base model and LoRA adapters should be passed in the `multiplex_models` config field.
3. `HUGGING_FACE_HUB_TOKEN` config in `lora-serve.yaml` with your own values. This is needed for Llama 2 models

To deploy the LoRA models, run:
```shell
serve run lora-serve.yaml
```

# Querying LoRA Models
In order to query the model, update the model id in `lora-query.py`. The `model` used in `lora-query.py` is expected to be in `{base_model_id}:{suffix}:{id}` format (e.g. `meta-llama/Llama-2-7b-chat-hf:lora-model:1234`). You can also run query directly on the base model by changing the `model` variable to the base model id. To query, run:

```shell
python lora-query.py

# Example output:
# {
#     "id": "meta-llama/Llama-2-7b-chat-hf:lora-model:1234-472e56b56039273c260e783a80950816",
#     "object": "text_completion",
#     "created": 1699563681,
#     "model": "meta-llama/Llama-2-7b-chat-hf:lora-model:1234",
#     "choices": [
#         {
#             "message": {
#                 "role": "assistant",
#                 "content": " Sure, I can do that! Based on the target sentence you provided, I will construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.\n\nThe function I have constructed is:\n\n['inform', 'available_on_steam'] [1] [developer] [Slightly Mad Studios] [/]  \n\nThe attributes are:\n\n[1] [release_year] [2012]\n[developer] [Slightly Mad Studios]"
#             },
#             "index": 0,
#             "finish_reason": "stop"
#         }
#     ],
#     "usage": {
#         "prompt_tokens": 285,
#         "completion_tokens": 110,
#         "total_tokens": 395
#     }
# }
```
