# Adding a new model

RayLLM supports fine-tuned versions of models in the `models` directory as well as model architectures supported by [vLLM](https://docs.vllm.ai/en/latest/models/supported_models.html). You can either bring a model from HuggingFace or artifact storage like S3, GCS. 

## Configuring a new model

To add an entirely new model to the zoo, you will need to create a new YAML file.
This file should follow the naming convention 
`<organisation-name>--<model-name>-<model-parameters>-<extra-info>.yaml`. We recommend using one of the existing models as a template (ideally, one that is the same architecture and number of parameters as the model you are adding). The examples in the `models` directory should help you get started. You can look at the [Advanced Model Configs](./AdvancedModelConfigs.md) for more details on these configurations.

```yaml
# true by default - you can set it to false to ignore this model
# during loading
enabled: true
deployment_config:
  # This corresponds to Ray Serve settings, as generated with
  # `serve build`.
  autoscaling_config:
    min_replicas: 1
    initial_replicas: 1
    max_replicas: 8
    target_num_ongoing_requests_per_replica: 1.0
    metrics_interval_s: 10.0
    look_back_period_s: 30.0
    smoothing_factor: 1.0
    downscale_delay_s: 300.0
    upscale_delay_s: 90.0
  ray_actor_options:
    # Resources assigned to each model deployment. The deployment will be
    # initialized first, and then start prediction workers which actually hold the model.
    resources:
      "accelerator_type:A100-40G": 0.01
engine_config:
  # Model id - this is a RayLLM id
  model_id: mosaicml/mpt-7b-instruct
  # Id of the model on Hugging Face Hub. Defaults to model_id if not specified.
  hf_model_id: mosaicml/mpt-7b-instruct
  # vLLM keyword arguments passed when constructing the model.
  engine_kwargs:
    trust_remote_code: true
  # Optional Ray Runtime Environment configuration. See Ray documentation for more details.
  # Add dependent libraries, environment variables, etc.
  runtime_env:
    env_vars:
      YOUR_ENV_VAR: "your_value"
  # Optional configuration for loading the model from S3 instead of Hugging Face Hub. You can use this to speed up downloads or load models not on Hugging Face Hub.
  s3_mirror_config:
    bucket_uri: s3://large-dl-models-mirror/models--mosaicml--mpt-7b-instruct/main-safetensors/
  generation:
    # Prompt format to wrap queries in. {instruction} refers to user-supplied input.
    prompt_format:
      system: "{instruction}\n"  # System message. Will default to default_system_message
      assistant: "### Response:\n{instruction}\n"  # Past assistant message. Used in chat completions API.
      trailing_assistant: "### Response:\n"  # New assistant message. After this point, model will generate tokens.
      user: "### Instruction:\n{instruction}\n"  # User message.
      default_system_message: "Below is an instruction that describes a task. Write a response that appropriately completes the request."  # Default system message.
      system_in_user: false  # Whether the system prompt is inside the user prompt. If true, the user field should include '{system}'
      add_system_tags_even_if_message_is_empty: false  # Whether to include the system tags even if the user message is empty.
      strip_whitespace: false  # Whether to automaticall strip whitespace from left and right of user supplied messages for chat completions
    # Stopping sequences. The generation will stop when it encounters any of the sequences, or the tokenizer EOS token.
    # Those can be strings, integers (token ids) or lists of integers.
    # Stopping sequences supplied by the user in a request will be appended to this.
    stopping_sequences: ["### Response:", "### End"]

# Resources assigned to each model replica.
scaling_config:
  # If using multiple GPUs set num_gpus_per_worker to be 1 and then set num_workers to be the number of GPUs you want to use.
  num_workers: 1
  num_gpus_per_worker: 1
  num_cpus_per_worker: 4
  resources_per_worker:
    # You can use custom resources to specify the instance type / accelerator type
    # to use for the model.
    "accelerator_type:A100-40G": 0.01

```

## Adding a private model 

For loading a model from S3 or GCS, set `engine_config.s3_mirror_config.bucket_uri` or `engine_config.gcs_mirror_config.bucket_uri` to point to a folder containing your model and tokenizer files (`config.json`, `tokenizer_config.json`, `.bin`/`.safetensors` files, etc.) and set `engine_config.model_id` to any ID you desire in the `organization/model` format, eg. `myorganization/llama2-finetuned`. The model will be downloaded to a folder in the `<TRANSFORMERS_CACHE>/models--<organization-name>--<model-name>/snapshots/<HASH>` directory on each node in the cluster. `<HASH>` will be determined by the contents of `hash` file in the S3 folder, or default to `0000000000000000000000000000000000000000`. See the [HuggingFace transformers documentation](https://huggingface.co/docs/transformers/main/en/installation#cache-setup).

For loading a model from an accessible S3 bucket:

```yaml
engine_config:
  model_id: YOUR_MODEL_NAME
  s3_mirror_config:
    bucket_uri: s3://YOUR_BUCKET_NAME/YOUR_MODEL_FOLDER
    extra_files: []
```

For loading a model from an accessible Google Cloud Storage bucket:

```yaml
engine_config:
  model_id: YOUR_MODEL_NAME
  s3_mirror_config:
    bucket_uri: gs://YOUR_BUCKET_NAME/YOUR_MODEL_FOLDER
    extra_files: []
```


### Example prompt config (Llama-based model)

```
prompt_format:
  system: "<<SYS>>\n{instruction}\n<</SYS>>\n\n"
  assistant: " {instruction} </s><s>"
  trailing_assistant: ""
  user: "[INST] {system}{instruction} [/INST]"
  system_in_user: true
  default_system_message: ""
stopping_sequences: []
```

### Example prompt config (Mistral-based-model)

```
prompt_format:
  system: "<<SYS>>\n{instruction}\n<</SYS>>\n\n"
  assistant: " {instruction} </s><s> "
  trailing_assistant: " "
  user: "[INST] {system}{instruction} [/INST]"
  system_in_user: true
  default_system_message: "Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity."
stopping_sequences: []
```

### Example prompt config (Falcon-based-model)

```
prompt_format:
  system: "<|prefix_begin|>{instruction}<|prefix_end|>"
  assistant: "<|assistant|>{instruction}<|endoftext|>"
  trailing_assistant: "<|assistant|>"
  user: "<|prompter|>{instruction}<|endoftext|>"
  default_system_message: "Below are a series of dialogues between various people and an AI assistant. The AI tries to be helpful, polite, honest, sophisticated, emotionally aware, and humble-but-knowledgeable. The assistant is happy to help with almost anything, and will do its best to understand exactly what is needed. It also tries to avoid giving false or misleading information, and it caveats when it isn't entirely sure about the right answer. That said, the assistant is practical and really does its best, and doesn't let caution get too much in the way of being useful."
stopping_sequences: ["<|prompter|>", "<|assistant|>", "<|endoftext|>"]
```
