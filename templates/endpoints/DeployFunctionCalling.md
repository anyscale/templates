# Serving Function calling Models

Anyscale Endpoints supports [function calling mode](https://www.anyscale.com/blog/anyscale-endpoints-json-mode-and-function-calling-features). You can serve function calling models by using following steps.

# Step 1 - Configuring Function calling model

If a model (example Mistral-7B-Instruct-v0.1) supports function calling, then you need to make following changes in model configuration file.

For Example, you can see `models/mistral/mistralai--Mistral-7B-Instruct-v0.1_a10g_tp1.yaml` which has function calling mode enabled.

1. Set `enable_json_logits_processors: true` under `engine_kwargs`

```
  engine_kwargs:
    trust_remote_code: true
    max_num_batched_tokens: 16384
    max_num_seqs: 64
    gpu_memory_utilization: 0.95
    num_tokenizer_actors: 2
    enable_cuda_graph: true
    enable_json_logits_processors: true
```

2. Set `standalone_function_calling_model: true` in top level configuration. 

# Step 2 - Deploying & Querying Function calling model

`func_calling-serve.yaml` and `llm-query.py` are provided for you in this template. 

In order to deploy a model in function calling mode you need to edit `func_calling-serve.yaml`:
Under `function_calling_models` add path to the model you want to use. You can add multiple model

To deploy the the models, run:
```shell
serve run func_calling-serve.yaml
```

# Step 3 - Query the service endpoint

In order to query the endpoint, you can modify the `func_calling-query.py` script, replacing the query url with the Service URL found in the Service UI.

Note: please make sure to include the path "/v1" at the end of the Service url.
