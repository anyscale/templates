# Serving Embedding Models

We support serving embedding models available in HuggingFace as well as optimizing these with ONNX. Sample configs are available in the `models/embedding_models` folder.

# Setting up Model

See an example for serving embedding models in `embedding-serve.yaml`. Notably the `args` field in the yaml file needs to contain the `embedding_models` field. This field contains a list of YAML files (in the `models` directory) for the embedding models you want to deploy. 

In order to deploy an embedding model run:
```shell
serve run embedding-serve.yaml
```

# Querying Embedding Models

You can use the OpenAI SDK to query the embedding models. Batch queries are also supported. In order to query the example above, run:

```shell
python embedding-query.py
```

# Optimizing Embedding Models

We support optimizing embedding models with ONNX. In order to enable this, set the flag under `engine_config` in your model yaml file. See `models/embedding_models\BAAI--bge-large-en-v1.5.yaml` for an example. 

```shell
engine_config:
  ...
  optimize: onnx
```

By default, the embedding models are setup to run on CPU. You can modify the configuration to run it on GPU instead.
