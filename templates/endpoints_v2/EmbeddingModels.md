# Serving Embedding Models

We support serving embedding models available in HuggingFace as well as optimizing these with ONNX. Sample configs are available in the `models/embedding_models` folder.

# Setting up Model

See an example for serving embedding models in `serve_embedding.yaml`. Notably the serve arguments need to contain the `embedding_models` field. 

In order to deploy an embedding model run:
```shell
serve run serve_embedding.yaml
```

# Querying Embedding Models

You can use the OpenAI SDK to query the embedding models. Batch queries are also supported. In order to query the example above, run:

```shell
python query_embedding.py
```
