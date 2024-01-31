# Endpoints - Deploy, configure, and serve LLMs 

The guide below walks you through the steps required for deployment of LLM endpoints. Based on Ray Serve and RayLLM, the foundation for [Anyscale-Hosted Endpoints](http://anyscale.com/endpoints), the Endpoints template provides an easy to configure solution for ML Platform teams, Infrastructure engineers, and Developers who want to deploy optimized LLMs in production.  We have provided a number of examples for popular open-source models (Llama2, Mistral, Mixtral, embedding models, and more) with different GPU accelerator and tensor-parallelism configurations in the `models` directory. 

# Step 1 - Deploy the model on Workspace

The llm-serve.yaml file in this example runs the Mistral-7B model on an A10G GPU. There are 2 important configurations you would need to modify:
1. The `models` config in `llm-serve.yaml` contains a list of YAML files for the models you want to deploy. You can run any of the models in the `models` directory or define your own model YAML file and run that instead. Follow the CustomModels [guide](CustomModels.md) for bringing your own models.
2. `HUGGING_FACE_HUB_TOKEN` - The Meta Llama-2 family of models need the HUGGING_FACE_HUB_TOKEN variable to be set to a Hugging Face Access Token for an account with permissions to download the model.

* Note that AWS offers A10G GPUs while GCP offers L4. Please switch to the correct config file for the GPU type. All config files follow the naming convention {model_name}_{accelerator_type}_{tensor_parallelism}

From the terminal use the Ray Serve CLI to deploy a model:

```shell
# Deploy the Mistral-7b model. 

serve run llm-serve.yaml
```

# Step 2 - Query the model

Once deployed you can use the OpenAI SDK to interact with the models, ensuring an easy integration for your applications. Run the following command in a separate terminal to query. 

```shell
python llm-query.py
```
```text
Output:
The top rated restaurants in San Francisco include:
 • Chez Panisse
 • Momofuku Noodle Bar
 • Nopa
 • Saison
 • Mission Chinese Food
 • Sushi Nakazawa
 • The French Laundry
```

Endpoints uses an OpenAI-compatible API, allowing us to use the OpenAI SDK to access Endpoint backends.

```python
from openai import OpenAI

client = OpenAI(
  base_url="http://localhost:8000/v1",
  api_key="NOT A REAL KEY",
)

# List all models.
models = client.models.list()
print(models)

# Note: not all arguments are currently supported and will be ignored by the backend.
chat_completion = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What are some of the highest rated restaurants in San Francisco?'."}],
    temperature=0.01
)

print(chat_completion)

```

# Step 3 - Deploying a production service

To deploy an application with one model on an Anyscale Service you can run:

```shell
anyscale service rollout -f llm-service.yaml --name {ENTER_NAME_FOR_SERVICE_HERE}
```

This is setup to run the Mistral-7B model, but can be easily modified to run any of the other models in this repo.

# Step 4 - Query the service endpoint

In order to query the endpoint, you can modify the `llm-query.py` script, replacing the query url with the Service URL found in the Service UI.

Note: please make sure to include the path "/v1" at the end of the Service url.

# More Guides

Endpoints makes it easy for LLM Developers to interact with OpenAI compatible APIs for their applications by providing an easy to manage backend for serving OSS LLMs.  It does this by:

- Providing an extensive suite of pre-configured open source LLMs and embedding models, with defaults that work out of the box. 
- Simplifying the addition of new LLMs.
- Simplifying the deployment of multiple LLMs
- Offering unique autoscaling support, including scale-to-zero.
- Fully supporting multi-GPU & multi-node model deployments.
- Offering high performance features like continuous batching, quantization and streaming.
- Providing a REST API that is similar to OpenAI's to make it easy to migrate and integrate with other tools.

Look at the following guides for more advanced use-cases -
* [Deploy models for embedding generation](EmbeddingModels.md)
* [Learn how to bring your own models](CustomModels.md)
* [Deploy multiple LoRA fine-tuned models](DeployLora.md)
* [Deploy Function calling models](DeployFunctionCalling.md)
* [Learn how to leverage different configurations that can optimize the latency and throughput of your models](OptimizeModels.md)
* [Learn how to fully configure your deployment including auto-scaling, optimization parameters and tensor-parallelism](AdvancedModelConfigs.md)

# Application Examples
See examples of building applications with your deployed endpoint on the [Anyscale Endpoints](https://docs.endpoints.anyscale.com/category/examples) page.

Be sure to update the api_base and token for your private deployment. This can be found under the "Serve deployments" tab on the "Query" button when deploying on your Workspace.
