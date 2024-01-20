# Endpoints - Deploy, configure, and serve LLMs 

The guide below walks you through the steps required for deployment of LLM endpoints. Based on Ray Serve and RayLLM, the foundation for [Anyscale-Hosted Endpoints](http://anyscale.com/endpoints), the Endpoints template provides an easy to configure solution for ML Platform teams, Infrastructure engineers, and Developers who want to deploy optimized LLMs in production. 

Endpoints makes it easy for LLM Developers to interact with OpenAI compatible APIs for their applications by providing an easy to manage backend for serving OSS LLMs.  It does this by:

- Providing an extensive suite of pre-configured open source LLMs, with defaults that work out of the box. You can deploy any model in the `models` directory of this repo, or define your own model YAML file and run that instead. Once deployed, LLM Developers can simply use an Open AI compatible api to interact with the deployed models.
- Supporting Transformer models hosted on Hugging Face Hub or present on local disk.
- Simplifying the deployment of multiple LLMs
- Simplifying the addition of new LLMs
- Offering unique autoscaling support, including scale-to-zero.
- Fully supporting multi-GPU & multi-node model deployments.
- Offering high performance features like continuous batching, quantization and streaming.
- Providing a REST API that is similar to OpenAI's to make it easy to migrate and integrate with other tools.

# Workspace Development

We will go over deploying a model locally using `serve run` as well as on an Anyscale Service. Once deployed you can use the OpenAI SDK to interact with the models, ensuring an easy integration for your applications.  

## Deploy the model on Workspace

The serve.yaml file in this example runs the Llama-7B model. There are 2 important configurations you would need to modify:
1. The `models` config in `serve.yaml` contains a list of YAML files for the models you want to deploy. We have provided a number of examples for popular open-source models with different GPU accelerator and tensor-parallelism configurations in the `models` directory. You can also define your own model YAML file in the `models/` directory and run that instead. Follow the CustomModels [guide](CustomModels.md) for that.
2. `HUGGING_FACE_HUB_TOKEN` - The Meta Llama-2 family of models need the HUGGING_FACE_HUB_TOKEN environment variable to be set to a Hugging Face Access Token for an account with permissions to download the model.

From the terminal use the Ray Serve CLI to deploy a model:

```shell
# Deploy the Llama-7b model. 

serve run serve.yaml
```

## Query the model

Run the following command in a separate terminal. 

```shell
python query.py
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
 • Delfina
 • Spices
 • Quince
 • Bistro L'Etoile
 • The Slanted Door
 • The Counter
 • The Chronicle
 • The Mint
 • The French Press
 • The Palace Cafe
 • The Inn at the Opera House
 • The Green Table
 • The Palace Cafe
```

## Using the OpenAI SDK

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
  model="meta-llama/Llama-2-7b-chat-hf",
  messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Say 'test'."}],
  temperature=0.7,
)
print(chat_completion)

```

# Deploying a production service

To deploy an application with one model on an Anyscale Service you can run:

```shell
anyscale service rollout -f service.yaml --name {ENTER_NAME_FOR_SERVICE_HERE}
```

This is setup to run the Llama-2-13B model, but can be easily modified to run any of the other models in this repo.
In order to query the endpoint, you can modify the `query.py` script, replacing the query url with the Service URL found in the Service UI.

Note: please make sure to include the path "/v1" at the end of the Service url.

Ansycale Services provide highly available fault tolerance for production LLM serving needs.  Learn more about [Anyscale Services](https://docs.anyscale.com/productionize/services/get-started)!

# Advanced Guides

* [Deploy models for embedding generation](EmbeddingModels.md)
* [Deploy multiple LoRA fine-tuned models](DeployLora.md)
* [Learn how to bring your own models](CustomModels.md)
* [Learn how to leverage different configurations that can optimize the latency and throughput of your models](CustomModels.md)

# Application Examples
See examples of building applications with your deployed endpoint on the [Anyscale Endpoints](https://docs.endpoints.anyscale.com/category/examples) page.

Be sure to update the api_base and token for your private deployment.  This can be found under the "Serve deployments" tab on the "Query" button when deploying on your Workspace.

When deploying on your production service the Service landing page has a "Query" button in the upper right hand corner with the url and token information.

# Getting Help and Filing Bugs / Feature Requests

We are eager to help you get started with Endpoints. You can get help on: 

- Via Slack -- fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSfAcoiLCHOguOm8e7Jnn-JJdZaCxPGjgVCvFijHB5PLaQLeig/viewform) to sign up. 
- Via [Discuss](https://discuss.ray.io/c/llms-generative-ai/27). 

We have people in both US and European time zones who will help answer your questions. 

