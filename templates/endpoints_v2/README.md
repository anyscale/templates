# Deploy, configure, and serve LLMs 

The guide below walks you through the steps required for deployment of LLM endpoints. Based on Ray Serve and RayLLM, the foundation for [Anyscale-Hosted Endpoints](http://anyscale.com/endpoints), the Endpoints template provides an easy to configure solution for ML Platform teams, Infrastructure engineers, and Developers who want to deploy optimized LLMs in production.  We have provided a number of examples for popular open-source models (Llama2, Mistral, Mixtral, embedding models, and more) with different GPU accelerator and tensor-parallelism configurations in the `models` directory.

This template also includes more advanced tutorials in the `examples/` folder, including those for:
- Embedding generation
- Deploying custom models
- Deploying LoRA and function-calling models
- How to configure autoscaling and other optimization parameters

## Step 1 - Run the model locally in the Workspace

The llm-serve.yaml file in this example runs the Mistral-7B model. There are 2 important configurations you would need to modify:
1. The `models` config in `llm-serve-*.yaml` contains a list of YAML files for the models you want to deploy. You can run any of the models in the `models` directory or define your own model YAML file and run that instead. All config files follow the naming convention `{model_name}_{accelerator_type}_{tensor_parallelism}`. Follow the CustomModels [guide](CustomModels.md) for bringing your own models.
2. `HUGGING_FACE_HUB_TOKEN` - The Meta Llama-2 family of models need the HUGGING_FACE_HUB_TOKEN variable to be set to a Hugging Face Access Token for an account with permissions to download the model.



From the VSCode terminal (press [**Ctrl + `**] in VSCode), use the Ray Serve CLI to deploy the model for testing. It will take a few minutes to initialize and download the model.

```bash
# Note: if using GCP cloud, use llm-serve-gcp.yaml instead to select L4 GPU instances.
$ serve run llm-serve-aws.yaml
```

## Step 2 - Query the model

Once deployed you can use the OpenAI SDK to interact with the models, ensuring an easy integration for your applications.

Run the following command to query. You should get the following output:
```
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

def query(base_url: str, api_key: str):
    if not base_url.endswith("/"):
        base_url += "/"

    client = OpenAI(
      base_url=base_url + "v1",
      api_key=api_key,
    )

    # List all models.
    models = client.models.list()
    print(models)

    # Note: not all arguments are currently supported and will be ignored by the backend.
    chat_completions = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are some of the highest rated restaurants in San Francisco?'."},
        ],
        temperature=0.01,
        stream=True
    )

    for chat in chat_completions:
        if chat.choices[0].delta.content is not None:
            print(chat.choices[0].delta.content, end="")
```


```python
# Query the local serve application we just deployed.

query("http://localhost:8000", "NOT A REAL KEY")
```

## Step 3 - Deploying a production service

To deploy an application with one model as an Anyscale Service you can run:


```python
# Deploy the serve app to production with a given service name.
# Change to llm-serve-gcp.yaml if needed.
!serve deploy --name=my_service_name llm-serve-aws.yaml
```

This is setup to run the Mistral-7B model, but can be easily modified to run any of the other models in this repo.

## Step 4 - Query the service endpoint

The above cell should print something like `(anyscale +2.9s) curl -H 'Authorization: Bearer XXXXXXXXX_XXXXXX-XXXXXXXXXXXX' https://YYYYYYYYYYYY.anyscaleuserdata.com/-/routes`, which contains information you need to fill out in the cell below to query the service.

You can also find this information by clicking the "Query" button in the Service UI.


```python
# Query the remote serve application we just deployed.

service_url = "https://YYYYYYYYYYYYY.anyscaleuserdata.com"  # FILL ME IN
service_bearer_token = "XXXXXXXXXX_XXXXXXX-XXXXXXXXXXXXXX"  # FILL ME IN

query(service_url, service_bearer_token)
```

## More Guides

Endpoints makes it easy for LLM Developers to interact with OpenAI compatible APIs for their applications by providing an easy to manage backend for serving OSS LLMs.

It provides a number of features making LLM development easy, including:
- An extensive suite of pre-configured open source LLMs and embedding models.
- An OpenAI compatible REST API.

As well as operational features for efficient scaling of LLM apps:
- Optimizations such as continuous batching, quantization and streaming.
- Production-grade autoscaling support, including scale-to-zero.
- Native multi-GPU & multi-node model deployments.

Look at the following guides for more advanced use-cases:
* [Deploy models for embedding generation](examples/embedding/EmbeddingModels.ipynb)
* [Learn how to bring your own models](examples/CustomModels.ipynb)
* [Deploy multiple LoRA fine-tuned models](examples/lora/DeployLora.ipynb)
* [Deploy Function calling models](examples/function_calling/DeployFunctionCalling.ipynb)
* [Learn how to leverage different configurations that can optimize the latency and throughput of your models](examples/OptimizeModels.ipynb)
* [Learn how to fully configure your deployment including auto-scaling, optimization parameters and tensor-parallelism](examples/AdvancedModelConfigs.ipynb)

## Application Examples

See examples of building applications with your deployed endpoint on the [Anyscale Endpoints](https://docs.endpoints.anyscale.com/category/examples) page.

Be sure to update the `api_base` and `token` for your private deployment. This information can be found under the "Query" button in the Anyscale Service UI.



