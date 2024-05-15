# Deploy, configure, and serve LLMs 

**⏱️ Time to complete**: 10 min (20 on GCP)

This guide walks you through how to deploy optimized LLM endpoints in Anyscale. It includes a number of pre-tuned configs for Llama2, Mistral, Mixtral, embedding models, and more in the `models` directory.

You can also find more advanced tutorials in the `examples/` folder, including those for:
- Embedding generation
- Deploying custom models
- Deploying LoRA and function-calling models
- How to configure autoscaling and other optimization parameters

**Note**: This guide is hosted within an Anyscale workspace, which provides easy access to compute resources. Check out the `Introduction to Workspaces` template for more details.

## Step 1 - Run the model locally in the Workspace

We provide a starter command to run Llama and Mistral-family models via Ray Serve. You can specify the arguments, such as Lora, GPU type and tensor parallelism via the command. You can also follow the [guide](examples/CustomModels.ipynb) to bring your own models.

The command will generate 2 files - a model config file and a serve config file that you can reference and re-run in the future.

Please note that if you would like to serve a model whose architecture is different from the provided list of models, we highly recommend you manually going over the generated model config file to provide the correct values.

```python
!python generate_config.py
```

If you didn't start the serve application in the previous step, you can start it using the following command (replace the file name with the generated `serve_` file name):

```python
!serve run serve_TIMESTAMP.yaml
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
    
    if "/routes" in base_url:
        raise ValueError("base_url must end with '.com'")

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

To deploy an application with one model as an Anyscale Service, update the file name to the generated one and run the following command :


```python
# Deploy the serve app to production with a given service name.
# Reference the serve file created in step 1
!anyscale service deploy -f serve_TIMESTAMP.yaml
```

After the command runs, click the deploy notification (or navigate to ``Home > Services``) to access the Service UI. Navigate to the Service UI and wait for the service to reach "Active". It will begin in "Starting" state.

## Step 4 - Query the service endpoint

The above command should print something like `(anyscale +2.9s) curl -H 'Authorization: Bearer XXXXXXXXX_XXXXXX-XXXXXXXXXXXX' https://YYYYYYYYYYYY.anyscaleuserdata.com`, which contains information you need to query the service.

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