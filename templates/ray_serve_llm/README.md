# Deploy, configure, and serve LLMs 

**⏱️ Time to complete**: 10 min

This template comes with a library for serving OSS LLMs on Anyscale called [Ray Serve LLM](https://docs.ray.io/en/releases-2.43.0/serve/llm/overview.html).

Ray Serve LLM provides a number of features that simplifies LLM development, including:
- An extensive suite of pre-configured open source LLMs.
- An OpenAI-compatible REST API.

As well as operational features to efficiently scale LLM apps:
- Optimizations such as continuous batching, quantization and streaming.
- Production-grade autoscaling support, including scale-to-zero.
- Native multi-GPU & multi-node model deployments.

This template explains how to set up, run, and query LLMs with Ray Serve LLM.

To learn more about Ray Serve LLM, check out [the docs](https://docs.ray.io/en/releases-2.43.0/serve/llm/overview.html). If you're interested in fine-tuning, check out the [fine-tuning template](https://console.anyscale.com/v2/template-preview/finetuning_llms_v2).

**Note**: This guide is hosted within an Anyscale workspace, which provides easy access to compute resources. Check out the [Introduction to Workspaces](https://docs.anyscale.com/examples/intro-workspaces/) template for more details.

## Step 1 - Generate a LLM config

Ray Serve LLM takes in a config that specifies the model you wish to serve, as well as its settings.

Use the Ray Serve LLM CLI in the workspace terminal to generate the config:

```bash
python -m ray.serve.llm.gen_config
```

**Note:** This command requires interactive inputs and should be executed directly in the terminal, not within a Jupyter notebook cell.

This command lets you pick from a common set of OSS LLMs and helps you configure them. You can tune settings like GPU type, tensor parallelism, and autoscaling parameters.

Please note that if you're configuring a model whose architecture is different from the provided list of models, we recommend that you closely review the generated model config file to provide the correct values.

This command generates 2 files - an LLM config file (saved in `model_config/`) and a Ray Serve config file (`serve_TIMESTAMP.yaml`) that you can reference and re-run in the future.

## Step 2 - Run the model locally in the workspace

If you didn't start the serve application in the previous step, you can start it using the following command (replace the file name with the generated `serve_` file name):


```python
!serve run serve_TIMESTAMP.yaml
```

## Step 3 - Query the model

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

Ray Serve LLM uses an OpenAI-compatible API, allowing us to use the OpenAI SDK to query the LLMs.


```python
from openai import OpenAI

# TODO: Replace this model ID with your own.
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def query(base_url: str, api_key: str | None = None):
    if not base_url.endswith("/"):
        base_url += "/"
    
    if "/routes" in base_url:
        raise ValueError("base_url must end with '.com'")

    client = OpenAI(
      base_url=base_url + "v1",
      api_key=api_key or "NOT A REAL KEY",
    )

    # List all models.
    models = client.models.list()
    print(models)

    # Note: not all arguments are currently supported and will be ignored by the backend.
    chat_completions = client.chat.completions.create(
        model=MODEL_ID,
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

## Step 4 - Deploying a production service

To deploy an application with one model as an Anyscale Service, update the file name to the generated one and run the following command:


```python
# Deploy the serve app to production with a given service name.
# Reference the serve file created in step 1
!anyscale service deploy -f serve_TIMESTAMP.yaml
```

After the command runs, click the deploy notification (or navigate to ``Home > Services``) to access the Service UI:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/ray_serve_llm/assets/service-notify.png" width=500px />

Navigate to the Service UI and wait for the service to reach "Active". It will begin in "Starting" state:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/ray_serve_llm/assets/service-starting.png" width=600px />


## Step 5 - Query the service endpoint

The above command should print something like `(anyscale +2.9s) curl -H 'Authorization: Bearer XXXXXXXXX_XXXXXXXXXXXXXXXXXX' https://YYYYYYYYYYYY.anyscaleuserdata.com`, which contains information you need to query the service.

You can also find this information by clicking the "Query" button in the Service UI.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/ray_serve_llm/assets/service-query.png" width=600px />


```python
# Query the remote serve application we just deployed.

service_url = "https://YYYYYYYYYYYYY.anyscaleuserdata.com"  # FILL ME IN
service_bearer_token = "XXXXXXXXXX_XXXXXXXXXXXXXXXXXXXXX"  # FILL ME IN

query(service_url, service_bearer_token)
```

## End-to-end examples

* [Build Tool Calling Feature for Any LLM via JSON Mode](./end-to-end-examples/function_calling/README.ipynb): This example demonstrates how to build a tool calling feature for any LLM via JSON mode.


## What's next?

* Try the [fine-tuning template](https://console.anyscale.com/v2/template-preview/finetuning_llms_v2) to tune some LLMs for your use case.
* See the [offline batch inference template](https://console.anyscale.com/v2/template-preview/batch-llm) to learn how to run LLMs for batch jobs.

