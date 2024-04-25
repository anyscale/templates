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

We provide a starter command to run Llama-2 and Mistral-family models via Ray Serve. Specify the model ID, GPU type, and tensor parallelism with the command arguments. You can also follow the [guide](examples/CustomModels.ipynb) to bring your own models.

Currently tensor parallelism defaults to 1 if not specified.

**Note**: For the Meta Llama-2 and Mistral families of models you need to set the `hf_token` variable to a Hugging Face Access Token for an account with permissions to download the model. Get your token from [Hugging Face's website at **Settings > Access Tokens**](https://huggingface.co/settings/tokens) and accept the terms on the model page to access the repository. 

Here is the list of currently supported model ID in the starter command:
- mistralai/Mistral-7B-Instruct-v0.1
- mistralai/Mixtral-8x7b
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-13b-chat-hf
- meta-llama/Llama-2-70b-chat-hf


```python
# Example command to serve Mistal-7B via A10 GPUs on AWS
!serve run rayllm.start:endpoint model_id=mistralai/Mistral-7B-Instruct-v0.1 gpu_type=A10 hf_token=YOUR_TOKEN --non-blocking

# Example command to serve Mistal-7B via L4 GPUs on GCP
# !serve run rayllm.start:endpoint model_id=mistralai/Mistral-7B-Instruct-v0.1 gpu_type=L4 hf_token=YOUR_TOKEN

# More example commands:
# !serve run rayllm.start:endpoint model_id=meta-llama/Llama-2-13b-chat-hf gpu_type=A100_40G tensor_parallelism=2 hf_token=YOUR_TOKEN
# !serve run rayllm.start:endpoint model_id=meta-llama/Llama-2-70b-chat-hf gpu_type=A100_80G tensor_parallelism=8 hf_token=YOUR_TOKEN
# !serve run rayllm.start:endpoint model_id=mistralai/Mixtral-8x7B-Instruct-v0.1 gpu_type=A100_80G tensor_parallelism=8 hf_token=YOUR_TOKEN
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

To deploy an application with one model as an Anyscale Service, run the next cell. This is setup to run the Mistral-7B model, but can be easily modified to run any of the other models in this repo:


```python
# Deploy the serve app to production with a given service name.
# Use the same "serve run" command. This is an command to serve Mistal-7B via A10 GPUs on AWS
!anyscale service deploy rayllm.start:endpoint model_id=mistralai/Mistral-7B-Instruct-v0.1 gpu_type=A10 hf_token=YOUR_TOKEN

# Example command to serve Mistal-7B via L4 GPUs on GCP
# !serve run rayllm.start:endpoint model_id=mistralai/Mistral-7B-Instruct-v0.1 gpu_type=L4 hf_token=YOUR_TOKEN
```

After the command runs, click the deploy notification (or navigate to ``Home > Services``) to access the Service UI:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/endpoints_v2/assets/service-notify.png" width=500px/>

Navigate to the Service UI and wait for the service to reach "Active". It will begin in "Starting" state:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/endpoints_v2/assets/service-starting.png" width=600px/>


## Step 4 - Query the service endpoint

The above cell should print something like `(anyscale +2.9s) curl -H 'Authorization: Bearer XXXXXXXXX_XXXXXX-XXXXXXXXXXXX' https://YYYYYYYYYYYY.anyscaleuserdata.com`, which contains information you need to fill out in the cell below to query the service.

You can also find this information by clicking the "Query" button in the Service UI.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/endpoints_v2/assets/service-query.png" width=600px/>


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



