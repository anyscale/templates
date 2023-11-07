# Endpoints - Deploy, configure, and serve LLMs 

Based on Ray Serve and the foundation for [Anyscale-Hosted Endpoints](http://anyscale.com/endpoints), the Endpoints template provides an easy to configure solution for ML Platform teams, Infrastructure engineers, and Developers who need more control over the application's resource usage, configuration, logic, or custom models.

If you are interested in a serverless one-click offering for deploying Endpoints in your account, reach out to the [Anyscale team to learn more](mailto:endpoints-help@anyscale.com?subject=Endpoints).

| Template Specification | Description |
| ---------------------- | ----------- |
| Time to Run | Around 5 minutes to deploy a model. |
| Minimum Compute Requirements | The default is a head node with 32 vCPU. Different models (ie LLama2-13B) have different compute requirements and accelerator requirements.  We recommend deploying LLama2-70b on A100 accelerators and specific hardware, while smaller models like LLama2-7b may use alternative accelerators like A10.  Read more to learn how to change the default instance selections and accelerators.|
| Cluster Environment | This template uses the latest Anyscale-provided 'Aviary docker image' (anyscale/aviary:0.3.1). If you want to change to a different cluster environment, make sure to follow the BYOD development flow and the image is based off the anyscale/aviary images. |

##  Prerequisites
The Meta Llama-2 family of models need the HUGGING_FACE_HUB_TOKEN environment variable to be set to a Hugging Face Access Token for an account with permissions to download the model.

To add your HUGGING_FACE_HUB_TOKEN go to the `serve.yaml` and `service.yaml` files and add your token in the Environment Variable place holder section.

##  Endpoints Background
Endpoints makes it easy for LLM Developers to interact with OpenAI compatible APIs for their applications by providing an easy to manage backend for serving OSS LLMs.  It does this by:

- Providing an extensive suite of pre-configured open source LLMs, with defaults that work out of the box.
- Supporting Transformer models hosted on Hugging Face Hub or present on local disk.
- Simplifying the deployment of multiple LLMs
- Simplifying the addition of new LLMs
- Offering unique autoscaling support, including scale-to-zero.
- Fully supporting multi-GPU & multi-node model deployments.
- Offering high performance features like continuous batching, quantization and streaming.
- Providing a REST API that is similar to OpenAI's to make it easy to migrate and cross test them.

RayLLM supports continuous batching by integrating with vLLM. Continuous batching allows you to get much better throughput and latency than static batching.

RayLLM leverages Ray Serve, which has native support for autoscaling and multi-node deployments. RayLLM can scale to zero and create new model replicas (each composed of multiple GPU workers) in response to demand.

## Table of Contents

- [Development- Deploying Endpoints Backend](#deploying-endpoints-for-development)
  * [Wokspaces](#worksapce-deployment)
    + [Ray Serve](#using-ray-serve)
  * [Query](#query)
    + [Model](#query-the-model)
- [Deploying as a Production Service](#deploying-on-anyscale-services)
- [Using the OpenAI SDK](#using-the-openai-sdk) 
- [Model Registry](#model-registry)
- [Serving LoRA Models](#servomg-lora-models)
- [Frequently Asked Questions](#frequently-asked-questions)

## Deploying Endpoints for Development

The guide below walks you through the steps required for deployment of Endpoints.  You can deploy any model in the `models` directory of this repo, 
or define your own model YAML file and run that instead.

Once deployed, LLM Developers can simply use an Open AI compatible api to interact with the deployed models.

### Workspace Deployment

In this guide, we will go over deploying a model locally using serve run as well as on an Anyscale Service. Once deployed you can use the OpenAI SDK to interact with the models ensuring an easy integration for your applications.  

#### Using Ray Serve
From the terminal use the Ray Serve CLI to deploy a model:

```shell
# Deploy the Llama-7b model. 

serve run serve.yaml
```

The serve YAML file runs the Llama-7B model. You can modify it to deploy any model in the `models` directory of this repo, provided you have the right GPU resources. You can also define your own model YAML file in the `models/` directory and run that instead. Follow the Model Registry [guide](models/README.md) for that.

### Query

#### Query the model

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

# Deploying on Anyscale Services

To deploy an application with one model on an Anyscale Service you can run:

```shell
anyscale service rollout -f service.yaml --name {ENTER_NAME_FOR_SERVICE_HERE}
```

This is setup to run the Llama-2-13B model, but can be easily modified to run any of the other models in this repo.
In order to query the endpoint, you can modify the `query.py` script, replacing the query url with the Service URL found in the Service UI.

Ansycale Services provide highly available fault tolerance for production LLM serving needs.  Learn more about [Anyscale Services](https://docs.anyscale.com/productionize/services/get-started)!

# Using the OpenAI SDK

Endpoints uses an OpenAI-compatible API, allowing us to use the OpenAI
SDK to access Endpoint backends. To do so, we need to set the `OPENAI_API_BASE` env var. From the terminal:

```shell
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY='not_a_real_key'
```

```python
import openai

# List all models.
models = openai.Model.list()
print(models)

# Note: not all arguments are currently supported and will be ignored by the backend.
chat_completion = openai.ChatCompletion.create(
    model="meta-llama/llama2-7b",
    messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Say 'test'."}],
    temperature=0.7
)
print(chat_completion)
```

# Application Examples
See examples of building applications with your deployed endpoint on the [Anyscale Endpoints](https://docs.endpoints.anyscale.com/category/examples) page.

Be sure to update the api_base and token for your private deployment.  This can be found under the "Serve deployments" tab on the "Query" button when deploying on your Workspace.

When deploying on your production service the Service landing page has a "Query" button in the upper right hand corner with the url and token information.

# Model Registry

Endpoints allows you to easily add new models by adding a single configuration file.
To learn more about how to customize or add new models, 
see the [Model Registry](models/README.md).

# Serving LoRA Models

`serve_lora.yaml` and `multiplex_lora_adapters/lora_config.yaml` are provided for you.
Make sure you replace the `HUGGING_FACE_HUB_TOKEN` config in `serve_lora.yaml` and `bucket_uri` config in `lora_mirror_config.lora_config.yaml` with your own values.

To deploy the LoRA model, run:
```shell
# Deploy the Llama-7b + "lora-viggo-finetuned" loRA models.

serve run serve_lora.yaml
```

This will take up to a minute or so to load depending on the model size.
To query the LoRA model, run:
```shell
python query_lora.py
```

A few tips for using LoRA:
1. The `base_model_id` config for the loRA base model in `multiplex_lora_adapters/lora_config.yaml` defines the based model of the loRA. The base model defined in `models/meta-llama--Llama-2-7b-chat-hf_a10.yaml` requires to have `engine_config.engine_kwargs.enable_lora` set to `true` to be able to use.
1. Each `multiplex_lora_adapters` in `serve_lora.yaml` can be loaded from any S3 buckets the workspace has access to. You can use an existing bucket where you have the loRA models or can use `$ANYSCALE_ARTIFACT_STORAGE` provided by Anyscale Workspace.
1. The `model_id` in `lora_mirror_config.lora_config.yaml` will be exactly the model you use to run query for the loRA model. It has to be unique across all models.
1. You can also run query directly on the base model by changing the `model` variable in `query_lora.py`.

# Frequently Asked Questions

## How do I add a new model?

The easiest way is to copy the configuration of the existing model's YAML file and modify it. See models/README.md for more details.

## How do I deploy multiple models at once?

You can append another application configuration to the YAML in `serve.yaml` file. Alternatively, you can use the CLI linked above.

## How do I deploy a model to multiple nodes?

All our default model configurations enforce a model to be deployed on one node for high performance. However, you can easily change this if you want to deploy a model across nodes for lower cost or GPU availability. In order to do that, go to the YAML file in the model registry and change `placement_strategy` to `PACK` instead of `STRICT_PACK`.

## How can I configure the resources / instances being used or the scaling behavior of my service?

You can edit the Compute Configuration direclty on your Workspace.  [Compute configurations](https://docs.anyscale.com/configure/compute-configs/overview) define the shape of the cluster and what resources Anyscale will use to deploy models and serve traffic.  If you would like to edit the default compute configuration choose "Edit" on your workspace and update the configuration.  When moving to production and deploying as an Ansycale Service the new configuration will be used.

Note that certain models require special accelerators.  Be aware that updating the resources make cause issues with your application.  

## My deployment isn't starting/working correctly, how can I debug?

There can be several reasons for the deployment not starting or not working correctly. Here are some things to check:
1. You might have specified an invalid model id.
2. Your model may require resources that are not available on the cluster. A common issue is that the model requires Ray custom resources (eg. `accelerator_type_a10`) in order to be scheduled on the right node type, while your cluster is missing those custom resources. You can either modify the model configuration to remove those custom resources or better yet, add them to the node configuration of your Ray cluster. You can debug this issue by looking at Ray Autoscaler logs ([monitor.log](https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html#system-component-logs)).
3. Your model is a gated Hugging Face model (eg. meta-llama). In that case, you need to set the `HUGGING_FACE_HUB_TOKEN` environment variable cluster-wide. You can do that either in the Ray cluster configuration or by setting it before running `serve run`.
4. Your model may be running out of memory. You can usually spot this issue by looking for keywords related to "CUDA", "memory" and "NCCL" in the replica logs or `serve run` output. In that case, consider reducing the `max_batch_prefill_tokens` and `max_batch_total_tokens` (if applicable). See models/README.md for more information on those parameters.

In general, [Ray Dashboard](https://docs.ray.io/en/latest/serve/monitoring.html#ray-dashboard) is a useful debugging tool, letting you monitor your application and access Ray logs.

# Getting Help and Filing Bugs / Feature Requests

We are eager to help you get started with Endpoints. You can get help on: 

- Via Slack -- fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLSfAcoiLCHOguOm8e7Jnn-JJdZaCxPGjgVCvFijHB5PLaQLeig/viewform) to sign up. 
- Via [Discuss](https://discuss.ray.io/c/llms-generative-ai/27). 

We have people in both US and European time zones who will help answer your questions. 
