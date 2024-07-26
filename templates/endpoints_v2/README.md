# Deploy, configure, and serve LLMs 

**⏱️ Time to complete**: 10 min (20 on GCP)

This guide walks you through how to deploy optimized LLMs in Anyscale through RayLLM. It includes a number of pre-tuned configs for Llama2, Mistral, Mixtral, embedding models, and more in the `models` directory.

You can also find more advanced tutorials in the `cookbooks/` folder, including those for:
- Embedding generation
- Deploying custom models
- Deploying LoRA and function-calling models
- Deploying vision language models
- How to configure autoscaling and other optimization parameters

**Note**: This guide is hosted within an Anyscale workspace, which provides easy access to compute resources. Check out the `Introduction to Workspaces` template for more details.

## Step 1 - Run the model locally in the Workspace

We provide a starter command to run Llama and Mistral-family models via Ray Serve. You can specify the arguments, such as Lora, GPU type and tensor parallelism via the command. You can also follow the [guide](cookbooks/CustomModels.ipynb) to bring your own models.

Please note that if you would like to serve a model whose architecture is different from the provided list of models, we highly recommend you manually going over the generated model config file to provide the correct values.

To generate the configuration file, run the following command directly in your terminal:
```
python generate_config.py
```
**Note:** This command requires interactive inputs and should be executed directly in the terminal, not within a Jupyter notebook cell.

The command will generate 2 files - a model config file (saved in `model_config/`) and a serve config file (`serve_TIMESTAMP.yaml`) that you can reference and re-run in the future.

If you didn't start the serve application in the previous step, you can start it using the following command (replace the file name with the generated `serve_` file name):


```python
!serve run serve_TIMESTAMP.yaml
```

    2024-07-22 18:31:24,455	INFO scripts.py:499 -- Running import path: 'serve_TIMESTAMP.yaml'.
    Traceback (most recent call last):
      File "/home/ray/anaconda3/bin/serve", line 8, in <module>
        sys.exit(cli())
      File "/home/ray/anaconda3/lib/python3.9/site-packages/click/core.py", line 1157, in __call__
        return self.main(*args, **kwargs)
      File "/home/ray/anaconda3/lib/python3.9/site-packages/click/core.py", line 1078, in main
        rv = self.invoke(ctx)
      File "/home/ray/anaconda3/lib/python3.9/site-packages/click/core.py", line 1688, in invoke
        return _process_result(sub_ctx.command.invoke(sub_ctx))
      File "/home/ray/anaconda3/lib/python3.9/site-packages/click/core.py", line 1434, in invoke
        return ctx.invoke(self.callback, **ctx.params)
      File "/home/ray/anaconda3/lib/python3.9/site-packages/click/core.py", line 783, in invoke
        return __callback(*args, **kwargs)
      File "/home/ray/anaconda3/lib/python3.9/site-packages/ray/serve/scripts.py", line 501, in run
        import_attr(import_path), args_dict
      File "/home/ray/anaconda3/lib/python3.9/site-packages/ray/_private/utils.py", line 1191, in import_attr
        module = importlib.import_module(module_name)
      File "/home/ray/anaconda3/lib/python3.9/importlib/__init__.py", line 127, in import_module
        return _bootstrap._gcd_import(name[level:], package, level)
      File "<frozen importlib._bootstrap>", line 1030, in _gcd_import
      File "<frozen importlib._bootstrap>", line 1007, in _find_and_load
      File "<frozen importlib._bootstrap>", line 984, in _find_and_load_unlocked
    ModuleNotFoundError: No module named 'serve_TIMESTAMP'
    [0m

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

RayLLM uses an OpenAI-compatible API, allowing us to use the OpenAI SDK to query the LLMs.


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

    Exception ignored in: <bound method IPythonKernel._clean_thread_parent_frames of <ipykernel.ipkernel.IPythonKernel object at 0x706462744f40>>
    Traceback (most recent call last):
      File "/home/ray/anaconda3/lib/python3.9/site-packages/ipykernel/ipkernel.py", line 775, in _clean_thread_parent_frames
        def _clean_thread_parent_frames(
    KeyboardInterrupt: 



```python
# Query the local serve application we just deployed.

query("http://localhost:8000", "NOT A REAL KEY")
```

## Step 3 - Deploying a production service

To deploy an application with one model as an Anyscale Service, update the file name to the generated one and run the following command:


```python
# Deploy the serve app to production with a given service name.
# Reference the serve file created in step 1
!anyscale service deploy -f serve_TIMESTAMP.yaml
```

After the command runs, click the deploy notification (or navigate to ``Home > Services``) to access the Service UI:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/endpoints_v2/assets/service-notify.png" width=500px/>

Navigate to the Service UI and wait for the service to reach "Active". It will begin in "Starting" state:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/endpoints_v2/assets/service-starting.png" width=600px/>


## Step 4 - Query the service endpoint

The above command should print something like `(anyscale +2.9s) curl -H 'Authorization: Bearer XXXXXXXXX_XXXXXX-XXXXXXXXXXXX' https://YYYYYYYYYYYY.anyscaleuserdata.com`, which contains information you need to query the service.

You can also find this information by clicking the "Query" button in the Service UI.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/endpoints_v2/assets/service-query.png" width=600px/>


```python
# Query the remote serve application we just deployed.

service_url = "https://YYYYYYYYYYYYY.anyscaleuserdata.com"  # FILL ME IN
service_bearer_token = "XXXXXXXXXX_XXXXXXX-XXXXXXXXXXXXXX"  # FILL ME IN

query(service_url, service_bearer_token)
```

## Final workds on RayLLM

RayLLM makes it easy for LLM Developers to interact with OpenAI compatible APIs for their applications by providing an easy to manage backend for serving OSS LLMs.

It provides a number of features making LLM development easy, including:
- An extensive suite of pre-configured open source LLMs and embedding models.
- An OpenAI compatible REST API.

As well as operational features for efficient scaling of LLM apps:
- Optimizations such as continuous batching, quantization and streaming.
- Production-grade autoscaling support, including scale-to-zero.
- Native multi-GPU & multi-node model deployments.

## Cookbooks

After you are done with the above, you can find recipies that extend the functionality of this template under the cookbooks folder:

* [Deploy models for embedding generation](cookbooks/embedding/EmbeddingModels.ipynb)
* [Deploy multiple LoRA fine-tuned models](cookbooks/lora/DeployLora.ipynb)
* [Deploy Function calling models](cookbooks/function_calling/DeployFunctionCalling.ipynb)
* [Deploy Vision Language Model](cookbooks/vision_language_model/README.ipynb): Build a Gradio appliation on top of LLaVA-NeXT Mistral 7B.
* [Learn how to bring your own models](cookbooks/CustomModels.ipynb)
* [Learn how to leverage different configurations that can optimize the latency and throughput of your models](cookbooks/OptimizeModels.ipynb)
* [Learn how to fully configure your deployment including auto-scaling, optimization parameters and tensor-parallelism](cookbooks/AdvancedModelConfigs.ipynb)


## Application examples

See examples of building applications with your deployed endpoint on the [Anyscale Endpoints](https://docs.anyscale.com/examples/work-with-openai) page.

Be sure to update the `api_base` and `token` for your private deployment. This information can be found under the "Query" button in the Anyscale Service UI.



