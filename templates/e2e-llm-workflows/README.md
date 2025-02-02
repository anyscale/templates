# End-to-end LLM Workflows Guide

In this guide, we'll learn how to execute the end-to-end LLM workflows to develop & productionize LLMs at scale.

- **Data preprocessing**: prepare our dataset for fine-tuning with batch data processing.
- **Fine-tuning**: tune our LLM (LoRA / full param) with key optimizations with distributed training.
- **Evaluation**: apply batch inference with our tuned LLMs to generate outputs and perform evaluation.
- **Serving**: serve our LLMs as a production application that can autoscale, swap between LoRA adapters, etc.

Throughout these workloads we'll be using [Ray](https://github.com/ray-project/ray), a framework for distributing ML, used by OpenAI, Netflix, Uber, etc. And [Anyscale](https://anyscale.com/?utm_source=goku), a platform to scale your ML workloads from development to production.


> **&nbsp;💵&nbsp;Cost**: $0 (using free [Anyscale](https://anyscale.com/?utm_source=goku) credits)<br/>
> **&nbsp;🕑&nbsp;Total time**: 90 mins (including fine-tuning) <br/>
> <b style="background-color: yellow;">&nbsp;🔄 REPLACE&nbsp;</b> indicates to replace with your unique values <br/>
> <b style="background-color: orange;">&nbsp;💡 INSIGHT&nbsp;</b> indicates infrastructure insight <br/>
> Join [Slack community](https://join.slack.com/t/anyscaleprevi-a4q8653/shared_invite/zt-2dfpjbnds-qw6jVYgG~HBeuanwtT9_tg) to share issues / questions.<br/>

## Set up

We can execute this notebook **entirely for free** (no credit card needed) by creating an [Anyscale account](https://console.anyscale.com/register/ha?utm_source=goku). Once you log in, you'll be directed to the main [console](https://console.anyscale.com/) where you'll see a collection of notebook templates. Click on the "End-to-end LLM Workflows" to open up our guide and click on the `README.ipynb` to get started.

> [Workspaces](https://docs.anyscale.com/workspaces/get-started/) are a fully managed development environment which allow us to use our favorite tools (VSCode, notebooks, terminal, etc.) on top of *infinite* compute (when we need it). In fact, by clicking on the compute at the top right (`✅ 1 node, 8 CPU`), we can see the cluster information:

- **Head node** (Workspace node): manages the cluster, distributes tasks, and hosts development tools.
- **Worker nodes**: machines that execute work orchestrated by the head node and can scale back to 0.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/setup-compute.png" width=550>

<b style="background-color: orange;">&nbsp;💡 INSIGHT&nbsp;</b>: Because we have `Auto-select worker nodes` enabled, that means that the required worker nodes (ex. GPU workers) will automagically be provisioned based on our workload's needs! They'll spin up, run the workload and then scale back to zero. This allows us to maintain a lean workspace environment (and only pay for compute when we need it) and completely remove the need to manage any infrastructure.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/auto-workers.png" width=350>

**Note**: we can explore all the metrics (ex. hardware util), logs, dashboards, manage dependencies (ex. images, pip packages, etc.) on the menu bar above.


```python
import os
import ray
import warnings
warnings.filterwarnings("ignore")
%load_ext autoreload
%autoreload 2
```

We'll need a free [Hugging Face token](https://huggingface.co/settings/tokens) to load our base LLMs and tokenizers. And since we are using Llama models, we need to login and accept the terms and conditions [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

<b style="background-color: yellow;">&nbsp;🔄 REPLACE&nbsp;</b>: Place your unique HF token below. If you accidentally ran this code block before pasting your HF token, then click the `Restart` button up top to restart the notebook kernel.


```python
# Initialize HF token
os.environ['HF_TOKEN'] = ''  # <-- replace with your token
ray.init(runtime_env={'env_vars': {'HF_TOKEN': os.environ['HF_TOKEN']}})
```

## Data Preprocessing

We'll start by preprocessing our data in preparation for fine-tuning our LLM. We'll use batch processing to apply our preprocessing across our dataset at scale.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/data-overview.png" width=500>

### Dataset

For our task, we'll be using the [Viggo dataset](https://huggingface.co/datasets/GEM/viggo), where the input (`meaning_representation`) is a structured collection of the overall intent (ex. `inform`) and entities (ex. `release_year`) and the output (`target`) is an unstructured sentence that incorporates all the structured input information. But for our task, we'll **reverse** this task where the input will be the unstructured sentence and the output will be the structured information.

```python
# Input (unstructured sentence):
"Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac."

# Output (intent + entities):
"inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+ (for Everyone 10 and Older)], genres[driving/racing, sport], platforms[PlayStation, Xbox, PC], available_on_steam[no], has_linux_release[no], has_mac_release[no])"
```


```python
from datasets import DatasetDict, load_dataset
ray.data.DataContext.get_current().enable_progress_bars = False
```


```python
# Load the VIGGO dataset
dataset: DatasetDict = load_dataset("GEM/viggo", trust_remote_code=True)  # type: ignore
```


```python
# Data splits
train_set = dataset['train']
val_set = dataset['validation']
test_set = dataset['test']
print (f"train: {len(train_set)}")
print (f"val: {len(val_set)}")
print (f"test: {len(test_set)}")
```

    train: 5103
    val: 714
    test: 1083



```python
# Sample
train_set[0]
```

```json
{
  "gem_id": "viggo-train-0",
  "meaning_representation": "inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+ (for Everyone 10 and Older)], genres[driving/racing, sport], platforms[PlayStation, Xbox, PC], available_on_steam[no], has_linux_release[no], has_mac_release[no])",
  "target": "Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac.",
  "references": [
    "Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac."
  ]
}
```


### Data Preprocessing

We'll use [Ray](https://docs.ray.io/) to load our dataset and apply preprocessing to batches of our data at scale.

```python
from ray.data import Dataset
```

```python
# Load as a Ray Dataset
train_ds = ray.data.from_items(train_set)
train_ds.take(1)
```

```json
[
  {
    "gem_id": "viggo-train-0",
    "meaning_representation": "inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+ (for Everyone 10 and Older)], genres[driving/racing, sport], platforms[PlayStation, Xbox, PC], available_on_steam[no], has_linux_release[no], has_mac_release[no])",
    "target": "Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac.",
    "references": [
      "Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac."
    ]
  }
]
```



The preprocessing we'll do involves formatting our dataset into the schema required for fine-tuning (`system`, `user`, `assistant`) conversations.

- `system`: description of the behavior or personality of the model. As a best practice, this should be the same for all examples in the fine-tuning dataset, and should remain the same system prompt when moved to production.
- `user`: user message, or "prompt," that provides a request for the model to respond to.
- `assistant`: stores previous responses but can also contain examples of intended responses for the LLM to return.

```yaml
conversations = [
    {"messages": [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': item['target']},
        {'role': 'assistant', 'content': item['meaning_representation']}]},
    {"messages": [...],}
    ...
]
```


```python
def to_schema(item, system_content):
    messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': item['target']},
        {'role': 'assistant', 'content': item['meaning_representation']}
    ]
    return {'messages': messages}
```

Our `system_content` will guide the LLM on how to behave. Our specific directions involve specifying the list of possible intents and entities to extract.


```python
# System content
system_content = (
    "Given a target sentence construct the underlying meaning representation of the input "
    "sentence as a single function with attributes and attribute values. This function "
    "should describe the target string accurately and the function must be one of the "
    "following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', "
    "'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes "
    "must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', "
    "'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', "
    "'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']"
)

```

To apply our function on our dataset at scale, we can pass it to [ray.data.Dataset.map](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map.html). Here, we can specify the function to apply to each sample in our data, what compute to use, etc. The diagram below shows how we can read from various data sources (ex. cloud storage) and apply operations at scale across different hardware (CPU, GPU). For our workload, we'll just use the default compute strategy which will use CPUs to scale out our workload.

**Note**: If we want to distribute a workload across `batches` of our data instead of individual samples, we can use [ray.data.Dataset.map_batches](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html). We'll see this in action when we perform batch inference in our evaluation template. There are also many other [distributed operations](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.html) we can perform on our dataset.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/data-detailed.png" width=800>



```python
# Distributed preprocessing
ft_train_ds: Dataset = train_ds.map(to_schema, fn_kwargs={'system_content': system_content})
ft_train_ds.take(1)
```



```json
[
  {
    "messages": [
      {
        "content": "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']",
        "role": "system"
      },
      {
        "content": "Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac.",
        "role": "user"
      },
      {
        "content": "inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+ (for Everyone 10 and Older)], genres[driving/racing, sport], platforms[PlayStation, Xbox, PC], available_on_steam[no], has_linux_release[no], has_mac_release[no])",
        "role": "assistant"
      }
    ]
  }
]
```




```python
# Repeat the steps for other splits
ft_val_ds: Dataset = ray.data.from_items(val_set).map(to_schema, fn_kwargs={'system_content': system_content})
ft_test_ds: Dataset = ray.data.from_items(test_set).map(to_schema, fn_kwargs={'system_content': system_content})
```

### Save and load data

We can save our data locally and/or to remote storage to use later (training, evaluation, etc.). All workspaces come with a default [cloud storage locations and shared storage](https://docs.anyscale.com/workspaces/storage) that we can write to.


```python
import anyscale
import os
from ray.data import Dataset
from rich import print as rprint
from src.utils import get_dataset_file_path
```

```python
# Upload as an Anyscale Dataset
def upload_dataset(dataset: Dataset, filename: str):
    with get_dataset_file_path(dataset) as dataset_file_path:
        dataset = anyscale.llm.dataset.upload(
            dataset_file_path,
            # john_doe/viggo/train.jsonl
            name=f"viggo/{filename}",
        )
    rprint(f"Metadata for '{filename}'")
    rprint(dataset)
    return dataset
```

```python
train_dataset = upload_dataset(ft_train_ds, 'train.jsonl')
val_dataset = upload_dataset(ft_val_ds, 'val.jsonl')
test_dataset = upload_dataset(ft_test_ds, 'test.jsonl')
```


```python
# Download as an Anyscale Dataset
ft_train_ds = ray.data.read_json(train_dataset.storage_uri)
ft_train_ds.take(1)
```
```json
[
  {
    "messages": [
      {
        "content": "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']",
        "role": "system"
      },
      {
        "content": "Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac.",
        "role": "user"
      },
      {
        "content": "inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+ (for Everyone 10 and Older)], genres[driving/racing, sport], platforms[PlayStation, Xbox, PC], available_on_steam[no], has_linux_release[no], has_mac_release[no])",
        "role": "assistant"
      }
    ]
  }
]
```

## Fine-tuning

In this template, we'll fine-tune a large language model (LLM) using our dataset from the previous data preprocessing template.

**Note**: We normally would not jump straight to fine-tuning a model. We would first experiment with a base model and evaluate it so that we can have a baseline performance to compare it to.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/train-overview.png" width=500>

### Configurations

We'll fine-tune our LLM by choosing a set of configurations. We have created recipes for different LLMs in the [`training configs`](configs/training/lora/llama-3-8b.yaml) folder which can be used as is or modified for experiments. These configurations provide flexibility over a broad range of parameters such as model, data paths, compute to use for training, number of training epochs, how often to save checkpoints, padding, loss, etc. We also include several [DeepSpeed](https://github.com/microsoft/DeepSpeed) [configurations](configs/deepspeed/zero_3_offload_optim+param.json) to choose from for further optimizations around data/model parallelism, mixed precision, checkpointing, etc.

We also have recipes for [LoRA](https://arxiv.org/abs/2106.09685) (where we train a set of small low ranked matrices instead of the original attention and feed forward layers) or full parameter fine-tuning. We recommend starting with LoRA as it's less resource intensive and quicker to train.


```python
# View the training (LoRA) configuration for llama-3-8B
!cat configs/training/lora/llama-3-8b.yaml
```
```yaml
    model_id: meta-llama/Meta-Llama-3-8B-Instruct # <-- change this to the model you want to fine-tune
    train_path: s3://llm-guide/data/viggo/train.jsonl # <-- change this to the path to your training data
    valid_path: s3://llm-guide/data/viggo/val.jsonl # <-- change this to the path to your validation data. This is optional
    context_length: 512 # <-- change this to the context length you want to use
    num_devices: 16 # <-- change this to total number of GPUs that you want to use
    num_epochs: 4 # <-- change this to the number of epochs that you want to train for
    train_batch_size_per_device: 16
    eval_batch_size_per_device: 16
    learning_rate: 1e-4
    padding: "longest" # This will pad batches to the longest sequence. Use "max_length" when profiling to profile the worst case.
    num_checkpoints_to_keep: 1
    output_dir: /mnt/local_storage
    deepspeed:
      config_path: configs/deepspeed/zero_3_offload_optim+param.json
    flash_attention_2: true
    trainer_resources:
      memory: 53687091200 # 50 GB memory
    worker_resources:
      accelerator_type:A10G: 0.001
    lora_config:
      r: 8
      lora_alpha: 16
      lora_dropout: 0.05
      target_modules:
        - q_proj
        - v_proj
        - k_proj
        - o_proj
        - gate_proj
        - up_proj
        - down_proj
        - embed_tokens
        - lm_head
      task_type: "CAUSAL_LM"
      modules_to_save: []
      bias: "none"
      fan_in_fan_out: false
      init_lora_weights: true
```

### Fine-tuning

This Workspace is still running on a small, lean head node. But based on the compute we want to use (ex. `num_devices` and `accelerator_type`) for fine-tuning, the appropriate worker nodes will automatically be initialized and execute the workload. And afterwards, they'll scale back to zero!

<b style="background-color: orange;">&nbsp;💡 INSIGHT&nbsp;</b>: With [Ray](https://docs.ray.io/) we're able to execute a large, compute intensive workload like this using smaller, more available resources (ex. using `A10`s instead of waiting for elusive `A100`s). And Anyscale's smart instance manager will automatically provision the appropriate and available compute for the workload based on what's needed.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/train-detailed.png" width=550>

While we could execute `llmforge anyscale finetune configs/training/lora/llama-3-8b.yaml` directly inside a Workspace notebook (see this [example](https://console.anyscale.com/v2/template-preview/finetuning_llms_v2)), we'll instead kick off the fine-tuning workload as an isolated job. An [Anyscale Job](https://docs.anyscale.com/jobs/get-started/) is a great way to scale and execute a specific workload. Here, we specify the command that needs to run (ex. `python [COMMAND][ARGS]`) along with the requirements (ex. docker image, additional, pip packages, etc.).

**Note**: Executing an Anyscale Job within a Workspace will ensure that files in the current working directory are available for the Job (unless excluded with `--exclude`). But we can also load files from anywhere (ex. Github repo, S3, etc.) if we want to launch a Job from anywhere.


```python
# View job yaml config
!cat deploy/jobs/ft.yaml
```

```yaml
    name: e2e-llm-workflows
    entrypoint: llmforge anyscale finetune configs/training/lora/llama-3-8b.yaml
    image_uri: localhost:5555/anyscale/llm-forge:0.5.4
    requirements: []
    max_retries: 1
    excludes: ["assets"]
```

**Note**: Be sure to checkout the fine-tuning documentation for the latest on how to use our [API](https://docs.anyscale.com/llms/finetuning/intro) and additional [capabilities](https://docs.anyscale.com/category/fine-tuning-beta/).

<b style="background-color: orange;">&nbsp;💡 INSIGHT&nbsp;</b>: When defining this [Job config](https://docs.anyscale.com/reference/job-api/), if we don't specify the [compute config](https://docs.anyscale.com/configure/compute-configs/overview/) to use, then Anyscale will autoselect based on the required compute. However, we also have the optionality to specify and even make highly cost effective decisions such as [spot to on-demand fallback](https://docs.anyscale.com/configure/compute-configs/ondemand-to-spot-fallback/) (or vice-versa).


```yaml
# Sample compute config
- name: gpu-worker-a10
  instance_type: g5.2xlarge
  min_workers: 0
  max_workers: 16
  use_spot: true
  fallback_to_ondemand: true
```


```python
import anyscale
from anyscale.job import JobConfig
```
```python
# Job submission
job_config = JobConfig.from_yaml("deploy/jobs/ft.yaml")
job_id = anyscale.job.submit(job_config)
```

    (anyscale +17m7.0s) Uploading local dir '.' to cloud storage.
    (anyscale +17m8.7s) Job 'e2e-llm-workflows' submitted, ID: 'prodjob_q1tzjcngnrwrp2yzpnbh4v7n8w'.
    (anyscale +17m8.7s) View the job in the UI: https://console.anyscale.com/jobs/prodjob_q1tzjcngnrwrp2yzpnbh4v7n8w


As the job runs, you can monitor logs, metrics, Ray dashboard, etc. by clicking on the generated Job link above (`https://console.anyscale.com/jobs/prodjob_...`)

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/gpu-util.png" width=800>

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/tensorboard.png" width=800>

### Load artifacts

To retrieve information about your fine-tuned model, Anyscale provides a convenient SDK.

<b>Note</b>: Wait for your fine-tuning job to finish first and then run the code below to programatically retrieve the model information.


```python
import anyscale
model_info = anyscale.llm.model.get(job_id=job_id)
print(model_info)
```

`model_info` has a number of helpful model metadata, such as the id `meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli` , the base model ID, storage URI for the final checkpoint, the model generation config, etc.

```
FineTunedModel(
    id='meta-llama/Meta-Llama-3-8B-Instruct:user:suffix',
    base_model_id='meta-llama/Meta-Llama-3-8B-Instruct',
    cloud_id='cld_123',
    created_at=1726013024,
    creator='test@anyscale.com',
    ft_type='LORA',
    generation_config={
        'prompt_format': {
            'system': '<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>',
            'assistant': '<|start_header_id|>assistant<|end_header_id|>\n\n{instruction}<|eot_id|>',
            'trailing_assistant': '<|start_header_id|>assistant<|end_header_id|>\n\n',
            'user': '<|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>',
            'bos': '<|begin_of_text|>',
            'default_system_message': '',
            'add_system_tags_even_if_message_is_empty': False,
            'system_in_user': False,
            'system_in_last_user': False,
            'strip_whitespace': True
        },
        'stopping_sequences': ['<|eot_id|>']
    },
    job_id='prodjob_123',
    project_id='prj_123',
    storage_uri='s3://my_bucket/my_folder',
    workspace_id=None
)
```


The storage URI for the best checkpoint can look like:

```
s3://anyscale-production-data-cld-ldm5ez4edlp7yh4yiakp2u294w/org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/goku__mohandas_gkdbtlxwnwirhpgqqzhawjazxxldhngwkxoi/llmforge-finetuning/meta-llama/Meta-Llama-3-8B-Instruct/TorchTrainer_2024-09-10_16-35-42/epoch-3
```

Note that with LoRA, we automatically forward this checkpoint to a common folder in [artifact storage](https://docs.anyscale.com/platform/workspaces/workspaces-storage#object-storage-s3-or-gcs-buckets): `{ANYSCALE_ARTIFACT_STORAGE}/lora_fine_tuning` . This becomes extremely helpful while serving LoRA checkpoints, which we'll see soon.

This information about the final checkpoint is also available in the logs for the job. For example, you might see:
```
Successfully copied files to bucket: anyscale-customer-dataplane-data-production-us-east-2 and path: artifact_storage/org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli
```

We'll now load the checkpoint from cloud storage to a local [cluster storage](https://docs.anyscale.com/workspaces/storage/#cluster-storage) to use for other workloads.


```python
from src.utils import download_files_from_remote
```

```python
# Locations
artifacts_dir = '/mnt/cluster_storage'  # storage accessible by head and worker nodes
model_id = model_info.id
artifacts_path = f"{os.environ['ANYSCALE_ARTIFACT_STORAGE']}/lora_fine_tuning/{model_id}"
```


```python
# Download artifacts
download_files_from_remote(
    uri=artifacts_path,
    local_dir=artifacts_dir)
```

    Downloaded org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/README.md to /mnt/cluster_storage/org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/README.md
    Downloaded org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/adapter_config.json to /mnt/cluster_storage/org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/adapter_config.json
    Downloaded org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/adapter_model.safetensors to /mnt/cluster_storage/org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/adapter_model.safetensors
    Downloaded org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/config.json to /mnt/cluster_storage/org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/config.json
    Downloaded org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/new_embeddings.safetensors to /mnt/cluster_storage/org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/new_embeddings.safetensors
    Downloaded org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/rayllm_generation_config.json to /mnt/cluster_storage/org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/rayllm_generation_config.json
    Downloaded org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/special_tokens_map.json to /mnt/cluster_storage/org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/special_tokens_map.json
    Downloaded org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/tokenizer.json to /mnt/cluster_storage/org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/tokenizer.json
    Downloaded org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/tokenizer_config.json to /mnt/cluster_storage/org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli/tokenizer_config.json


## Evaluation

Now we'll evaluate our fine-tuned LLM to see how well it performs on our task. There are a lot of different ways to perform evaluation. For our task, we can use traditional metrics (ex. accuracy, precision, recall, etc.) since we know what the outputs should be (extracted intent and entities).

However for many generative tasks, the outputs are very unstructured and highly subjective. For these scenarios, we can use [distance/entropy](https://github.com/huggingface/evaluate) based metrics like cosine, bleu, perplexity, etc. But, these metrics are often not very representative of the underlying task. A common strategy here is to use a larger LLM to [judge the quality](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1#evaluation) of the generated outputs. We can ask the larger LLM to directly assess the quality of the response (ex. rate between `1-5`) with a set of rules or compare it to a golden / preferred output and rate it against that.

We'll start by performing offline batch inference where we will use our tuned model to generate the outputs.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/offline-overview.png" width=500>

### Load test data


```python
# Load test set for eval
ft_test_ds = ray.data.read_json(test_dataset.storage_uri)
test_data = ft_test_ds.take_all()
test_data[0]
```

```json
{
  "messages": [
    {
      "content": "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']",
      "role": "system"
    },
    {
      "content": "Have you ever given any games on PC but not on Steam a try, like The Sims?",
      "role": "user"
    },
    {
      "content": "suggest(name[The Sims], platforms[PC], available_on_steam[no])",
      "role": "assistant"
    }
  ]
}
```



```python
# Separate into inputs/outputs
test_inputs = []
test_outputs = []
for item in test_data:
    test_inputs.append([message for message in item['messages'] if message['role'] != 'assistant'])
    test_outputs.append([message for message in item['messages'] if message['role'] == 'assistant'])
```

### Tokenizer

We'll also load the appropriate tokenizer to apply to our input data.


```python
from transformers import AutoTokenizer
```


```python
# Model and tokenizer
HF_MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
```

### Chat template

When we fine-tuned our model, special tokens (ex. beginning/end of text, etc.) were automatically added to our inputs. We want to apply the same special tokens to our inputs prior to generating outputs using our tuned model. Luckily, the chat template to apply to our inputs (and add those tokens) is readily available inside our tuned model's `tokenizer_config.json` file. We can use our tokenizer to apply this template to our inputs.


```python
import json
```


```python
# Extract chat template used during fine-tuning
artifacts_path = artifacts_path.split('/', 3)[-1]
with open(os.path.join(artifacts_dir, artifacts_path, 'tokenizer_config.json')) as file:
    tokenizer_config = json.load(file)
chat_template = tokenizer_config['chat_template']
print (chat_template)
```

    {% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>

    '+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>

    ' }}{% endif %}



```python
# Apply chat template
test_input_prompts = [{'inputs': tokenizer.apply_chat_template(
    conversation=inputs,
    chat_template=chat_template,
    add_generation_prompt=True,
    tokenize=False,
    return_tensors='np'), 'outputs': outputs} for inputs, outputs in zip(test_inputs, test_outputs)]
test_input_prompts_ds = ray.data.from_items(test_input_prompts)
print (test_input_prompts_ds.take(1))
```

```json
[
  {
    "inputs": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nGiven a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHave you ever given any games on PC but not on Steam a try, like The Sims?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "outputs": [
      {
        "content": "suggest(name[The Sims], platforms[PC], available_on_steam[no])",
        "role": "assistant"
      }
    ]
  }
]
```


### Batch inference

We will use [vLLM](https://github.com/vllm-project/vllm)'s offline LLM class to load the model and use it for inference. We can easily load our LoRA weights and merge them with the base model (just pass in `lora_path`). And we'll wrap all of this functionality in a class that we can pass to [ray.data.Dataset.map_batches](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html) to apply batch inference at scale.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/offline-detailed.png" width=750>


```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
```


```python
class LLMPredictor:
    def __init__(self, hf_model, sampling_params, lora_path=None):
        self.llm = LLM(model=hf_model, enable_lora=bool(lora_path))
        self.sampling_params = sampling_params
        self.lora_path = lora_path

    def __call__(self, batch):
        if not self.lora_path:
            outputs = self.llm.generate(
                prompts=batch['inputs'],
                sampling_params=self.sampling_params)
        else:
            outputs = self.llm.generate(
                prompts=batch['inputs'],
                sampling_params=self.sampling_params,
                lora_request=LoRARequest('lora_adapter', 1, self.lora_path))
        inputs = []
        generated_outputs = []
        for output in outputs:
            inputs.append(output.prompt)
            generated_outputs.append(' '.join([o.text for o in output.outputs]))
        return {
            'prompt': inputs,
            'expected_output': batch['outputs'],
            'generated_text': generated_outputs,
        }
```

During our data preprocessing template, we used the default compute strategy with `map_batches`. But this time we'll specify a custom compute strategy (`concurrency`, `num_gpus`, `batch_size` and `accelerator_type`).


```python
# Fine-tuned model
hf_model = 'meta-llama/Meta-Llama-3-8B-Instruct'
sampling_params = SamplingParams(temperature=0, max_tokens=2048)
ft_pred_ds = test_input_prompts_ds.map_batches(
    LLMPredictor,
    concurrency=4,  # number of LLM instances
    num_gpus=1,  # GPUs per LLM instance
    batch_size=10,  # maximize until OOM, if OOM then decrease batch_size
    fn_constructor_kwargs={
        'hf_model': hf_model,
        'sampling_params': sampling_params,
        'lora_path': os.path.join(artifacts_dir, artifacts_path)
    },
    accelerator_type='A10G',  # A10G or L4
)
```


```python
# Batch inference will take ~4 minutes
ft_pred = ft_pred_ds.take_all()
ft_pred[3]
```

```json
{
  "prompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nGiven a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI like first person games normally, but not even that could make a music game fun for me. In fact in Guitar Hero: Smash Hits, I think the perspective somehow made an already bad game even worse.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
  "expected_output": [
    {
      "content": "give_opinion(name[Guitar Hero: Smash Hits], rating[poor], genres[music], player_perspective[first person])",
      "role": "assistant"
    }
  ],
  "generated_text": "give_opinion(name[Guitar Hero: Smash Hits], rating[poor], genres[music], player_perspective[first person])<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
}
```


### Evaluation


```python
# Exact match (strict!)
matches = 0
mismatches = []
for item in ft_pred:
    if item['expected_output'][0]['content'] == item['generated_text'].split('<|eot_id|>')[0]:
        matches += 1
    else:
        mismatches.append(item)
matches / float(len(ft_pred))
```




    0.9399815327793167



**Note**: you can train for more epochs (`num_epochs: 10`) to further improve the performance.

Even our mismatches are not too far off and sometimes it might be worth a closer look because the dataset itself might have a few errors that the model may have identified.


```python
# Inspect a few of the mismatches
mismatches[0:2]
```


## Serving

For model serving, we'll first serve it locally, test it and then launch a production grade service that can autoscale to meet any demand.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/online-overview.png" width=500>

We'll start by generating the configuration for our service. We provide a convenient CLI experience to generate this configuration but you can create one from scratch as well. Here we can specify where our model lives, autoscaling behavior, accelerators to use, lora adapters, etc.

<b style="background-color: orange;">&nbsp;💡 INSIGHT&nbsp;</b>: Ray Serve and Anyscale support [serving multiple LoRA adapters](https://github.com/anyscale/templates/blob/main/templates/endpoints_v2/examples/lora/DeployLora.ipynb) with a common base model in the same request batch which allows you to serve a wide variety of use-cases without increasing hardware spend. In addition, we use Serve multiplexing to reduce the number of swaps for LoRA adapters. There is a slight latency overhead to serving a LoRA model compared to the base model, typically 10-20%.


We can use the model metadata `model_info` for the model ID. For serving, we'll use the root folder for the LoRA checkpoints.

**model**: `meta-llama/Meta-Llama-3-8B-Instruct:gokum:yehli` (`model_info.id`)

**LoRA weights storage URI**: `s3://org_4snvy99zwbmh4gbtk64jfqggmj/cld_ldm5ez4edlp7yh4yiakp2u294w/artifact_storage/lora_fine_tuning` ( or `{ANYSCALE_ARTIFACT_STORAGE}/lora_fine_tuning`)

We'll start by running the rayllm CLI command below to start the workflow to generate the service yaml configuration:
```bash
mkdir /home/ray/default/deploy/services
cd /home/ray/default/deploy/services
rayllm gen-config
```

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/cli.png" width=500>

<b style="background-color: yellow;">&nbsp;🔄 REPLACE&nbsp;</b>: Use the serve configuration generated for you.


```python
# Generated service configuration
!cat /home/ray/default/deploy/services/serve_{TIMESTAMP}.yaml
```

    applications:
    - args:
        llm_configs:
        - ./model_config/meta-llama--Meta-Llama-3-8B-Instruct_20240825011129.yaml
      import_path: rayllm:app
      name: llm-endpoint
      route_prefix: /
    query_auth_token_enabled: false


This also generates a model configuration file that has all the information on auto scaling, inference engine, workers, compute, etc. It will be located under `/home/ray/default/deploy/services/model_config/{MODEL_NAME}-{TIMESTAMP}.yaml`. This configuration also includes the `prompt_format` which seamlessly matches any formatting we did prior to fine-tuning and applies it during inference automatically.

### Local deployment

We can now serve our model locally and query it. Run the follow in the terminal (change to your serve yaml config):

```bash
cd /home/ray/default/deploy/services
serve run serve_{TIMESTAMP}.yaml
```

**Note**: This will take a few minutes to spin up the first time since we're loading the model weights.


```python
from openai import OpenAI
```


```python
# Query function to call the running service
def query(base_url: str, api_key: str):
    if not base_url.endswith("/"):
        base_url += "/"

    # List all models
    client = OpenAI(base_url=base_url + "v1", api_key=api_key)
    models = client.models.list()
    print(models)

    # Note: not all arguments are currently supported and will be ignored by the backend
    chat_completions = client.chat.completions.create(
        model=model_info.id,  # with your unique model ID
        messages=[
            {"role": "system", "content": "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']"},
            {"role": "user", "content": "I remember you saying you found Little Big Adventure to be average. Are you not usually that into single-player games on PlayStation?"},
        ],
        temperature=0,
        stream=True
    )

    response = ""
    for chat in chat_completions:
        if chat.choices[0].delta.content is not None:
            response += chat.choices[0].delta.content
    return response
```


```python
# Generate response
response = query("http://localhost:8000", "NOT A REAL KEY")
print(response.split('<|eot_id|>')[0])
```




    'verify_attribute(name[Little Big Adventure], rating[average], has_multiplayer[no], platforms[PlayStation])'



### Production service

Now we'll create a production service that can truly scale. We have full control over this Service from autoscaling behavior, monitoring via dashboard, canary rollouts, termination, etc. → [Anyscale Services](https://docs.anyscale.com/examples/intro-services/)

<b style="background-color: orange;">&nbsp;💡 INSIGHT&nbsp;</b>: With Ray Serve and Anyscale, it's extremely easy to define our configuration that can scale to meet any demand but also scale back to zero to create the most efficient service possible. Check out this [guide](https://github.com/anyscale/templates/blob/main/templates/endpoints_v2/examples/OptimizeModels.ipynb) on how to optimize behavior around auto scaling, latency/throughout, etc.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/online-detailed.png" width=650>

Stop the local service (Control + C) and run the following:
```bash
cd /home/ray/default/deploy/services
anyscale service deploy -f serve_{TIMESTAMP}.yaml
```

**Note**: This will take a few minutes to spin up the first time since we're loading the model weights.


Go to `Home` > `Services` (left panel) to view the production service.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/services.png" width=650>

<b style="background-color: yellow;">&nbsp;🔄 REPLACE&nbsp;</b>: the `service_url` and `service_bearer_token` generated for your service (top right corner under the `Query` button on the Service's page).


```python
# Query the remote serve application we just deployed
service_url = "your_api_url"  # REPLACE ME
service_bearer_token = "your_secret_bearer_token"  # REPLACE ME
query(service_url, service_bearer_token)
```




    'verify_attribute(name[Little Big Adventure], rating[average], has_multiplayer[no], platforms[PlayStation])'



**Note**: If we chose to fine-tune our model using the simpler [Anyscale serverless endpoints](https://docs.anyscale.com/endpoints/fine-tuning/fine-tuning-api/) method, then we can serve that model by going to `Endpoints API > Services` on the left panel of the main [console page](https://console.anyscale.com/). Click on the three dots on the right side of your tuned model and follow the instructions to query it.

## Dev → Prod

We've now served our model into production via [Anyscale Services](https://docs.anyscale.com/examples/intro-services/) but we can just easily productionize our other workloads with [Anyscale Jobs](https://docs.anyscale.com/examples/intro-jobs/) (like we did for fine-tuning above) to execute this entire workflow completely programmatically outside of Workspaces.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/jobs.png" width=650>

For example, suppose that we want to preprocess batches of new incoming data, fine-tune a model, evaluate it and then compare it to the existing production version. All of this can be productionized by simply launching the workload as a [Job](https://docs.anyscale.com/examples/intro-jobs), which can be triggered manually, periodically (cron) or event-based (via webhooks, etc.). We also provide integrations with your platform/tools to make all of this connect with your existing production workflows.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/ai-platform.png" width=650>

<b style="background-color: orange;">&nbsp;💡 INSIGHT&nbsp;</b>: Most industry ML issues arise from a discrepancy between the development (ex. local laptop) and production (ex. large cloud clusters) environments. With Anyscale, your development and production environments can be exactly the same so there is little to no difference introduced. And with features like smart instance manager, the development environment can stay extremely lean while having the power to scale as needed.

## Clean up

<b style="background-color: yellow;">&nbsp;🛑 IMPORTANT&nbsp;</b>: Please `Terminate` your service from the Service page to avoid depleting your free trial credits.


```python
# Clean up
!python src/clear_cell_nums.py
!find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
!find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
!rm -rf __pycache__ data .HF_TOKEN deploy/services
```

## Next steps

We have a lot more guides that address more nuanced use cases:

Fine-tuning:
- [Control over 50+ hyperparameters](https://docs.anyscale.com/llms/finetuning/guides/modify_hyperparams/)
- [Fine-tune any HF model](https://docs.anyscale.com/llms/finetuning/guides/bring_any_hf_model/)
- [Full-parameter or LoRA fine-tuning](https://docs.anyscale.com/llms/finetuning/guides/lora_vs_full_param/)
- [Classification fine-tuning / Routing](https://www.anyscale.com/blog/building-an-llm-router-for-high-quality-and-cost-effective-responses)
- [Function calling fine-tuning](https://github.com/anyscale/templates/blob/main/templates/fine-tune-llm_v2/end-to-end-examples/fine-tune-function-calling/README.ipynb)
- [Longer context fine-tuning](https://www.anyscale.com/blog/fine-tuning-llms-for-longer-context-and-better-rag-systems)
- [Continued fine-tuning from checkpoint](https://github.com/anyscale/templates/tree/main/templates/fine-tune-llm_v2/cookbooks/continue_from_checkpoint)
- Training on more available hardware (ex. A10s) with model parallelism
- [End-to-end LLM workflows (including batch data processing, batch inference)](https://www.anyscale.com/blog/end-to-end-llm-workflows-guide)
- Distillation (Coming in <2 weeks)

Serving:
- [Deploy with autoscaling + optimize for latency vs. throughput](https://docs.anyscale.com/examples/deploy-llms/)
- [Serving multiple LoRA adapters](https://docs.anyscale.com/llms/serving/guides/multi_lora/)
- [Migration from OpenAI](https://docs.anyscale.com/llms/serving/guides/openai_to_oss/)
- [Spot to on-demand fallback (vice versa)](https://docs.anyscale.com/1.0.0/configure/compute-configs/ondemand-to-spot-fallback/)
- [Batch inference with vLLM](https://docs.anyscale.com/examples/batch-llm/)

And more!
- [Batch text embeddings with Ray data](https://github.com/anyscale/templates/tree/main/templates/text-embeddings)
- [Production RAG applications](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)
- [Router](https://github.com/anyscale/llm-router) between different models (base, fine-tuned, closed-source) to optimize for cost and quality
- Stable diffusion [fine-tuning](https://github.com/anyscale/templates/tree/main/templates/fine-tune-stable-diffusion) and [serving](https://github.com/anyscale/templates/tree/main/templates/serve-stable-diffusion)

And if you're interested in using our hosted Anyscale or connecting it to your own cloud, reach out to us at [Anyscale](https://www.anyscale.com/get-started?utm_source=goku). And follow us on [Twitter](https://x.com/anyscalecompute) and [LinkedIn](https://www.linkedin.com/company/joinanyscale/) for more real-time updates on new features!
