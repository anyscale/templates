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

    2024-06-12 16:52:33,877	INFO worker.py:1564 -- Connecting to existing Ray cluster at address: 10.0.46.70:6379...
    2024-06-12 16:52:33,883	INFO worker.py:1740 -- Connected to Ray cluster. View the dashboard at [1m[32mhttps://session-zdbj1t4fe6firefy7rxgrcyj7c.i.anyscaleuserdata.com [39m[22m
    2024-06-12 16:52:33,896	INFO packaging.py:358 -- Pushing file package 'gcs://_ray_pkg_f8c87dcbafb22dcfc23fd80fe43ba56d14d0593c.zip' (2.56MiB) to Ray cluster...
    2024-06-12 16:52:33,906	INFO packaging.py:371 -- Successfully pushed file package 'gcs://_ray_pkg_f8c87dcbafb22dcfc23fd80fe43ba56d14d0593c.zip'.





<div class="lm-Widget p-Widget lm-Panel p-Panel jp-Cell-outputWrapper">
    <div style="margin-left: 50px;display: flex;flex-direction: row;align-items: center">
        <div class="jp-RenderedHTMLCommon" style="display: flex; flex-direction: row;">
  <svg viewBox="0 0 567 224" fill="none" xmlns="http://www.w3.org/2000/svg" style="height: 3em;">
    <g clip-path="url(#clip0_4338_178347)">
        <path d="M341.29 165.561H355.29L330.13 129.051C345.63 123.991 354.21 112.051 354.21 94.2307C354.21 71.3707 338.72 58.1807 311.88 58.1807H271V165.561H283.27V131.661H311.8C314.25 131.661 316.71 131.501 319.01 131.351L341.25 165.561H341.29ZM283.29 119.851V70.0007H311.82C331.3 70.0007 342.34 78.2907 342.34 94.5507C342.34 111.271 331.34 119.861 311.82 119.861L283.29 119.851ZM451.4 138.411L463.4 165.561H476.74L428.74 58.1807H416L367.83 165.561H380.83L392.83 138.411H451.4ZM446.19 126.601H398L422 72.1407L446.24 126.601H446.19ZM526.11 128.741L566.91 58.1807H554.35L519.99 114.181L485.17 58.1807H472.44L514.01 129.181V165.541H526.13V128.741H526.11Z" fill="var(--jp-ui-font-color0)"/>
        <path d="M82.35 104.44C84.0187 97.8827 87.8248 92.0678 93.1671 87.9146C98.5094 83.7614 105.083 81.5067 111.85 81.5067C118.617 81.5067 125.191 83.7614 130.533 87.9146C135.875 92.0678 139.681 97.8827 141.35 104.44H163.75C164.476 101.562 165.622 98.8057 167.15 96.2605L127.45 56.5605C121.071 60.3522 113.526 61.6823 106.235 60.3005C98.9443 58.9187 92.4094 54.9203 87.8602 49.0574C83.3109 43.1946 81.0609 35.8714 81.5332 28.4656C82.0056 21.0599 85.1679 14.0819 90.4252 8.8446C95.6824 3.60726 102.672 0.471508 110.08 0.0272655C117.487 -0.416977 124.802 1.86091 130.647 6.4324C136.493 11.0039 140.467 17.5539 141.821 24.8501C143.175 32.1463 141.816 39.6859 138 46.0505L177.69 85.7505C182.31 82.9877 187.58 81.4995 192.962 81.4375C198.345 81.3755 203.648 82.742 208.33 85.3976C213.012 88.0532 216.907 91.9029 219.616 96.5544C222.326 101.206 223.753 106.492 223.753 111.875C223.753 117.258 222.326 122.545 219.616 127.197C216.907 131.848 213.012 135.698 208.33 138.353C203.648 141.009 198.345 142.375 192.962 142.313C187.58 142.251 182.31 140.763 177.69 138L138 177.7C141.808 184.071 143.155 191.614 141.79 198.91C140.424 206.205 136.44 212.75 130.585 217.313C124.731 221.875 117.412 224.141 110.004 223.683C102.596 223.226 95.6103 220.077 90.3621 214.828C85.1139 209.58 81.9647 202.595 81.5072 195.187C81.0497 187.779 83.3154 180.459 87.878 174.605C92.4405 168.751 98.9853 164.766 106.281 163.401C113.576 162.035 121.119 163.383 127.49 167.19L167.19 127.49C165.664 124.941 164.518 122.182 163.79 119.3H141.39C139.721 125.858 135.915 131.673 130.573 135.826C125.231 139.98 118.657 142.234 111.89 142.234C105.123 142.234 98.5494 139.98 93.2071 135.826C87.8648 131.673 84.0587 125.858 82.39 119.3H60C58.1878 126.495 53.8086 132.78 47.6863 136.971C41.5641 141.163 34.1211 142.972 26.7579 142.059C19.3947 141.146 12.6191 137.574 7.70605 132.014C2.79302 126.454 0.0813599 119.29 0.0813599 111.87C0.0813599 104.451 2.79302 97.2871 7.70605 91.7272C12.6191 86.1673 19.3947 82.5947 26.7579 81.6817C34.1211 80.7686 41.5641 82.5781 47.6863 86.7696C53.8086 90.9611 58.1878 97.2456 60 104.44H82.35ZM100.86 204.32C103.407 206.868 106.759 208.453 110.345 208.806C113.93 209.159 117.527 208.258 120.522 206.256C123.517 204.254 125.725 201.276 126.771 197.828C127.816 194.38 127.633 190.677 126.253 187.349C124.874 184.021 122.383 181.274 119.205 179.577C116.027 177.88 112.359 177.337 108.826 178.042C105.293 178.746 102.113 180.654 99.8291 183.44C97.5451 186.226 96.2979 189.718 96.3 193.32C96.2985 195.364 96.7006 197.388 97.4831 199.275C98.2656 201.163 99.4132 202.877 100.86 204.32ZM204.32 122.88C206.868 120.333 208.453 116.981 208.806 113.396C209.159 109.811 208.258 106.214 206.256 103.219C204.254 100.223 201.275 98.0151 197.827 96.97C194.38 95.9249 190.676 96.1077 187.348 97.4873C184.02 98.8669 181.274 101.358 179.577 104.536C177.879 107.714 177.337 111.382 178.041 114.915C178.746 118.448 180.653 121.627 183.439 123.911C186.226 126.195 189.717 127.443 193.32 127.44C195.364 127.443 197.388 127.042 199.275 126.259C201.163 125.476 202.878 124.328 204.32 122.88ZM122.88 19.4205C120.333 16.8729 116.981 15.2876 113.395 14.9347C109.81 14.5817 106.213 15.483 103.218 17.4849C100.223 19.4868 98.0146 22.4654 96.9696 25.9131C95.9245 29.3608 96.1073 33.0642 97.4869 36.3922C98.8665 39.7202 101.358 42.4668 104.535 44.1639C107.713 45.861 111.381 46.4036 114.914 45.6992C118.447 44.9949 121.627 43.0871 123.911 40.301C126.195 37.515 127.442 34.0231 127.44 30.4205C127.44 28.3772 127.038 26.3539 126.255 24.4664C125.473 22.5788 124.326 20.8642 122.88 19.4205ZM19.42 100.86C16.8725 103.408 15.2872 106.76 14.9342 110.345C14.5813 113.93 15.4826 117.527 17.4844 120.522C19.4863 123.518 22.4649 125.726 25.9127 126.771C29.3604 127.816 33.0638 127.633 36.3918 126.254C39.7198 124.874 42.4664 122.383 44.1635 119.205C45.8606 116.027 46.4032 112.359 45.6988 108.826C44.9944 105.293 43.0866 102.114 40.3006 99.8296C37.5145 97.5455 34.0227 96.2983 30.42 96.3005C26.2938 96.3018 22.337 97.9421 19.42 100.86ZM100.86 100.86C98.3125 103.408 96.7272 106.76 96.3742 110.345C96.0213 113.93 96.9226 117.527 98.9244 120.522C100.926 123.518 103.905 125.726 107.353 126.771C110.8 127.816 114.504 127.633 117.832 126.254C121.16 124.874 123.906 122.383 125.604 119.205C127.301 116.027 127.843 112.359 127.139 108.826C126.434 105.293 124.527 102.114 121.741 99.8296C118.955 97.5455 115.463 96.2983 111.86 96.3005C109.817 96.299 107.793 96.701 105.905 97.4835C104.018 98.2661 102.303 99.4136 100.86 100.86Z" fill="#00AEEF"/>
    </g>
    <defs>
        <clipPath id="clip0_4338_178347">
            <rect width="566.93" height="223.75" fill="white"/>
        </clipPath>
    </defs>
  </svg>
</div>

        <table class="jp-RenderedHTMLCommon" style="border-collapse: collapse;color: var(--jp-ui-font-color1);font-size: var(--jp-ui-font-size1);">
    <tr>
        <td style="text-align: left"><b>Python version:</b></td>
        <td style="text-align: left"><b>3.9.19</b></td>
    </tr>
    <tr>
        <td style="text-align: left"><b>Ray version:</b></td>
        <td style="text-align: left"><b>2.22.0</b></td>
    </tr>
    <tr>
    <td style="text-align: left"><b>Dashboard:</b></td>
    <td style="text-align: left"><b><a href="http://session-zdbj1t4fe6firefy7rxgrcyj7c.i.anyscaleuserdata.com" target="_blank">http://session-zdbj1t4fe6firefy7rxgrcyj7c.i.anyscaleuserdata.com</a></b></td>
</tr>

</table>

    </div>
</div>




## Data Preprocessing

We'll start by preprocessing our data in preparation for fine-tuning our LLM. We'll use batch processing to apply our preprocessing across our dataset at scale.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/data-overview.png" width=500>

### Dataset

For our task, we'll be using the [Viggo dataset](https://huggingface.co/datasets/GEM/viggo) dataset, where the input (`meaning_representation`) is a structured collection of the overall intent (ex. `inform`) and entities (ex. `release_year`) and the output (`target`) is an unstructured sentence that incorporates all the structured input information. But for our task, we'll **reverse** this task where the input will be the unstructured sentence and the output will be the structured information.

```python
# Input (unstructured sentence):
"Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac."

# Output (intent + entities): 
"inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+ (for Everyone 10 and Older)], genres[driving/racing, sport], platforms[PlayStation, Xbox, PC], available_on_steam[no], has_linux_release[no], has_mac_release[no])"
```


```python
from datasets import load_dataset
ray.data.set_progress_bars(enabled=False)
```


```python
# Load the VIGGO dataset
dataset = load_dataset("GEM/viggo", trust_remote_code=True)
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




    {'gem_id': 'viggo-train-0',
     'meaning_representation': 'inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+ (for Everyone 10 and Older)], genres[driving/racing, sport], platforms[PlayStation, Xbox, PC], available_on_steam[no], has_linux_release[no], has_mac_release[no])',
     'target': "Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac.",
     'references': ["Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac."]}



### Data Preprocessing

We'll use [Ray](https://docs.ray.io/) to load our dataset and apply preprocessing to batches of our data at scale.


```python
import re
```


```python
# Load as a Ray Dataset
train_ds = ray.data.from_items(train_set)
train_ds.take(1)
```

    2024-06-10 16:24:06,312	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2024-06-10_12-57-23_539567_2694/logs/ray-data
    2024-06-10 16:24:06,313	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> LimitOperator[limit=1]





    [{'gem_id': 'viggo-train-0',
      'meaning_representation': 'inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+ (for Everyone 10 and Older)], genres[driving/racing, sport], platforms[PlayStation, Xbox, PC], available_on_steam[no], has_linux_release[no], has_mac_release[no])',
      'target': "Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac.",
      'references': ["Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac."]}]



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
        {'role': 'assistant', 'content': item['meaning_representation']}]
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
    "'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']")

```

To apply our function on our dataset at scale, we can pass it to [ray.data.Dataset.map](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map.html). Here, we can specify the function to apply to each sample in our data, what compute to use, etc. The diagram below shows how we can read from various data sources (ex. cloud storage) and apply operations at scale across different hardware (CPU, GPU). For our workload, we'll just use the default compute strategy which will use CPUs to scale out our workload.

**Note**: If we want to distribute a workload across `batches` of our data instead of individual samples, we can use [ray.data.Dataset.map_batches](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html). We'll see this in action when we perform batch inference in our evaluation template. There are also many other [distributed operations](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.html) we can perform on our dataset.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/data-detailed.png" width=800>



```python
# Distributed preprocessing
ft_train_ds = train_ds.map(to_schema, fn_kwargs={'system_content': system_content})
ft_train_ds.take(1)
```

    2024-06-10 16:24:45,381	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2024-06-10_12-57-23_539567_2694/logs/ray-data
    2024-06-10 16:24:45,382	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[Map(to_schema)] -> LimitOperator[limit=1]





    [{'messages': [{'content': "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']",
        'role': 'system'},
       {'content': "Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac.",
        'role': 'user'},
       {'content': 'inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+ (for Everyone 10 and Older)], genres[driving/racing, sport], platforms[PlayStation, Xbox, PC], available_on_steam[no], has_linux_release[no], has_mac_release[no])',
        'role': 'assistant'}]}]




```python
# Repeat the steps for other splits
ft_val_ds = ray.data.from_items(val_set).map(to_schema, fn_kwargs={'system_content': system_content})
ft_test_ds = ray.data.from_items(test_set).map(to_schema, fn_kwargs={'system_content': system_content})
```

### Save and load data

We can save our data locally and/or to remote storage to use later (training, evaluation, etc.). All workspaces come with a default [cloud storage locations and shared storage](https://docs.anyscale.com/workspaces/storage) that we can write to.


```python
!pip install s3fs==0.4.2 -q
import pyarrow
from s3fs import S3FileSystem
os.environ['ANYSCALE_ARTIFACT_STORAGE']
```

    s3://anyscale-customer-dataplane-data-production-us-east-2/artifact_storage/org_2byxy1usultrdke7v3ys1cczet/cld_du8lgwhc3n26cjye7bw1ds62p7



```python
# Write to cloud storage
fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(S3FileSystem()))
storage_path = os.environ['ANYSCALE_ARTIFACT_STORAGE'][len('s3://'):]
ft_train_ds.write_json(f'{storage_path}/viggo/train.jsonl', filesystem=fs)
ft_val_ds.write_json(f'{storage_path}/viggo/val.jsonl', filesystem=fs)
ft_test_ds.write_json(f'{storage_path}/viggo/test.jsonl', filesystem=fs)
```

    2024-06-10 16:24:49,437	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2024-06-10_12-57-23_539567_2694/logs/ray-data
    2024-06-10 16:24:49,438	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[Map(to_schema)->Write]
    2024-06-10 16:24:57,190	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2024-06-10_12-57-23_539567_2694/logs/ray-data
    2024-06-10 16:24:57,191	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[Map(to_schema)->Write]
    2024-06-10 16:25:04,008	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2024-06-10_12-57-23_539567_2694/logs/ray-data
    2024-06-10 16:25:04,009	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[Map(to_schema)->Write]



```python
# Load from cloud storage
ft_train_ds = ray.data.read_json(f"{os.environ['ANYSCALE_ARTIFACT_STORAGE']}/viggo/train.jsonl")
ft_train_ds.take(1)
```

    2024-06-10 16:25:27,747	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2024-06-10_12-57-23_539567_2694/logs/ray-data
    2024-06-10 16:25:27,748	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadJSON] -> LimitOperator[limit=1]





    [{'messages': [{'content': "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']",
        'role': 'system'},
       {'content': "Dirt: Showdown from 2012 is a sport racing game for the PlayStation, Xbox, PC rated E 10+ (for Everyone 10 and Older). It's not available on Steam, Linux, or Mac.",
        'role': 'user'},
       {'content': 'inform(name[Dirt: Showdown], release_year[2012], esrb[E 10+ (for Everyone 10 and Older)], genres[driving/racing, sport], platforms[PlayStation, Xbox, PC], available_on_steam[no], has_linux_release[no], has_mac_release[no])',
        'role': 'assistant'}]}]



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
    dataset_size_scaling_factor: 10000
    output_dir: /mnt/local_storage
    deepspeed:
      config_path: configs/deepspeed/zero_3_offload_optim+param.json
    dataset_size_scaling_factor: 10000 # internal flag. No need to change
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


### Fine-tuning

This Workspace is still running on a small, lean head node. But based on the compute we want to use (ex. `num_devices` and `accelerator_type`) for fine-tuning, the appropriate worker nodes will automatically be initialized and execute the workload. And afterwards, they'll scale back to zero!

<b style="background-color: orange;">&nbsp;💡 INSIGHT&nbsp;</b>: With [Ray](https://docs.ray.io/) we're able to execute a large, compute intensive workload like this using smaller, more available resources (ex. using `A10`s instead of waiting for elusive `A100`s). And Anyscale's smart instance manager will automatically provision the appropriate and available compute for the workload based on what's needed.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/train-detailed.png" width=550>

While we could execute `python src/ft.py configs/training/lora/llama-3-8b.yaml` directly inside a Workspace notebook (see this [example](https://console.anyscale.com/v2/template-preview/finetuning_llms_v2)), we'll instead kick off the fine-tuning workload as an isolated job. An [Anyscale Job](https://docs.anyscale.com/jobs/get-started/) is a great way to scale and execute a specific workload. Here, we specify the command that needs to run (ex. `python [COMMAND][ARGS]`) along with the requirements (ex. docker image, additional, pip packages, etc.).

**Note**: Executing an Anyscale Job within a Workspace will ensure that files in the current working directory are available for the Job (unless excluded with `--exclude`). But we can also load files from anywhere (ex. Github repo, S3, etc.) if we want to launch a Job from anywhere.


```python
# View job yaml config
!cat deploy/jobs/ft.yaml
```

    name: llm-fine-tuning-guide
    entrypoint: python src/ft.py configs/training/lora/llama-3-8b.yaml
    image_uri: localhost:5555/anyscale/llm-forge:0.4.3.2
    requirements: []
    max_retries: 0
    


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
# Job submission
!anyscale job submit --config-file deploy/jobs/ft.yaml --exclude assets
```

    [1m[36mOutput[0m[0m
    [0m[1m[36m(anyscale +0.8s)[0m [0m[0m[0m[0mSubmitting job with config JobConfig(name='llm-fine-tuning-guide', image_uri='localhost:5555/anyscale/llm-forge:0.4.3.2', compute_config=None, env_vars=None, py_modules=None).[0m
    [0m[1m[36m(anyscale +3.2s)[0m [0m[0m[0m[0mUploading local dir '.' to cloud storage.[0m
    [0m[1m[36m(anyscale +4.8s)[0m [0m[0m[0m[0mJob 'llm-fine-tuning-guide' submitted, ID: 'prodjob_515se1nqf8ski7scytd52vx65e'.[0m
    [0m[1m[36m(anyscale +4.8s)[0m [0m[0m[0m[0mView the job in the UI: https://console.anyscale.com/jobs/prodjob_515se1nqf8ski7scytd52vx65e[0m
    [0m[1m[36m(anyscale +4.8s)[0m [0m[0m[0m[0mUse `--wait` to wait for the job to run and stream logs.[0m
    [0m[0m

This workload (we set to five epochs) will take ~45 min. to complete. As the job runs, you can monitor logs, metrics, Ray dashboard, etc. by clicking on the generated Job link above (`https://console.anyscale.com/jobs/prodjob_...`)

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/gpu-util.png" width=800>

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/tensorboard.png" width=800>

### Load artifacts

From the very end of the logs, we can also see where our model artifacts are stored. For example: 

```
Successfully copied files to to bucket: anyscale-customer-dataplane-data-production-us-east-2 and path: artifact_storage/org_2byxy1usultrdke7v3ys1cczet/cld_du8lgwhc3n26cjye7bw1ds62p7/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:goku_:ueewk
```

We'll load these artifacts from cloud storage to a local [cluster storage](https://docs.anyscale.com/workspaces/storage/#cluster-storage) to use for other workloads.


```python
from src.utils import download_files_from_bucket
```

<b style="background-color: yellow;">&nbsp;🔄 REPLACE&nbsp;</b>: Update the information below for the specific model and artifacts path for our fine-tuned model (retrieved from the logs from the Anyscale Job we launched above).


```python
# Locations
artifacts_dir = '/mnt/cluster_storage'  # storage accessible by head and worker nodes
model = 'meta-llama/Meta-Llama-3-8B-Instruct'
uuid = 'goku_:ueewk'  # REPLACE with your NAME + MODEL ID (from Job logs)
artifacts_path = (
    f"{os.environ['ANYSCALE_ARTIFACT_STORAGE'].split(os.environ['ANYSCALE_CLOUD_STORAGE_BUCKET'])[-1][1:]}"
    f"/lora_fine_tuning/{model}:{uuid}")
```


```python
# Download artifacts
download_files_from_bucket(
    bucket=os.environ['ANYSCALE_CLOUD_STORAGE_BUCKET'],
    path=artifacts_path,
    local_dir=artifacts_dir)

```

    Downloaded org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/README.md to /mnt/cluster_storage/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/README.md
    Downloaded org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/adapter_config.json to /mnt/cluster_storage/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/adapter_config.json
    Downloaded org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/adapter_model.safetensors to /mnt/cluster_storage/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/adapter_model.safetensors
    Downloaded org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/config.json to /mnt/cluster_storage/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/config.json
    Downloaded org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/new_embeddings.safetensors to /mnt/cluster_storage/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/new_embeddings.safetensors
    Downloaded org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/special_tokens_map.json to /mnt/cluster_storage/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/special_tokens_map.json
    Downloaded org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/tokenizer.json to /mnt/cluster_storage/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/tokenizer.json
    Downloaded org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/tokenizer_config.json to /mnt/cluster_storage/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning/meta-llama/Meta-Llama-3-8B-Instruct:gokum:atyhk/tokenizer_config.json


## Evaluation

Now we'll evaluate our fine-tuned LLM to see how well it performs on our task. There are a lot of different ways to perform evaluation. For our task, we can use traditional metrics (ex. accuracy, precision, recall, etc.) since we know what the outputs should be (extracted intent and entities).

However for many generative tasks, the outputs are very unstructured and highly subjective. For these scenarios, we can use [distance/entropy](https://github.com/huggingface/evaluate) based metrics like cosine, bleu, perplexity, etc. But, these metrics are often not very representative of the underlying task. A common strategy here is to use a larger LLM to [judge the quality](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1#evaluation) of the generated outputs. We can ask the larger LLM to directly assess the quality of the response (ex. rate between `1-5`) with a set of rules or compare it to a golden / preferred output and rate it against that.

We'll start by performing offline batch inference where we will use our tuned model to generate the outputs.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/offline-overview.png" width=500>

### Load test data


```python
# Load test set for eval
ft_test_ds = ray.data.read_json(f"{os.environ['ANYSCALE_ARTIFACT_STORAGE']}/viggo/test.jsonl")
test_data = ft_test_ds.take_all()
test_data[0]

```

    2024-06-10 16:26:57,300	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2024-06-10_12-57-23_539567_2694/logs/ray-data
    2024-06-10 16:26:57,301	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[ReadJSON]





    {'messages': [{'content': "Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']",
       'role': 'system'},
      {'content': 'Have you ever given any games on PC but not on Steam a try, like The Sims?',
       'role': 'user'},
      {'content': 'suggest(name[The Sims], platforms[PC], available_on_steam[no])',
       'role': 'assistant'}]}




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

    2024-06-10 16:27:20,604	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2024-06-10_12-57-23_539567_2694/logs/ray-data
    2024-06-10 16:27:20,604	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> LimitOperator[limit=1]


    [{'inputs': "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nGiven a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHave you ever given any games on PC but not on Steam a try, like The Sims?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 'outputs': [{'content': 'suggest(name[The Sims], platforms[PC], available_on_steam[no])', 'role': 'assistant'}]}]


### Batch inference

We will use [vLLM](https://github.com/vllm-project/vllm)'s offline LLM class to load the model and use it for inference. We can easily load our LoRA weights and merge them with the base model (just pass in `lora_path`). And we'll wrap all of this functionality in a class that we can pass to [ray.data.Dataset.map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html) to apply batch inference at scale.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/offline-detailed.png" width=750>


```python
from vllm import LLM, SamplingParams
from vllm.anyscale.lora.utils import LoRARequest
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
        'lora_path': os.path.join(artifacts_dir, artifacts_path)},
    accelerator_type='A10G',  # A10G or L4
)
```


```python
# Batch inference will take ~4 minutes
ft_pred = ft_pred_ds.take_all()
ft_pred[3]
```

    2024-06-10 16:27:26,007	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2024-06-10_12-57-23_539567_2694/logs/ray-data
    2024-06-10 16:27:26,008	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> ActorPoolMapOperator[MapBatches(LLMPredictor)]


    [36m(autoscaler +6m32s)[0m Tip: use `ray status` to view detailed cluster status. To disable these messages, set RAY_SCHEDULER_EVENTS=0.
    [36m(autoscaler +6m32s)[0m [autoscaler] [4xA10G:48CPU-192GB] Upscaling 1 node(s).
    [36m(autoscaler +6m33s)[0m [autoscaler] [4xA10G:48CPU-192GB|g5.12xlarge] [us-west-2a] [on-demand] Launched 1 instances.
    [36m(autoscaler +7m46s)[0m [autoscaler] Cluster upscaled to {56 CPU, 4 GPU}.


    [36m(_MapWorker pid=2862, ip=10.0.21.202)[0m /home/ray/anaconda3/lib/python3.9/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
    [36m(_MapWorker pid=2862, ip=10.0.21.202)[0m   warnings.warn(
    [36m(_MapWorker pid=2862, ip=10.0.21.202)[0m Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s].0.21.202)[0m 
    [36m(_MapWorker pid=2860, ip=10.0.21.202)[0m /home/ray/anaconda3/lib/python3.9/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.[32m [repeated 3x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
    [36m(_MapWorker pid=2860, ip=10.0.21.202)[0m   warnings.warn([32m [repeated 3x across cluster][0m
    [36m(MapWorker(MapBatches(LLMPredictor)) pid=2862, ip=10.0.21.202)[0m Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.[32m [repeated 4x across cluster][0m
    Processed prompts:  10%|█         | 1/10 [00:01<00:17,  2.00s/it]2)[0m 
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.09it/s])[0m 
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.12s/it]02)[0m 
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s][32m [repeated 11x across cluster][0m
    [36m(MapWorker(MapBatches(LLMPredictor)) pid=2860, ip=10.0.21.202)[0m Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.[32m [repeated 3x across cluster][0m
    Processed prompts:  30%|███       | 3/10 [00:01<00:03,  1.82it/s][32m [repeated 19x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.17it/s][32m [repeated 4x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.91s/it][32m [repeated 8x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 9x across cluster][0m
    Processed prompts:  10%|█         | 1/10 [00:00<00:08,  1.01it/s][32m [repeated 14x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.61it/s][32m [repeated 4x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.57it/s][32m [repeated 7x across cluster][0m
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s][32m [repeated 12x across cluster][0m
    Processed prompts:  40%|████      | 4/10 [00:01<00:02,  2.84it/s][32m [repeated 16x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.29it/s][32m [repeated 3x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.22s/it][32m [repeated 7x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 9x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.34it/s])[0m 
    Processed prompts:  60%|██████    | 6/10 [00:01<00:00,  5.80it/s][32m [repeated 14x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.89it/s][32m [repeated 3x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.39s/it][32m [repeated 8x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 12x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.77it/s])[0m 
    Processed prompts:  40%|████      | 4/10 [00:02<00:02,  2.03it/s][32m [repeated 13x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.48it/s][32m [repeated 2x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.45s/it][32m [repeated 5x across cluster][0m
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s][32m [repeated 12x across cluster][0m
    Processed prompts:  40%|████      | 4/10 [00:01<00:02,  2.46it/s][32m [repeated 17x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.90it/s][32m [repeated 6x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.84it/s][32m [repeated 9x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 12x across cluster][0m
    Processed prompts:  60%|██████    | 6/10 [00:01<00:00,  4.87it/s][32m [repeated 21x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.24it/s][32m [repeated 4x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.89it/s][32m [repeated 7x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.20it/s])[0m 
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 12x across cluster][0m
    Processed prompts:  50%|█████     | 5/10 [00:02<00:01,  3.07it/s][32m [repeated 16x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.25it/s][32m [repeated 4x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.20it/s][32m [repeated 8x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 12x across cluster][0m
    Processed prompts:  80%|████████  | 8/10 [00:02<00:00,  5.75it/s][32m [repeated 15x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.06it/s][32m [repeated 5x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.01it/s][32m [repeated 6x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 11x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.08it/s])[0m 
    Processed prompts:  50%|█████     | 5/10 [00:02<00:01,  2.95it/s][32m [repeated 15x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  6.06it/s][32m [repeated 4x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.47it/s][32m [repeated 5x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 14x across cluster][0m
    Processed prompts:  70%|███████   | 7/10 [00:02<00:00,  3.33it/s][32m [repeated 13x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:03<00:00,  3.33it/s][32m [repeated 5x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.05it/s][32m [repeated 9x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 12x across cluster][0m
    Processed prompts:  60%|██████    | 6/10 [00:02<00:01,  2.61it/s][32m [repeated 14x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.84it/s][32m [repeated 3x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.86it/s][32m [repeated 8x across cluster][0m
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s][32m [repeated 10x across cluster][0m
    Processed prompts:  10%|█         | 1/10 [00:01<00:12,  1.34s/it][32m [repeated 11x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:03<00:00,  3.33it/s][32m [repeated 3x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.33s/it][32m [repeated 5x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 10x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.54it/s])[0m 
    Processed prompts:  80%|████████  | 8/10 [00:02<00:00,  3.47it/s][32m [repeated 13x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.65it/s][32m [repeated 6x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.06s/it][32m [repeated 7x across cluster][0m
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s][32m [repeated 12x across cluster][0m
    Processed prompts:  40%|████      | 4/10 [00:01<00:01,  3.37it/s][32m [repeated 21x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.34it/s][32m [repeated 3x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.05it/s][32m [repeated 6x across cluster][0m
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s][32m [repeated 11x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.15it/s])[0m 
    Processed prompts:  10%|█         | 1/10 [00:01<00:09,  1.09s/it][32m [repeated 14x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.52it/s][32m [repeated 6x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.33s/it][32m [repeated 6x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 10x across cluster][0m
    Processed prompts:  10%|█         | 1/10 [00:01<00:12,  1.41s/it][32m [repeated 14x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.06it/s][32m [repeated 5x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.26s/it][32m [repeated 6x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 13x across cluster][0m
    Processed prompts:  10%|█         | 1/10 [00:01<00:15,  1.74s/it][32m [repeated 17x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.13it/s][32m [repeated 5x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.54s/it][32m [repeated 7x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.29it/s])[0m 
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 11x across cluster][0m
    Processed prompts:  40%|████      | 4/10 [00:01<00:01,  3.50it/s][32m [repeated 16x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  6.05it/s][32m [repeated 6x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.38s/it][32m [repeated 7x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.49it/s])[0m 
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 13x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.60it/s])[0m 
    Processed prompts:  10%|█         | 1/10 [00:01<00:15,  1.72s/it][32m [repeated 13x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.36it/s][32m [repeated 3x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.57s/it][32m [repeated 6x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 10x across cluster][0m
    Processed prompts:  80%|████████  | 8/10 [00:02<00:00,  5.73it/s][32m [repeated 18x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.84it/s][32m [repeated 4x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.73s/it][32m [repeated 7x across cluster][0m
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s][32m [repeated 10x across cluster][0m
    Processed prompts:  40%|████      | 4/10 [00:01<00:01,  3.15it/s][32m [repeated 11x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.40it/s][32m [repeated 3x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.64s/it][32m [repeated 7x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 11x across cluster][0m
    Processed prompts:  50%|█████     | 5/10 [00:01<00:01,  4.49it/s][32m [repeated 15x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.60it/s][32m [repeated 5x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.45s/it][32m [repeated 5x across cluster][0m
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s][32m [repeated 11x across cluster][0m
    Processed prompts:  10%|█         | 1/10 [00:02<00:21,  2.37s/it][32m [repeated 9x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:03<00:00,  3.30it/s][32m [repeated 5x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.03s/it][32m [repeated 5x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:03<00:00,  3.26it/s])[0m 
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s][32m [repeated 10x across cluster][0m
    Processed prompts:  70%|███████   | 7/10 [00:02<00:00,  3.28it/s][32m [repeated 16x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.78it/s][32m [repeated 3x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:02<00:00,  2.08s/it][32m [repeated 6x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.77it/s])[0m 
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s][32m [repeated 10x across cluster][0m
    Processed prompts:  40%|████      | 4/10 [00:01<00:01,  3.90it/s][32m [repeated 15x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.63it/s][32m [repeated 4x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.16it/s][32m [repeated 7x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 11x across cluster][0m
    Processed prompts:  70%|███████   | 7/10 [00:01<00:00,  4.89it/s][32m [repeated 15x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.29it/s][32m [repeated 6x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.03s/it][32m [repeated 6x across cluster][0m
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s][32m [repeated 12x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.33it/s])[0m 
    Processed prompts:  40%|████      | 4/10 [00:01<00:02,  2.84it/s][32m [repeated 9x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.92it/s][32m [repeated 2x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.26s/it][32m [repeated 6x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 10x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.61it/s])[0m 
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.23it/s])[0m 
    Processed prompts:  70%|███████   | 7/10 [00:01<00:00,  5.13it/s][32m [repeated 21x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.98it/s][32m [repeated 5x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.03s/it][32m [repeated 6x across cluster][0m
    Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s][32m [repeated 11x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:01<00:00,  5.23it/s])[0m 
    Processed prompts:  10%|█         | 1/10 [00:01<00:13,  1.45s/it][32m [repeated 21x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.48it/s][32m [repeated 6x across cluster][0m
    Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.29s/it][32m [repeated 3x across cluster][0m
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s][32m [repeated 10x across cluster][0m
    Processed prompts:  10%|█         | 1/10 [00:01<00:09,  1.03s/it][32m [repeated 27x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.74it/s][32m [repeated 8x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.83it/s][32m [repeated 2x across cluster][0m
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s][32m [repeated 10x across cluster][0m
    Processed prompts:  80%|████████  | 8/10 [00:02<00:00,  4.66it/s][32m [repeated 27x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:03<00:00,  3.28it/s][32m [repeated 6x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.40it/s])[0m 
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.13it/s])[0m 
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  3.55it/s])[0m 
    Processed prompts:   0%|          | 0/10 [00:00<?, ?it/s][32m [repeated 8x across cluster][0m
    Processed prompts: 100%|██████████| 10/10 [00:02<00:00,  4.42it/s])[0m 





    {'prompt': "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nGiven a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nI like first person games normally, but not even that could make a music game fun for me. In fact in Guitar Hero: Smash Hits, I think the perspective somehow made an already bad game even worse.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
     'expected_output': array([{'content': 'give_opinion(name[Guitar Hero: Smash Hits], rating[poor], genres[music], player_perspective[first person])', 'role': 'assistant'}],
           dtype=object),
     'generated_text': 'give_opinion(name[Guitar Hero: Smash Hits], rating[poor], genres[music], player_perspective[first person])<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'}



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




    [{'prompt': "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nGiven a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nDance Dance Revolution Universe 3 got poor ratings when it came out in 2008. It might have been the worst multiplayer music game for the Xbox.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
      'expected_output': array([{'content': 'inform(name[Dance Dance Revolution Universe 3], release_year[2008], rating[poor], genres[music], has_multiplayer[yes], platforms[Xbox])', 'role': 'assistant'}],
            dtype=object),
      'generated_text': 'give_opinion(name[Dance Dance Revolution Universe 3], release_year[2008], rating[poor], has_multiplayer[yes], platforms[Xbox])<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'},
     {'prompt': "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nGiven a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values. This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute']. The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nA first person game I recently got on Steam is Assetto Corsa. Have you heard of it?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
      'expected_output': array([{'content': 'recommend(name[Assetto Corsa], player_perspective[first person], available_on_steam[yes])', 'role': 'assistant'}],
            dtype=object),
      'generated_text': 'recommend(name[Assetto Corsa], available_on_steam[yes])<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'}]



## Serving

For model serving, we'll first serve it locally, test it and then launch a production grade service that can autoscale to meet any demand.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/online-overview.png" width=500>

We'll start by generating the configuration for our service. We provide a convenient CLI experience to generate this configuration but you can create one from scratch as well. Here we can specify where our model lives, autoscaling behavior, accelerators to use, lora adapters, etc.

<b style="background-color: orange;">&nbsp;💡 INSIGHT&nbsp;</b>: Ray Serve and Anyscale support [serving multiple LoRA adapters](https://github.com/anyscale/templates/blob/main/templates/endpoints_v2/examples/lora/DeployLora.ipynb) with a common base model in the same request batch which allows you to serve a wide variety of use-cases without increasing hardware spend. In addition, we use Serve multiplexing to reduce the number of swaps for LoRA adapters. There is a slight latency overhead to serving a LoRA model compared to the base model, typically 10-20%.

**LoRA weights storage URI**: `s3://anyscale-customer-dataplane-data-production-us-east-2 and path: artifact_storage/org_2byxy1usultrdke7v3ys1cczet/cld_du8lgwhc3n26cjye7bw1ds62p7/lora_fine_tuning`

**model**: `meta-llama/Meta-Llama-3-8B-Instruct:goku_:ueewk`


We'll start by running the python command below to start the CLI workflow to generate the service yaml configuration:
```bash
mkdir /home/ray/default/deploy/services
cd /home/ray/default/deploy/services
python /home/ray/default/src/generate_serve_config.py 
```

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-llm-workflows/assets/cli.png" width=500>

<b style="background-color: yellow;">&nbsp;🔄 REPLACE&nbsp;</b>: Use the serve configuration generated for you.


```python
# Generated service configuration
!cat /home/ray/default/deploy/services/serve_{TIMESTAMP}.yaml
```

    applications:
    - args:
        dynamic_lora_loading_path: s3://anyscale-test-data-cld-i2w99rzq8b6lbjkke9y94vi5/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/lora_fine_tuning
        embedding_models: []
        function_calling_models: []
        models: []
        multiplex_lora_adapters: []
        multiplex_models:
        - ./model_config/model_config_20240516095237.yaml
        vllm_base_models: []
      import_path: aviary_private_endpoints.backend.server.run:router_application
      name: llm-endpoint
      route_prefix: /


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
        model=f'{model}:{uuid}',  # with your unique model ID
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
print (response.split('<|eot_id|>')[0])
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

- [Batch text embeddings with Ray data](https://github.com/anyscale/templates/tree/main/templates/text-embeddings)
- [Continued fine-tuning from checkpoint](https://github.com/anyscale/templates/tree/main/templates/fine-tune-llm_v2/cookbooks/continue_from_checkpoint)
- [Serving multiple LoRA adapters with same base model](https://github.com/anyscale/templates/blob/main/templates/endpoints_v2/examples/lora/DeployLora.ipynb) (+ multiplexing)
- [Deploy models for embedding generation](https://github.com/anyscale/templates/blob/main/templates/endpoints_v2/examples/embedding/EmbeddingModels.ipynb)
- Function calling [fine-tuning](https://github.com/anyscale/templates/tree/main/templates/fine-tune-llm_v2/end-to-end-examples/fine-tune-function-calling) and [deployment](https://github.com/anyscale/templates/blob/main/templates/endpoints_v2/examples/function_calling/DeployFunctionCalling.ipynb)
- [Configs to optimize the latency/throughput](https://github.com/anyscale/templates/blob/main/templates/endpoints_v2/examples/OptimizeModels.ipynb)
- [Configs to control optimization parameters and tensor-parallelism](https://github.com/anyscale/templates/blob/main/templates/endpoints_v2/examples/AdvancedModelConfigs.ipynb)
- Creating a [Router](https://github.com/anyscale/llm-router) between different models (base, fine-tuned, closed-source) to optimize for cost and quality.
- Stable diffusion [fine-tuning](https://github.com/anyscale/templates/tree/main/templates/fine-tune-stable-diffusion) and [serving](https://github.com/anyscale/templates/tree/main/templates/serve-stable-diffusion)

And if you're interested in using our hosted Anyscale or connecting it to your own cloud, reach out to us at [Anyscale](https://www.anyscale.com/get-started?utm_source=goku). And follow us on [Twitter](https://x.com/anyscalecompute) and [LinkedIn](https://www.linkedin.com/company/joinanyscale/) for more real-time updates on new features!
