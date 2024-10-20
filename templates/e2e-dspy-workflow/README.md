# End-to-end DSPy Workflows Guide 

Time to complete: 1 hour

## Building an Efficient LLM Pipeline with DSPy and Anyscale

## Problem Statement
You are a bank looking to categorize customer support queries into 25 categories. With only 100 hand-labeled examples and 4,000 unlabeled examples, traditional classifiers aren't viable. While Large Language Models (LLMs) could solve this, you need a cost-effective solution that doesn't compromise on accuracy.

## Why DSPy and Anyscale?
DSPy simplifies the complex workflow of:
- Data Collection/Labeling
- Fine-tuning
- Prompt Optimization
- Evaluation
- Deployment

The solution leverages DSPy on Anyscale to distill knowledge from a 70B model into a more cost-effective 1B model, making it practical for production deployment.

## Implementation Roadmap

### 1. Setup
- Install DSPy
- Configure environment
- Load dataset
- Set up program, metric, and evaluator

### 2. Data Processing and Labeling
- Process 4,000 unlabeled customer queries
- Use a 70B oracle model locally to generate labels
- Incorporate Chains of Thought to capture reasoning patterns
- Note: The 100 hand-labeled examples will be used in future iterations

### 3. Model Fine-tuning
- Use DSPy's fine-tuning tools to optimize a 1B model
- Leverage Anyscale's LLMForge backend
- Estimated runtime: 20 minutes on 4xA100-80GB GPUs

### 4. Evaluation and Optimization
- Evaluate fine-tuned 1B model checkpoints against labeled dataset
- Perform prompt optimization
- Compare best checkpoint against un-finetuned 1B baseline
- Generate comprehensive evaluation metrics

### 5. Production Deployment
- Deploy optimized 1B model using Anyscale's RayLLM

## Future Improvements
- Optimize batch inference with DSPy pipeline
- Explore alternative fine-tuning approaches
- Conduct hyperparameter optimization
- Integrate the hand-labeled dataset for validation

## Technical Details
The implementation follows these key processes:
1. Knowledge distillation from 70B to 1B model
2. Chain of Thought prompting for better reasoning
3. Efficient model serving with RayLLM
4. Continuous evaluation and optimization

## Implementation Flow
```
Raw Data → Oracle Labeling → Fine-tuning → Optimization → Deployment
    ↑                            ↓             ↓              ↓
    └── Validation Dataset ←── Evaluation ── Metrics ── Production Serving
```

## Set up

Node Set up:

We will be running everything on a head node that uses 4xA100-80GB GPUs. I find that L4s are usually available and suitable for this usecase. You can also use any more powerful node.

To change to use A100 GPUs, click the "1 active node" in the top right corner, then for workspace node, click the pencil icon and navigate to the A100 tab and select the 4xA100 option. If you do not see A100 in the list of GPUs, they may not be available on your cloud.


```python
%load_ext autoreload
%autoreload 2
```


```python
import importlib.util

if importlib.util.find_spec("dspy") is None:
    print("Installing dspy")
    !git clone https://github.com/stanfordnlp/dspy.git dspy-ai
    !pip install ./dspy-ai
else:
    print("dspy is already installed")

!pip install matplotlib
```

    Installing dspy
    Cloning into 'dspy-ai'...
    remote: Enumerating objects: 38344, done.[K
    remote: Counting objects: 100% (1777/1777), done.[K
    remote: Compressing objects: 100% (690/690), done.[K
    remote: Total 38344 (delta 1168), reused 1598 (delta 1067), pack-reused 36567 (from 1)[K
    Receiving objects: 100% (38344/38344), 35.30 MiB | 31.60 MiB/s, done.
    Resolving deltas: 100% (19370/19370), done.
    Processing ./dspy-ai
      Installing build dependencies ... [?25ldone
    [?25h  Getting requirements to build wheel ... [?25ldone
    [?25h  Preparing metadata (pyproject.toml) ... [?25ldone
    [?25hCollecting backoff~=2.2 (from dspy==2.5.12)
      Downloading backoff-2.2.1-py3-none-any.whl.metadata (14 kB)
    Requirement already satisfied: joblib~=1.3 in /home/ray/anaconda3/lib/python3.9/site-packages (from dspy==2.5.12) (1.4.2)
    Requirement already satisfied: openai in /home/ray/anaconda3/lib/python3.9/site-packages (from dspy==2.5.12) (1.45.0)
    Requirement already satisfied: pandas in /home/ray/anaconda3/lib/python3.9/site-packages (from dspy==2.5.12) (1.5.3)
    Requirement already satisfied: regex in /home/ray/anaconda3/lib/python3.9/site-packages (from dspy==2.5.12) (2024.9.11)
    Collecting ujson (from dspy==2.5.12)
      Downloading ujson-5.10.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.3 kB)
    Requirement already satisfied: tqdm in /home/ray/anaconda3/lib/python3.9/site-packages (from dspy==2.5.12) (4.65.0)
    Requirement already satisfied: datasets<3.0.0,>=2.14.6 in /home/ray/anaconda3/lib/python3.9/site-packages (from dspy==2.5.12) (2.19.2)
    Requirement already satisfied: requests in /home/ray/anaconda3/lib/python3.9/site-packages (from dspy==2.5.12) (2.32.3)
    Collecting optuna (from dspy==2.5.12)
      Downloading optuna-4.0.0-py3-none-any.whl.metadata (16 kB)
    Requirement already satisfied: pydantic~=2.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from dspy==2.5.12) (2.9.1)
    Requirement already satisfied: structlog in /home/ray/anaconda3/lib/python3.9/site-packages (from dspy==2.5.12) (24.4.0)
    Requirement already satisfied: jinja2 in /home/ray/anaconda3/lib/python3.9/site-packages (from dspy==2.5.12) (3.1.2)
    Collecting magicattr~=0.1.6 (from dspy==2.5.12)
      Downloading magicattr-0.1.6-py2.py3-none-any.whl.metadata (3.2 kB)
    Collecting litellm (from dspy==2.5.12)
      Downloading litellm-1.50.0-py3-none-any.whl.metadata (32 kB)
    Requirement already satisfied: diskcache in /home/ray/anaconda3/lib/python3.9/site-packages (from dspy==2.5.12) (5.6.3)
    Requirement already satisfied: filelock in /home/ray/anaconda3/lib/python3.9/site-packages (from datasets<3.0.0,>=2.14.6->dspy==2.5.12) (3.13.1)
    Requirement already satisfied: numpy>=1.17 in /home/ray/anaconda3/lib/python3.9/site-packages (from datasets<3.0.0,>=2.14.6->dspy==2.5.12) (1.23.5)
    Requirement already satisfied: pyarrow>=12.0.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from datasets<3.0.0,>=2.14.6->dspy==2.5.12) (12.0.1)
    Requirement already satisfied: pyarrow-hotfix in /home/ray/anaconda3/lib/python3.9/site-packages (from datasets<3.0.0,>=2.14.6->dspy==2.5.12) (0.6)
    Requirement already satisfied: dill<0.3.9,>=0.3.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from datasets<3.0.0,>=2.14.6->dspy==2.5.12) (0.3.8)
    Requirement already satisfied: xxhash in /home/ray/anaconda3/lib/python3.9/site-packages (from datasets<3.0.0,>=2.14.6->dspy==2.5.12) (3.5.0)
    Requirement already satisfied: multiprocess in /home/ray/anaconda3/lib/python3.9/site-packages (from datasets<3.0.0,>=2.14.6->dspy==2.5.12) (0.70.16)
    Requirement already satisfied: fsspec<=2024.3.1,>=2023.1.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from fsspec[http]<=2024.3.1,>=2023.1.0->datasets<3.0.0,>=2.14.6->dspy==2.5.12) (2023.5.0)
    Requirement already satisfied: aiohttp in /home/ray/anaconda3/lib/python3.9/site-packages (from datasets<3.0.0,>=2.14.6->dspy==2.5.12) (3.9.5)
    Requirement already satisfied: huggingface-hub>=0.21.2 in /home/ray/anaconda3/lib/python3.9/site-packages (from datasets<3.0.0,>=2.14.6->dspy==2.5.12) (0.24.7)
    Requirement already satisfied: packaging in /home/ray/anaconda3/lib/python3.9/site-packages (from datasets<3.0.0,>=2.14.6->dspy==2.5.12) (23.0)
    Requirement already satisfied: pyyaml>=5.1 in /home/ray/anaconda3/lib/python3.9/site-packages (from datasets<3.0.0,>=2.14.6->dspy==2.5.12) (6.0.1)
    Requirement already satisfied: annotated-types>=0.6.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from pydantic~=2.0->dspy==2.5.12) (0.6.0)
    Requirement already satisfied: pydantic-core==2.23.3 in /home/ray/anaconda3/lib/python3.9/site-packages (from pydantic~=2.0->dspy==2.5.12) (2.23.3)
    Requirement already satisfied: typing-extensions>=4.6.1 in /home/ray/anaconda3/lib/python3.9/site-packages (from pydantic~=2.0->dspy==2.5.12) (4.12.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /home/ray/anaconda3/lib/python3.9/site-packages (from requests->dspy==2.5.12) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /home/ray/anaconda3/lib/python3.9/site-packages (from requests->dspy==2.5.12) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /home/ray/anaconda3/lib/python3.9/site-packages (from requests->dspy==2.5.12) (1.26.19)
    Requirement already satisfied: certifi>=2017.4.17 in /home/ray/anaconda3/lib/python3.9/site-packages (from requests->dspy==2.5.12) (2023.11.17)
    Requirement already satisfied: MarkupSafe>=2.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from jinja2->dspy==2.5.12) (2.1.3)
    Requirement already satisfied: click in /home/ray/anaconda3/lib/python3.9/site-packages (from litellm->dspy==2.5.12) (8.1.7)
    Requirement already satisfied: importlib-metadata>=6.8.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from litellm->dspy==2.5.12) (6.11.0)
    Collecting jsonschema<5.0.0,>=4.22.0 (from litellm->dspy==2.5.12)
      Downloading jsonschema-4.23.0-py3-none-any.whl.metadata (7.9 kB)
    Collecting openai (from dspy==2.5.12)
      Downloading openai-1.52.0-py3-none-any.whl.metadata (24 kB)
    Requirement already satisfied: python-dotenv>=0.2.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from litellm->dspy==2.5.12) (1.0.1)
    Requirement already satisfied: tiktoken>=0.7.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from litellm->dspy==2.5.12) (0.7.0)
    Requirement already satisfied: tokenizers in /home/ray/anaconda3/lib/python3.9/site-packages (from litellm->dspy==2.5.12) (0.19.1)
    Requirement already satisfied: anyio<5,>=3.5.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from openai->dspy==2.5.12) (3.7.1)
    Requirement already satisfied: distro<2,>=1.7.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from openai->dspy==2.5.12) (1.8.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from openai->dspy==2.5.12) (0.27.2)
    Requirement already satisfied: jiter<1,>=0.4.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from openai->dspy==2.5.12) (0.5.0)
    Requirement already satisfied: sniffio in /home/ray/anaconda3/lib/python3.9/site-packages (from openai->dspy==2.5.12) (1.3.1)
    Collecting alembic>=1.5.0 (from optuna->dspy==2.5.12)
      Downloading alembic-1.13.3-py3-none-any.whl.metadata (7.4 kB)
    Collecting colorlog (from optuna->dspy==2.5.12)
      Downloading colorlog-6.8.2-py3-none-any.whl.metadata (10 kB)
    Collecting sqlalchemy>=1.3.0 (from optuna->dspy==2.5.12)
      Downloading SQLAlchemy-2.0.36-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (9.7 kB)
    Requirement already satisfied: python-dateutil>=2.8.1 in /home/ray/anaconda3/lib/python3.9/site-packages (from pandas->dspy==2.5.12) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /home/ray/anaconda3/lib/python3.9/site-packages (from pandas->dspy==2.5.12) (2022.7.1)
    Collecting Mako (from alembic>=1.5.0->optuna->dspy==2.5.12)
      Downloading Mako-1.3.5-py3-none-any.whl.metadata (2.9 kB)
    Requirement already satisfied: exceptiongroup in /home/ray/anaconda3/lib/python3.9/site-packages (from anyio<5,>=3.5.0->openai->dspy==2.5.12) (1.2.2)
    Requirement already satisfied: aiosignal>=1.1.2 in /home/ray/anaconda3/lib/python3.9/site-packages (from aiohttp->datasets<3.0.0,>=2.14.6->dspy==2.5.12) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from aiohttp->datasets<3.0.0,>=2.14.6->dspy==2.5.12) (24.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /home/ray/anaconda3/lib/python3.9/site-packages (from aiohttp->datasets<3.0.0,>=2.14.6->dspy==2.5.12) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /home/ray/anaconda3/lib/python3.9/site-packages (from aiohttp->datasets<3.0.0,>=2.14.6->dspy==2.5.12) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from aiohttp->datasets<3.0.0,>=2.14.6->dspy==2.5.12) (1.9.4)
    Requirement already satisfied: async-timeout<5.0,>=4.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from aiohttp->datasets<3.0.0,>=2.14.6->dspy==2.5.12) (4.0.3)
    Requirement already satisfied: httpcore==1.* in /home/ray/anaconda3/lib/python3.9/site-packages (from httpx<1,>=0.23.0->openai->dspy==2.5.12) (1.0.5)
    Requirement already satisfied: h11<0.15,>=0.13 in /home/ray/anaconda3/lib/python3.9/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai->dspy==2.5.12) (0.14.0)
    Requirement already satisfied: zipp>=0.5 in /home/ray/anaconda3/lib/python3.9/site-packages (from importlib-metadata>=6.8.0->litellm->dspy==2.5.12) (3.19.2)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /home/ray/anaconda3/lib/python3.9/site-packages (from jsonschema<5.0.0,>=4.22.0->litellm->dspy==2.5.12) (2023.12.1)
    Requirement already satisfied: referencing>=0.28.4 in /home/ray/anaconda3/lib/python3.9/site-packages (from jsonschema<5.0.0,>=4.22.0->litellm->dspy==2.5.12) (0.35.1)
    Requirement already satisfied: rpds-py>=0.7.1 in /home/ray/anaconda3/lib/python3.9/site-packages (from jsonschema<5.0.0,>=4.22.0->litellm->dspy==2.5.12) (0.20.0)
    Requirement already satisfied: six>=1.5 in /home/ray/anaconda3/lib/python3.9/site-packages (from python-dateutil>=2.8.1->pandas->dspy==2.5.12) (1.16.0)
    Collecting greenlet!=0.4.17 (from sqlalchemy>=1.3.0->optuna->dspy==2.5.12)
      Downloading greenlet-3.1.1-cp39-cp39-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (3.8 kB)
    Downloading backoff-2.2.1-py3-none-any.whl (15 kB)
    Downloading magicattr-0.1.6-py2.py3-none-any.whl (4.7 kB)
    Downloading litellm-1.50.0-py3-none-any.whl (6.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.3/6.3 MB[0m [31m106.1 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading openai-1.52.0-py3-none-any.whl (386 kB)
    Downloading optuna-4.0.0-py3-none-any.whl (362 kB)
    Downloading ujson-5.10.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (53 kB)
    Downloading alembic-1.13.3-py3-none-any.whl (233 kB)
    Downloading jsonschema-4.23.0-py3-none-any.whl (88 kB)
    Downloading SQLAlchemy-2.0.36-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.1/3.1 MB[0m [31m184.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading colorlog-6.8.2-py3-none-any.whl (11 kB)
    Downloading greenlet-3.1.1-cp39-cp39-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (597 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m597.4/597.4 kB[0m [31m56.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading Mako-1.3.5-py3-none-any.whl (78 kB)
    Building wheels for collected packages: dspy
      Building wheel for dspy (pyproject.toml) ... [?25ldone
    [?25h  Created wheel for dspy: filename=dspy-2.5.12-py3-none-any.whl size=319214 sha256=26daf8ae2676163b5c6b451c1069e851350f0404af30963de3dcb4fa3dc8cddf
      Stored in directory: /mnt/local_storage/data/cache/pip/wheels/89/3d/07/ce46fde97bfc61290dff1a025ff7d8e4142c4360d71c2a0fe5
    Successfully built dspy
    Installing collected packages: magicattr, ujson, Mako, greenlet, colorlog, backoff, sqlalchemy, openai, jsonschema, alembic, optuna, litellm, dspy
      Attempting uninstall: backoff
        Found existing installation: backoff 1.10.0
        Uninstalling backoff-1.10.0:
          Successfully uninstalled backoff-1.10.0
      Attempting uninstall: openai
        Found existing installation: openai 1.45.0
        Uninstalling openai-1.45.0:
          Successfully uninstalled openai-1.45.0
      Attempting uninstall: jsonschema
        Found existing installation: jsonschema 4.21.1
        Uninstalling jsonschema-4.21.1:
          Successfully uninstalled jsonschema-4.21.1
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    rayllm-oss 0.3.1 requires jsonschema~=4.21.1, but you have jsonschema 4.23.0 which is incompatible.
    rayllm-oss 0.3.1 requires pydantic~=2.6.0, but you have pydantic 2.9.1 which is incompatible.[0m[31m
    [0mSuccessfully installed Mako-1.3.5 alembic-1.13.3 backoff-2.2.1 colorlog-6.8.2 dspy-2.5.12 greenlet-3.1.1 jsonschema-4.23.0 litellm-1.50.0 magicattr-0.1.6 openai-1.52.0 optuna-4.0.0 sqlalchemy-2.0.36 ujson-5.10.0
    
    [93m#################
    
    ANYSCALE WARNING:
    Local packages ./dspy-ai are not supported across cluster, please check our documentations for workarounds: https://docs.anyscale.com/configuration/dependency-management/dependency-development
    
    #################[0m
    
    
    Requirement already satisfied: matplotlib in /home/ray/anaconda3/lib/python3.9/site-packages (3.9.2)
    Requirement already satisfied: contourpy>=1.0.1 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (1.3.0)
    Requirement already satisfied: cycler>=0.10 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (4.54.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (1.4.7)
    Requirement already satisfied: numpy>=1.23 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (1.23.5)
    Requirement already satisfied: packaging>=20.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (23.0)
    Requirement already satisfied: pillow>=8 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (9.2.0)
    Requirement already satisfied: pyparsing>=2.3.1 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (3.1.4)
    Requirement already satisfied: python-dateutil>=2.7 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (2.8.2)
    Requirement already satisfied: importlib-resources>=3.2.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from matplotlib) (6.4.5)
    Requirement already satisfied: zipp>=3.1.0 in /home/ray/anaconda3/lib/python3.9/site-packages (from importlib-resources>=3.2.0->matplotlib) (3.19.2)
    Requirement already satisfied: six>=1.5 in /home/ray/anaconda3/lib/python3.9/site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)


In order to run this notebook, you need to have the following environment variables set:
- HF_TOKEN
- HF_HOME=/mnt/local_storage/huggingface
- (optional) WANDB_API_KEY

You can get a HF_TOKEN [here](https://huggingface.co/settings/tokens).

You can get a WANDB_API_KEY [here](https://wandb.ai/authorize).


```python
import dspy
dspy.settings.configure(experimental=True)

import ujson

# Theoretically this isnt needed if the environment variables work correctly
# from dotenv import load_dotenv
# load_dotenv()

from src import set_dspy_cache_location
set_dspy_cache_location("/home/ray/default/dspy/cache2")
```


```python
import os

os.environ["HF_TOKEN"] = "hf_12345"
os.environ["HF_HOME"] = "/mnt/local_storage/huggingface"
os.environ["WANDB_API_KEY"] = "..."
```


```python
from src import check_env_vars
check_env_vars()
```


```python
from src import init_ray
init_ray()
```

    2024-10-20 18:38:05,421	INFO worker.py:1601 -- Connecting to existing Ray cluster at address: 10.0.0.60:6379...
    2024-10-20 18:38:05,428	INFO worker.py:1777 -- Connected to Ray cluster. View the dashboard at https://session-74s48fpq311xtc218ullz9snys.i.anyscaleuserdata.com 
    2024-10-20 18:38:05,452	INFO packaging.py:531 -- Creating a file package for local directory '/home/ray/anaconda3/lib/python3.9/site-packages/dspy'.
    2024-10-20 18:38:05,483	INFO packaging.py:359 -- Pushing file package 'gcs://_ray_pkg_3507d9586ca57c32.zip' (1.16MiB) to Ray cluster...
    2024-10-20 18:38:05,494	INFO packaging.py:372 -- Successfully pushed file package 'gcs://_ray_pkg_3507d9586ca57c32.zip'.
    2024-10-20 18:38:05,505	INFO packaging.py:531 -- Creating a file package for local directory '/home/ray/anaconda3/lib/python3.9/site-packages/dsp'.
    2024-10-20 18:38:05,522	INFO packaging.py:359 -- Pushing file package 'gcs://_ray_pkg_1600dff5599964a8.zip' (0.55MiB) to Ray cluster...
    2024-10-20 18:38:05,528	INFO packaging.py:372 -- Successfully pushed file package 'gcs://_ray_pkg_1600dff5599964a8.zip'.
    2024-10-20 18:38:05,586	INFO packaging.py:359 -- Pushing file package 'gcs://_ray_pkg_b8fa96072b60d0416b79aace937ddf97e6fc802e.zip' (28.30MiB) to Ray cluster...
    2024-10-20 18:38:05,848	INFO packaging.py:372 -- Successfully pushed file package 'gcs://_ray_pkg_b8fa96072b60d0416b79aace937ddf97e6fc802e.zip'.


We will make use of a random number generator in this notebook. We are creating a Random object here to ensure that our notebook is reproducible.


```python
from src import set_random_seed
rng = set_random_seed()
```

We will be using the PolyAI/banking77 dataset for this tutorial. We use the built in dspy DataLoader to load the dataset from Huggingface as a list of dspy.Example objects.


```python
%%capture
# Prepare the dataset
from src import load_data_from_huggingface, convert_int_label_to_string
full_trainset, full_testset = load_data_from_huggingface()

full_trainset_processed, full_testset_processed = convert_int_label_to_string(full_trainset, full_testset)
print("Example training set: ", full_trainset_processed[0])
```

The dataset is originally called "banking77" because there are 77 labels. We will be reducing this to the top 25 most frequent labels.


```python
from src import filter_to_top_n_labels
full_trainset_filtered, full_testset_filtered, top_25_labels = filter_to_top_n_labels(full_trainset_processed, full_testset_processed, n=25)

print(f"Dataset filtered to top 25 labels. New sizes:")
print(f"Training set size: {len(full_trainset_filtered)}; Test set size: {len(full_testset_filtered)}")
print(f"Top 25 labels: {', '.join(str(label) for label in top_25_labels)}")
print(f"Example training set: {full_trainset_filtered[0]}")
print(f"Example test set: {full_testset_filtered[0]}")

```

    Dataset filtered to top 25 labels. New sizes:
    Training set size: 4171; Test set size: 1000
    Top 25 labels: card_payment_fee_charged, direct_debit_payment_not_recognised, balance_not_updated_after_cheque_or_cash_deposit, wrong_amount_of_cash_received, cash_withdrawal_charge, transaction_charged_twice, declined_cash_withdrawal, transfer_fee_charged, balance_not_updated_after_bank_transfer, transfer_not_received_by_recipient, request_refund, card_payment_not_recognised, card_payment_wrong_exchange_rate, extra_charge_on_statement, wrong_exchange_rate_for_cash_withdrawal, refund_not_showing_up, reverted_card_payment, cash_withdrawal_not_recognised, activate_my_card, pending_card_payment, cancel_transfer, beneficiary_not_allowed, card_arrival, declined_card_payment, pending_top_up
    Example training set: Example({'label': 'card_arrival', 'text': 'I am still waiting on my card?'}) (input_keys={'text'})
    Example test set: Example({'label': 'card_arrival', 'text': 'How do I locate my card?'}) (input_keys={'text'})



```python
labels_in_use = top_25_labels
print(labels_in_use)
```

    ['card_payment_fee_charged', 'direct_debit_payment_not_recognised', 'balance_not_updated_after_cheque_or_cash_deposit', 'wrong_amount_of_cash_received', 'cash_withdrawal_charge', 'transaction_charged_twice', 'declined_cash_withdrawal', 'transfer_fee_charged', 'balance_not_updated_after_bank_transfer', 'transfer_not_received_by_recipient', 'request_refund', 'card_payment_not_recognised', 'card_payment_wrong_exchange_rate', 'extra_charge_on_statement', 'wrong_exchange_rate_for_cash_withdrawal', 'refund_not_showing_up', 'reverted_card_payment', 'cash_withdrawal_not_recognised', 'activate_my_card', 'pending_card_payment', 'cancel_transfer', 'beneficiary_not_allowed', 'card_arrival', 'declined_card_payment', 'pending_top_up']


Now we will shuffle our training set and split it into a training and labeled set.

The scenario we are emulating is that we only have 100 labeled examples to train on. We are saying that we have 4K (length of the training set) unlabeled examples we can then label using an oracle model, and then distill the knowledge from the oracle model into our 1B model.


```python
from src import common_kwargs

shuffled_trainset = [d for d in full_trainset_filtered]
rng.shuffle(shuffled_trainset)

# The devset shouldn't overlap
ft_trainset = shuffled_trainset[:-100]
labeled_trainset = shuffled_trainset[-100:]

testset = full_testset_filtered
evaluate_testset = dspy.Evaluate(devset=testset, **common_kwargs)
```

This is a simple, 1 step Chain of Thought program.

In DSPy, you define a Signature to show your inputs and outputs. You define a module to run the different steps of your program.

We need to pass the labels to the LLM somehow.

In DSPy, we can do this by either including it in the docstring of the program or by adding it as an input field to the Signature.

Here, we will add it to the docstring, because the set of labels is fixed.

We then have an `intent` field which is the input to the program.

Finally we have a `label` field which is the output of the program.

We give both of these fields a short description.


```python
class IntentClassification(dspy.Signature):
    """As a part of a banking issue traiging system, classify the intent of a natural language query into one of the 25 labels.
    The intent should exactly match one of the following:
    ['card_payment_fee_charged', 'direct_debit_payment_not_recognised', 'balance_not_updated_after_cheque_or_cash_deposit', 'wrong_amount_of_cash_received', 'cash_withdrawal_charge', 'transaction_charged_twice', 'declined_cash_withdrawal', 'transfer_fee_charged', 'balance_not_updated_after_bank_transfer', 'transfer_not_received_by_recipient', 'request_refund', 'card_payment_not_recognised', 'card_payment_wrong_exchange_rate', 'extra_charge_on_statement', 'wrong_exchange_rate_for_cash_withdrawal', 'refund_not_showing_up', 'reverted_card_payment', 'cash_withdrawal_not_recognised', 'activate_my_card', 'pending_card_payment', 'cancel_transfer', 'beneficiary_not_allowed', 'card_arrival', 'declined_card_payment', 'pending_top_up']
    """

    intent = dspy.InputField(desc="Intent of the query")
    label = dspy.OutputField(desc="Type of the intent; Should just be one of the 25 labels with no other text")
```

For the module, we create a dspy.Module class that contains the Chain of Thought predictor using the signature we defined above.
We also pass in the valid labels to the module.

Inside the forward method, we pass the text to the predictor, do a little cleaning, and return the prediction.


```python
class IntentClassificationModule(dspy.Module):
    def __init__(self, labels_in_use):
        self.intent_classifier = dspy.ChainOfThought(IntentClassification)
        self.valid_labels = set(labels_in_use)

    def forward(self, text):
        prediction = self.intent_classifier(intent=text)
        sanitized_prediction = dspy.Prediction(label=prediction.label.lower().strip().replace(" ", "_"), reasoning=prediction.reasoning)
        return sanitized_prediction
```

Lastly, we set up some the vanilla program we will use throughout the notebook.


```python
from src import MODEL_PARAMETERS, LOCAL_API_PARAMETERS
vanilla_program = IntentClassificationModule(labels_in_use)
```


```python
# Note: Run above this to do all setup without launching any models
# This is useful if you have already collected data and want to start from finetuning or from evaluation
```

We will be using a local VLLM instance to run the initial benchmarks and data collection.

# Gathering training data and running the 70B Model


## Preparation

Before running the 70B model:
1. Remember to set your HF_TOKEN and HF_HOME environment variables
2. Use the following command to start the 70B server:

   ```
   vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --port 8000 --pipeline_parallel_size 2 --enable_prefix_caching --tensor_parallel_size 2
   ```

## Parallelism Configuration

We've chosen pipeline parallelism = 2 and tensor parallelism = 2 for running the 70B model based on our current GPU setup.



```python
!serve run --non-blocking serve_70B.yaml
```

    2024-10-20 18:42:32,147	INFO scripts.py:489 -- Running config file: 'serve_70B.yaml'.
    2024-10-20 18:42:32,902	INFO worker.py:1601 -- Connecting to existing Ray cluster at address: 10.0.0.60:6379...
    2024-10-20 18:42:32,907	INFO worker.py:1777 -- Connected to Ray cluster. View the dashboard at https://session-74s48fpq311xtc218ullz9snys.i.anyscaleuserdata.com 
    2024-10-20 18:42:32,974	INFO packaging.py:359 -- Pushing file package 'gcs://_ray_pkg_22ab7971ea0ad46f991e5a47717bf76272628755.zip' (28.30MiB) to Ray cluster...
    2024-10-20 18:42:33,236	INFO packaging.py:372 -- Successfully pushed file package 'gcs://_ray_pkg_22ab7971ea0ad46f991e5a47717bf76272628755.zip'.
    INFO 2024-10-20 18:42:36,888 serve 7919 api.py:277 - Started Serve in namespace "serve".
    2024-10-20 18:42:36,895	SUCC scripts.py:540 -- Submitted deploy config successfully.
    (ServeController pid=8012) INFO 2024-10-20 18:42:36,891 controller 8012 application_state.py:881 - Deploying new app 'llm-endpoint'.
    (ServeController pid=8012) INFO 2024-10-20 18:42:36,892 controller 8012 application_state.py:457 - Importing and building app 'llm-endpoint'.
    (ProxyActor pid=8067) INFO 2024-10-20 18:42:36,867 proxy 10.0.0.60 proxy.py:1235 - Proxy starting on node 940d540c9335498800b1d2562c3435287fc2b5938ca57536626219d3 (HTTP port: 8000).



```python
llama_70b = dspy.LM(model="openai/meta-llama/Meta-Llama-3.1-70B-Instruct", **MODEL_PARAMETERS, **LOCAL_API_PARAMETERS)
```

 Here's the reasoning:

1. Model size: The 70B model has 30 parts of ~5 GB each (based on [HuggingFace documentation](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct/tree/main)).
   - Total size: 30 * 5 GB = 150 GB

2. Available VRAM:
   - Our GPUs: 80 GB VRAM x 4 = 320 GB
   - Tensor parallelism: floor(320/150) = 2
   - Pipeline parallelism: floor(num_gpus/2) = 2
   - To use all 4 GPUs efficiently:
     - Pipeline parallel size: 2
     - Tensor parallelism: 2

3. Alternative setup (8x24GB GPUs):
   - Pipeline parallel size: 1
   - Tensor parallelism: ceil(150/24) = 7

This configuration allows us to run the 70B model efficiently across our available GPU resources.

Note that on Anyscale, you CANNOT download a 70B model without changing HF_HOME on most machines. The folder `/mnt/local_storage/' has enough space for a model download. It is not persisted across cluster restarts, but that is fine for model weights we don't need to save.



```python
from src import sanity_check_program

sanity_check_program(llama_70b, vanilla_program, ft_trainset[0])
```

    Program input: Example({'label': 'extra_charge_on_statement', 'text': 'I still have not received an answer as to why I was charged $1.00 in a transaction?'}) (input_keys={'text'})


    Program output label: extra_charge_on_statement


### Bootstrap Data


In this section, we bootstrap data for fine-tuning.

We delete all the true labels to be accurate to the scenario, and then collect data from the oracle LLM.

We use a metric that checks if the prediction is in the set of labels we are using to get rid of any nonsense labels that the oracle LLM may hallucinate.


```python
from dspy.teleprompt.finetune_teleprompter import bootstrap_data, convert_to_module_level_message_data
from src import delete_labels, NUM_THREADS, write_jsonl
from src.data_preprocess import valid_label_metric

# For realism of this scenario, we are going to delete all our labels except for our test set(which is cheating and we wouldn't have in production) and our 100 true labeled examples
ft_trainset_to_label = delete_labels(ft_trainset)

with dspy.context(lm=llama_70b):
    collected_data = bootstrap_data(vanilla_program, ft_trainset_to_label, num_threads=NUM_THREADS, max_errors=10000, metric=valid_label_metric)
    # Make sure to only include the labels we are actively using or that arent hallucinated by the oracle
    collected_data_filtered = [x for x in collected_data if x["prediction"]["label"] in labels_in_use]
    
    dataset = convert_to_module_level_message_data(collected_data_filtered, program=vanilla_program, exclude_demos=True)

    dataset_formatted = [{"messages": item} for item in dataset]

print(dataset_formatted[0])
print("Length of dataset:\t", len(dataset))
```

    Average Metric: 4064 / 4071  (99.8): 100%|██████████| 4071/4071 [07:15<00:00,  9.35it/s]


    {'messages': [{'role': 'system', 'content': "Your input fields are:\n1. `intent` (str): Intent of the query\n\nYour output fields are:\n1. `reasoning` (str): ${produce the output fields}. We ...\n2. `label` (str): Type of the intent; Should just be one of the 25 labels with no other text\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## intent ## ]]\n{intent}\n\n[[ ## reasoning ## ]]\n{reasoning}\n\n[[ ## label ## ]]\n{label}\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is: \n        As a part of a banking issue traiging system, classify the intent of a natural language query into one of the 25 labels.\n        The intent should exactly match one of the following:\n        ['card_payment_fee_charged', 'direct_debit_payment_not_recognised', 'balance_not_updated_after_cheque_or_cash_deposit', 'wrong_amount_of_cash_received', 'cash_withdrawal_charge', 'transaction_charged_twice', 'declined_cash_withdrawal', 'transfer_fee_charged', 'balance_not_updated_after_bank_transfer', 'transfer_not_received_by_recipient', 'request_refund', 'card_payment_not_recognised', 'card_payment_wrong_exchange_rate', 'extra_charge_on_statement', 'wrong_exchange_rate_for_cash_withdrawal', 'refund_not_showing_up', 'reverted_card_payment', 'cash_withdrawal_not_recognised', 'activate_my_card', 'pending_card_payment', 'cancel_transfer', 'beneficiary_not_allowed', 'card_arrival', 'declined_card_payment', 'pending_top_up']"}, {'role': 'user', 'content': '[[ ## intent ## ]]\nI still have not received an answer as to why I was charged $1.00 in a transaction?\n\nRespond with the corresponding output fields, starting with the field `reasoning`, then `label`, and then ending with the marker for `completed`.'}, {'role': 'assistant', 'content': '[[ ## reasoning ## ]]\nThe user is inquiring about a $1.00 transaction charge and has not received an explanation for it, indicating a concern about an unexpected fee.\n\n[[ ## label ## ]]\nextra_charge_on_statement\n\n[[ ## completed ## ]]'}]}
    Length of dataset:	 4062



```python
# Nice utility to save the data in case you do not run the notebook all the way through
if True:
    with open("collected_data_filtered.jsonl", "w") as f:
        for item in collected_data_filtered:
            f.write(ujson.dumps({"example": item["example"], "prediction": item["prediction"]}) + "\n")
else:
    with open("collected_data_filtered.jsonl", "r") as f:
        collected_data_filtered = [ujson.loads(line) for line in f]

```

# Fine-tuning

We will use LLM Forge to fine-tune the 1B model.

In order to do this, we need to format our data into the correct format (Follows OpenAI messaging format).

Anyscale now has a first class integration with DSPy for finetuning. Anyscale offers a tool for finetuning called LLMForge, which DSPy will interface with to do the actual finetuning using your own cluster on the task you defined above.

We can let DSPy do the rest, where it will properly generate the config and run the finetuning.

Be sure to checkout the fine-tuning documentation for the latest on how to use our [API](https://docs.anyscale.com/llms/finetuning/intro) and additional [capabilities](https://docs.anyscale.com/category/fine-tuning-beta/).

We'll fine-tune our LLM by choosing a set of configurations. We have created recipes for different LLMs in the [`training configs`](configs/training/lora/llama-3-8b.yaml) folder which can be used as is or modified for experiments. These configurations provide flexibility over a broad range of parameters such as model, data paths, compute to use for training, number of training epochs, how often to save checkpoints, padding, loss, etc. We also include several [DeepSpeed](https://github.com/microsoft/DeepSpeed) [configurations](configs/deepspeed/zero_3_offload_optim+param.json) to choose from for further optimizations around data/model parallelism, mixed precision, checkpointing, etc.

We also have recipes for [LoRA](https://arxiv.org/abs/2106.09685) (where we train a set of small low ranked matrices instead of the original attention and feed forward layers) or full parameter fine-tuning. We recommend starting with LoRA as it's less resource intensive and quicker to train.


```python
from dspy.clients.lm import TrainingMethod
from src import load_finetuning_kwargs

train_data = dataset_formatted
method = TrainingMethod.SFT

finetuneable_lm = dspy.LM(model="meta-llama/Llama-3.2-1B-Instruct")

try:
    finetuning_job = finetuneable_lm.finetune(train_data=train_data, train_kwargs=load_finetuning_kwargs(), train_method=method, provider="anyscale")
    finetuning_job.result()
    model_names = finetuning_job.model_names
except Exception as e:
    print(e)
```

    Copying file:///home/ray/.dspy_cache/finetune/anyscale_b73f7a0607473b52.jsonl to gs://storage-bucket-cld-tffbxe9ia5phqr1unxhz4f7e1e/org_4snvy99zwbmh4gbtk64jfqggmj/cld_tffbxe9ia5phqr1unxhz4f7e1e/artifact_storage/anyscale_b73f7a0607473b52.jsonl
      
    .
    (anyscale +27m15.6s) Using workspace runtime dependencies env vars: {'HF_HOME': '/mnt/local_storage/huggingface', 'HF_TOKEN': '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', 'WANDB_API_KEY': 'c75a837e8271ce763121d06742fb9fc3fd2cc7f0'}.
    (anyscale +27m15.6s) Uploading local dir '.' to cloud storage.
    (anyscale +27m20.3s) Job 'dspy-llmforge-fine-tuning-job' submitted, ID: 'prodjob_j51vzutbhnsa63ngiraac98pcl'.
    (anyscale +27m20.4s) View the job in the UI: https://console.anyscale.com/jobs/prodjob_j51vzutbhnsa63ngiraac98pcl
    (anyscale +27m20.6s) Waiting for job 'prodjob_j51vzutbhnsa63ngiraac98pcl' to reach target state SUCCEEDED, currently in state: STARTING
    (anyscale +29m14.7s) Job 'prodjob_j51vzutbhnsa63ngiraac98pcl' transitioned from STARTING to RUNNING
    (anyscale +54m38.0s) Job 'prodjob_j51vzutbhnsa63ngiraac98pcl' transitioned from RUNNING to SUCCEEDED
    (anyscale +54m38.0s) Job 'prodjob_j51vzutbhnsa63ngiraac98pcl' reached target state, exiting


# Evaluation

## Performance comparisons

**Synthetic Devset:**
- 1B Non-finetuned
- 1B Non-finetuned + Prompt Optimization
- 1B Finetuned (all checkpoints)
- 1B Finetuned (all checkpoints) + Prompt Optimization

**Test set:**
- 1B Non-finetuned + Prompt Optimization
- 1B Finetuned + Prompt Optimization (best on devset)

Note that for this task, where the eval loss of a checkpoint isn't necessarily informative of the downstream performance of the program, because there are chains of though inside output, we need to test all possible checkpoints to see which one performs best.


```python
print(model_names)
```

    ['meta-llama/Llama-3.2-1B-Instruct:epochs-0-total-trained-steps-32', 'meta-llama/Llama-3.2-1B-Instruct:epochs-5-total-trained-steps-192', 'meta-llama/Llama-3.2-1B-Instruct:epochs-1-total-trained-steps-64', 'meta-llama/Llama-3.2-1B-Instruct:epochs-4-total-trained-steps-160', 'meta-llama/Llama-3.2-1B-Instruct:epochs-2-total-trained-steps-96', 'meta-llama/Llama-3.2-1B-Instruct:epochs-3-total-trained-steps-128']


We will run a local RayLLM instance that serves the model.

Provided with this template is are two files, `serve_1B.yaml` and `\model_configs\meta-llama--Llama-3_2-1B-Instruct.yaml`. 

The first file, `serve_1B.yaml`, contains the serve configuration to load the model with RayLLM.

The second file, `\model_configs\meta-llama--Llama-3_2-1B-Instruct.yaml`, contains the necessary configurations to run the 1B model.

The important part of the second file is the "dynamic_lora_loading_path" field. This is the path to the folder where the LoRA weights are stored.

DSPy will automatically save the LoRA weights to a folder in your cloud environment at $ANYSCALE_HOME/dspy/{job_id} # TODO: check

<b style="background-color: yellow;">&nbsp;🔄 REPLACE&nbsp;</b>:
Make sure you set your HF_TOKEN and HF_HOME environment variables in the workspace runtime environment variables, and run the following command to start the server:


```python
from src import update_serve_config_hf_token

update_serve_config_hf_token("serve_1B.yaml")
```

Run this command to start the RayLLM server:


```python
!serve run --non-blocking serve_1B.yaml
```

    2024-10-20 19:33:31,064	INFO scripts.py:489 -- Running config file: 'serve_1B.yaml'.
    2024-10-20 19:33:31,840	INFO worker.py:1601 -- Connecting to existing Ray cluster at address: 10.0.0.60:6379...
    2024-10-20 19:33:31,846	INFO worker.py:1777 -- Connected to Ray cluster. View the dashboard at https://session-74s48fpq311xtc218ullz9snys.i.anyscaleuserdata.com 
    2024-10-20 19:33:31,912	INFO packaging.py:359 -- Pushing file package 'gcs://_ray_pkg_27f027cf5ff3ec0ecc83ec98f5bb231bf31be798.zip' (29.67MiB) to Ray cluster...
    2024-10-20 19:33:32,189	INFO packaging.py:372 -- Successfully pushed file package 'gcs://_ray_pkg_27f027cf5ff3ec0ecc83ec98f5bb231bf31be798.zip'.
    (ProxyActor pid=30631) INFO 2024-10-20 19:33:35,797 proxy 10.0.0.60 proxy.py:1235 - Proxy starting on node 940d540c9335498800b1d2562c3435287fc2b5938ca57536626219d3 (HTTP port: 8000).
    INFO 2024-10-20 19:33:35,818 serve 30480 api.py:277 - Started Serve in namespace "serve".
    2024-10-20 19:33:35,825	SUCC scripts.py:540 -- Submitted deploy config successfully.
    (ServeController pid=30568) INFO 2024-10-20 19:33:35,820 controller 30568 application_state.py:881 - Deploying new app 'llm-endpoint'.
    (ServeController pid=30568) INFO 2024-10-20 19:33:35,821 controller 30568 application_state.py:457 - Importing and building app 'llm-endpoint'.



```python
from src import get_llama_lms_from_model_names

all_llamas = get_llama_lms_from_model_names(model_names)
```


```python
# Sanity check that the finetuned models are working

finetuned_llama = list(all_llamas.values())[1]
sanity_check_program(finetuned_llama, vanilla_program, ft_trainset[0])
```

    Program input: Example({'text': 'I still have not received an answer as to why I was charged $1.00 in a transaction?'}) (input_keys={'text'})


    Program output label: transfer_fee_charged


We are going to be doing prompt optimization using DSPy's `BootstrapFewShotWithRandomSearch (BFRS)` function.

BFRS will:
- Collect a set of chains of thought from the oracle
- Use these examples that lead to a correct prediction to "bootstrap" the program
- See which set of examples lead to the most correct predictions across your evaluation metric
- Continue this process for a set number of iterations, using the best performing programs to bootstrap the next iteration
- Return the best program

Let's go over what the hyperparameters mean:
- **max_bootstrapped_demos**: DSPy will "bootstrap" the program by collecting examples at each step that are successful and reusing those in the pipeline. This means that it will automatically collect and add chains of thought to the pipeline.
- **max_labeled_demos**: DSPy will also insert some labeled demonstrations from the training set. These would be unmodified examples from the training set that are just using the given answer.
- **num_candidate_programs**: This is the number of candidate programs that the optimizer will generate. The actual number of programs that are created is this plus three, as DSPy will also try a program with no examples, a program with just the labeled demonstrations, and a bootstrapped program with the first few examples.



```python
from src import bootstrap_fewshot_random_search_parameters, metric

print("Parameters:")
for k, v in bootstrap_fewshot_random_search_parameters.items():
    print(f"{k}: {v}")
```

    Parameters:
    max_bootstrapped_demos: 3
    max_labeled_demos: 3
    num_candidate_programs: 6



```python
from src import split_into_devset_and_optimizer_sets

def collected_data_to_example(data):
    return dspy.Example(text=data["example"]["text"], label=data["prediction"]["label"]).with_inputs("text")

collected_data_examples = [collected_data_to_example(x) for x in collected_data_filtered]

devset_synthetic, ft_optimizer_trainset, ft_optimizer_devset = split_into_devset_and_optimizer_sets(collected_data_examples, dev_size=1000, optimizer_num_val=300)
print("Lengths:")
print("Synthetic Devset:\t", len(devset_synthetic))
print("Optimizer Trainset:\t", len(ft_optimizer_trainset))
print("Optimizer Devset:\t", len(ft_optimizer_devset))
print("Example from synthetic devset:")
print(devset_synthetic[0])
```

    Lengths:
    Synthetic Devset:	 1000
    Optimizer Trainset:	 2762
    Optimizer Devset:	 300
    Example from synthetic devset:
    Example({'text': 'I still have not received an answer as to why I was charged $1.00 in a transaction?', 'label': 'extra_charge_on_statement'}) (input_keys={'text'})


Now we will take all of our checkpoints and the base mode, prompt optimize them, and evaluate them on the synthetic devset.

Note that there is a `%%capture` below. This is to suppress the output of the evaluation and prompt optimization because it is quite long. We will graph the results in the cell after. You can remove it to see the output.

You can expect this to take around 25 to 30 minutes to run.


```python
%%capture
from src import evaluate_and_prompt_optimize

evaluation_kwargs = {
    "models": all_llamas,
    "module_class": IntentClassificationModule,
    "optimizer_trainset": ft_optimizer_trainset,
    "optimizer_valset": ft_optimizer_devset,
    "devset": devset_synthetic,
    "metric": metric,
    "labels_in_use": labels_in_use
}

ft_results = evaluate_and_prompt_optimize(**evaluation_kwargs)
```


```python
if True:
    import json
    with open("ft_results.json", "w") as f:
        json.dump(ft_results, f)
else:
    ft_results = json.load(open("ft_results.json"))
```


```python
from src import graph_devset_results, graph_testset_results

graph_devset_results(ft_results)
```


    
![png](README_files/README_53_0.png)
    


    Highest Dev Set Score: 60.2, Model: Epoch 4


We see that the highest performing model is the final epoch with a score of 50.2 on our synthetic devset.

We will now take this best performing model and evaluate it and our prompt optimized base model on the true test set to see if we have improved performance.

This should take around X minutes


```python
%%capture
# Now we need to evaluate the test set
from src import run_testset_evaluation

testset_evaluation_kwargs = {
    "ft_results": ft_results,
    "all_llamas": all_llamas,
    "labels_in_use": labels_in_use,
    "testset": testset,
    "metric": metric,
    "module_class": IntentClassificationModule
}

ft_results_testset, (best_program_path, best_model, best_score) = run_testset_evaluation(**testset_evaluation_kwargs)
```


```python
graph_testset_results(ft_results_testset)
```


    
![png](README_files/README_56_0.png)
    



```python
print(f"Best testset result: \n{best_model} with score: {best_score}")
```

    Best testset result: 
    meta-llama/Llama-3.2-1B-Instruct:epochs-4-total-trained-steps-160 with score: 54.0


# Serving

<b style="background-color: blue;">&nbsp;🔄 RUN (optional)&nbsp;</b>:
You can optionally deploy your model to Anyscale in order to use it in production.
To do this, run the following command:

```
!anyscale service deploy -f serve_1B.yaml
```

Follow the URL in order to find your service URL and API key for your deployed service.

If you choose not to deploy your model, you can run the following code to run the model locally.
```
serve run serve_1B.yaml
```

If you never took down your service from the previous section, there is no need to rerun the service run command.


```python
# !anyscale service deploy -f serve_1B.yaml
# !serve run serve_1B.yaml
```

<b style="background-color: yellow;">&nbsp;🔄 REPLACE&nbsp;</b>:
Replace the following variables with your Anyscale service URL and API key.

```
ANYSCALE_SERVICE_BASE_URL = None
ANYSCALE_API_KEY = None
```

You can find them by clicking the query button on the Anyscale dashboard for your service.

<!-- <img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-dspy-workflow/files/service-query.png" alt="Service Query" width="500"> -->
![Service Query](README_files/service-query.png)


```python
ANYSCALE_SERVICE_BASE_URL = None
ANYSCALE_API_KEY = None
```


```python
from src import MODEL_PARAMETERS, LOCAL_API_PARAMETERS
if ANYSCALE_SERVICE_BASE_URL and ANYSCALE_API_KEY:
    API_PARAMETERS = {"api_base": ANYSCALE_SERVICE_BASE_URL, "api_key": ANYSCALE_API_KEY}
else:
    API_PARAMETERS = LOCAL_API_PARAMETERS
```

Now we can use ray serve in order to deploy our DSPy program.

The RayLLM instance you deployed will autoscale according to the number of requests you make based on the configuration inside of the `serve_1B.yaml` file.

Ray serve does all the hard work for you there, so all you need to do is provide the URL and API key to query your model.

Now to deploy the DSPy program on top of the RayLLM instance, we can create a FastAPI wrapper around our DSPy program.


```python
from ray import serve
from fastapi import FastAPI

app = FastAPI()

@serve.deployment(
    ray_actor_options={"num_cpus": 0.1},
    autoscaling_config=dict(min_replicas=1, max_replicas=3)
)
@serve.ingress(app)
class LLMClient:
    def __init__(self):
        self.llm = dspy.LM(model="openai/" + best_model, **MODEL_PARAMETERS, **API_PARAMETERS)
        dspy.settings.configure(experimental=True, lm=self.llm)
        self.program = IntentClassificationModule(labels_in_use)
        self.program.load(best_program_path)

    @app.get("/")
    async def classify_intent(
        self,
        query: str,
    ):
        """Answer the given question and provide sources."""
        retrieval_response = self.program(query)

        return retrieval_response.label

llm_client = LLMClient.bind()
llm_handle = serve.run(llm_client, route_prefix="/classify_intent", name="llm_client")
```

    INFO 2024-10-19 01:34:51,639 serve 14470 api.py:259 - Connecting to existing Serve app in namespace "serve". New http options will not be applied.
    WARNING 2024-10-19 01:34:51,641 serve 14470 api.py:85 - The new client HTTP config differs from the existing one in the following fields: ['host']. The new HTTP config is ignored.


    INFO 2024-10-19 01:34:57,689 serve 14470 client.py:492 - Deployment 'LLMClient:xrueus8l' is ready at `http://0.0.0.0:8000/classify_intent`. component=serve deployment=LLMClient
    INFO 2024-10-19 01:34:57,692 serve 14470 api.py:549 - Deployed app 'llm_client' successfully.



```python
example_query = ft_trainset[1]["text"]
llm_response = await llm_handle.classify_intent.remote(
    query=example_query,
)
print(example_query)
print(llm_response)
```

    My card was charged more than expected.
    card_payment_fee_charged


We can also query directly using HTTP requests, because we use the `@app` decorator on our FastAPI app.


```python
import requests
try:
    response = requests.get(f"http://localhost:8000/classify_intent/classify_intent?query={example_query}")
    print(response.json())
except Exception as e:
    print(e)
```

    card_payment_fee_charged


<b style="background-color: yellow;">&nbsp;🛑 IMPORTANT&nbsp;</b>: Please `Terminate` your service from the Service page to avoid depleting your free trial credits.


```python
# Clean up
!python src/clear_cell_nums.py
!find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
!find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
!rm -rf __pycache__ data .HF_TOKEN deploy/services
```
