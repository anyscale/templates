# End-to-end DSPy Workflows Guide 

Time to complete: X Hours

## Problem

You are a bank, and you want to categorize customer support queries into one of 25 categories. You have hand labeled 100 examples and have collected 4,000 unlabeled examples.

You want to use an LLM to solve this problem, because you don't have enough labeled data to train a traditional classifier.

You also don't want to spend a lot of money on inference. 

## Motivation

You decide to use DSPy to solve this problem because the following flow is fairly difficult to orchestrate manually:
Data Collection/Labeling -> Fine-tuning -> Prompt Optimization -> Evaluation -> Deployment

By using DSPy on Anyscale, you can easily orchestrate this flow and solve the problem.

## Solution

You don't want pay to host a 70B model, so you will instead finetune a 1B model, which you can host and serve at a low cost using Anyscale's RayLLM offering.

In order to help the 1B model understand the reasoning behind why the 70B model makes certain classifications, you will use Chains of Thought to distill knowledge from the 70B model.

So what will this look like?

1. Collect 4,000 unlabeled examples
2. Label all of them with your 70B oracle model running locally
3. Use the new DSPy finetuning tools to finetune a 1B model
- This takes about 20 minutes on 4xA100-80GB GPUs, and uses Anyscale's LLMForge in the background to finetune the model
4. Evaluate and prompt optimize your 1B model checkpoints against the labeled dataset
5. Take the best performing 1B checkpoint and compare it to the un-finetuned 1B model on the true test set
6. Deploy the optimized 1B model/DSPy pipeline to production using Anyscale's RayLLM # TODO

Note(isaac): we arent doing anything with the 100 labeled examples yet

# Table of Contents

## Set Up
1. Installing DSPy
2. Setting up the environment
3. Loading the dataset
4. Setting up the program, metric, and evaluator

## Data Collection

1. Collect 4,000 unlabeled examples
2. Label all of them with your 70B oracle model running locally

## Fine-tuning

1. Use the new DSPy finetuning tools to finetune a 1B model
- This takes about 20 minutes on 4xA100-80GB GPUs, and uses Anyscale's LLMForge in the background to finetune the model

## Evaluation
1. Evaluate and prompt optimize your 1B model checkpoints against the labeled dataset
2. Find the best performing 1B checkpoint and compare it to the un-finetuned 1B model on the true test set

## Serving
1. Deploy the optimized 1B model/DSPy pipeline to production using Anyscale's RayLLM # TODO

## Future Work and Open Questions
- Efficient batch inference with a DSPy pipeline
- Exploring different fine-tuning methods and hyperparameter sweeps

This guide aims to provide a comprehensive overview of building, optimizing, and deploying LLM pipelines using DSPy and Anyscale.

## Set up

Node Set up:

We will be running everything on a head node that uses 4xA100-80GB GPUs. I find that L4s are usually available and suitable for this usecase. You can also use any more powerful node.

To change to use A100 GPUs, click the "1 active node" in the top right corner, then for workspace node, click the pencil icon and navigate to the A100 tab and select the 4xA100 option. If you do not see A100 in the list of GPUs, they may not be available on your cloud.


```python
%load_ext autoreload
%autoreload 2
```


```python
# TODO: Pin to a certain branch of DSPy until merged into main

# !pip install -e dspy

# ignore future warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```


```python
import dspy
import dsp
import os
import ujson

from dotenv import load_dotenv

cache_dir = "/home/ray/default/dspy/cache"
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# I have included a .env.example with the necessary environment variables to be set
# You can also set them manually if you prefer

os.environ["DSP_CACHEDIR"] = cache_dir

load_dotenv()

dspy.settings.configure(experimental=True)
```


```python
import litellm

litellm.set_verbose=False
litellm.suppress_debug_info=True
```


```python
necessary_env_vars = [
    "DSP_CACHEDIR",
    "HF_TOKEN",
    "HF_HOME"
]

for var in necessary_env_vars:
    assert os.environ[var], f"{var} is not set"
```


```python
import ray

if not ray.is_initialized():
    ray.init(runtime_env={"env_vars": os.environ, "py_modules": [dspy, dsp]})
```

We will make use of a random number generator in this notebook. We are creating a Random object here to ensure that our notebook is reproducible.


```python
import random

rng = random.Random(0)
```

We will be using the PolyAI/banking77 dataset for this tutorial. We use the built in dspy DataLoader to load the dataset from Huggingface as a list of dspy.Example objects.


```python
# Prepare the dataset
from dspy.datasets import DataLoader

dl = DataLoader()
full_trainset = dl.from_huggingface(
    dataset_name="PolyAI/banking77", # Dataset name from Huggingface
    fields=("label", "text"), # Fields needed
    input_keys=("text",), # What our model expects to recieve to generate an output
    split="train"
)

full_testset = dl.from_huggingface(
    dataset_name="PolyAI/banking77", # Dataset name from Huggingface
    fields=("label", "text"), # Fields needed
    input_keys=("text",), # What our model expects to recieve to generate an output
    split="test"
)
```


```python
# Mapping of the label string to an integer
int_to_label_dict = {
    0: "activate_my_card",
    1: "age_limit",
    2: "apple_pay_or_google_pay",
    3: "atm_support",
    4: "automatic_top_up",
    5: "balance_not_updated_after_bank_transfer",
    6: "balance_not_updated_after_cheque_or_cash_deposit",
    7: "beneficiary_not_allowed",
    8: "cancel_transfer",
    9: "card_about_to_expire",
    10: "card_acceptance",
    11: "card_arrival",
    12: "card_delivery_estimate",
    13: "card_linking",
    14: "card_not_working",
    15: "card_payment_fee_charged",
    16: "card_payment_not_recognised",
    17: "card_payment_wrong_exchange_rate",
    18: "card_swallowed",
    19: "cash_withdrawal_charge",
    20: "cash_withdrawal_not_recognised",
    21: "change_pin",
    22: "compromised_card",
    23: "contactless_not_working",
    24: "country_support",
    25: "declined_card_payment",
    26: "declined_cash_withdrawal",
    27: "declined_transfer",
    28: "direct_debit_payment_not_recognised",
    29: "disposable_card_limits",
    30: "edit_personal_details",
    31: "exchange_charge",
    32: "exchange_rate",
    33: "exchange_via_app",
    34: "extra_charge_on_statement",
    35: "failed_transfer",
    36: "fiat_currency_support",
    37: "get_disposable_virtual_card",
    38: "get_physical_card",
    39: "getting_spare_card",
    40: "getting_virtual_card",
    41: "lost_or_stolen_card",
    42: "lost_or_stolen_phone",
    43: "order_physical_card",
    44: "passcode_forgotten",
    45: "pending_card_payment",
    46: "pending_cash_withdrawal",
    47: "pending_top_up",
    48: "pending_transfer",
    49: "pin_blocked",
    50: "receiving_money",
    51: "refund_not_showing_up",
    52: "request_refund",
    53: "reverted_card_payment",
    54: "supported_cards_and_currencies",
    55: "terminate_account",
    56: "top_up_by_bank_transfer_charge",
    57: "top_up_by_card_charge",
    58: "top_up_by_cash_or_cheque",
    59: "top_up_failed",
    60: "top_up_limits",
    61: "top_up_reverted",
    62: "topping_up_by_card",
    63: "transaction_charged_twice",
    64: "transfer_fee_charged",
    65: "transfer_into_account",
    66: "transfer_not_received_by_recipient",
    67: "transfer_timing",
    68: "unable_to_verify_identity",
    69: "verify_my_identity",
    70: "verify_source_of_funds",
    71: "verify_top_up",
    72: "virtual_card_not_working",
    73: "visa_or_mastercard",
    74: "why_verify_identity",
    75: "wrong_amount_of_cash_received",
    76: "wrong_exchange_rate_for_cash_withdrawal"
}

label_to_int_dict = {v: k for k, v in int_to_label_dict.items()}
```


```python
def convert_int_to_label(example):
    example["label"] = int_to_label_dict[example["label"]]
    return example

full_trainset = [convert_int_to_label(example) for example in full_trainset]
full_testset = [convert_int_to_label(example) for example in full_testset]

print("Example training set: ", full_trainset[0])
```

The dataset is originally called "banking77" because there are 77 labels. We will be reducing this to the top 25 most frequent labels.


```python
from collections import Counter

label_counts = Counter(example['label'] for example in full_trainset)

# get the top 25 labels sorted by name for any ties
top_25_labels = sorted(label_counts.keys(), key=lambda x: (-label_counts[x], x))[:25]

# Filter the datasets to only include examples with the top 25 labels
full_trainset_filtered = [example for example in full_trainset if example['label'] in top_25_labels]
full_testset_filtered = [example for example in full_testset if example['label'] in top_25_labels]

print(f"Dataset filtered to top 25 labels. New sizes:")
print(f"Training set size: {len(full_trainset_filtered)}")
print(f"Test set size: {len(full_testset_filtered)}")
print(f"Top 25 labels: {', '.join(str(label) for label in top_25_labels)}")

print(full_trainset_filtered[0])
```

We need to pass the labels to the LLM somehow.

In DSPy, we can do this by either including it in the docstring of the program or by adding it as an input field to the Signature.

Here, we will add it to the docstring, because the set of labels is fixed.


```python
labels_in_use = top_25_labels
print(labels_in_use)
```

Now we will shuffle our training set and split it into a training and labeled set.

The scenario we are emulating is that we only have 100 labeled examples to train on. We are saying that we have 4K (length of the training set) unlabeled examples we can then label using an oracle model, and then distill the knowledge from the oracle model into our 1B model.


```python
shuffled_trainset = [d for d in full_trainset_filtered]
rng.shuffle(shuffled_trainset)

# The devset shouldn't overlap
ft_trainset = shuffled_trainset[:-100]
labeled_trainset = shuffled_trainset[-100:]

testset = full_testset
```

This is a simple, 1 step Chain of Thought program.

In DSPy, you define a Signature to show your inputs and outputs. You define a module to run the different steps of your program.

Our signature has a note at the top containing a simple prompt along with the list of valid outputs.

We then have an `intent` field which is the input to the program.

Finally we have a `label` field which is the output of the program.

We give both of these fields a short description.


```python
from dspy.evaluate import Evaluate

# We are setting the experimental flag to True to make use of the fine-tuning
# features that are still in development.
dspy.settings.configure(experimental=True)

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
    def __init__(self):
        self.intent_classifier = dspy.ChainOfThought(IntentClassification)
        self.valid_labels = set(labels_in_use)

    def forward(self, text):
        prediction = self.intent_classifier(intent=text)
        sanitized_prediction = dspy.Prediction(label=prediction.label.lower().strip().replace(" ", "_"), reasoning=prediction.reasoning)
        return sanitized_prediction
```

Here we set up the metric and evaluator. We will be using the answer exact match metric.

The evaluator is what we will consider as our test set.

We choose `num_threads=300` because a single VLLM instance can anecdotally handle ~256 concurrent requests and it will have 50 requests pending.

The `adjusted_exact_match` is because the default `answer_exact_match` metric checks that example.answer and pred.answer are equal rather than checking the label itself.


```python
# Prepare the metric and evaluator
from dspy.evaluate import answer_exact_match

NUM_THREADS = 300

def adjusted_exact_match(example, pred, trace=None, frac=1.0):
    example.answer = example.label
    pred.answer = pred.label
    return answer_exact_match(example, pred, trace, frac)

metric = adjusted_exact_match
common_kwargs = dict(metric=metric, num_threads=NUM_THREADS, display_progress=True, max_errors=10000)

evaluate_testset = Evaluate(devset=testset, **common_kwargs)
```


```python
def read_jsonl(filename):
    with open(filename, "r") as f:
        return [ujson.loads(line) for line in f]

def write_jsonl(filename, data):
    with open(filename, "w") as f:
        for item in data:
            f.write(ujson.dumps(item) + "\n")
```

Lastly, we set up some parameters we will use throughout the notebook.


```python
MAX_TOKENS = 1000
MODEL_PARAMETERS = {
  "max_tokens": MAX_TOKENS,
  "temperature": 0,
}

LOCAL_API_PARAMETERS = {
  "api_base": "http://localhost:8000/v1",
  "api_key": "fake-key-doesnt-matter"
}
vanilla_program = IntentClassificationModule()
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

We've chosen pipeline parallelism and tensor parallelism of 2 for the 70B model based on our current setup. Here's the reasoning:

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

Note that on Anyscale, you CANNOT download a model without changing HF_HOME on most machines. The folder `/mnt/local_storage/' has enough space for a model download. It is not persisted across cluster restarts, but that is fine for model weights we don't need to save.


```python
# Command for easy copying: 
# `export HF_HOME=/mnt/local_storage/huggingface; vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --port 8000 --pipeline_parallel_size 2 --enable_prefix_caching --tensor_parallel_size 2`

input("Press Enter once you have the vllm server running...")
```


```python
llama_70b = dspy.LM(model="openai/meta-llama/Meta-Llama-3.1-70B-Instruct", **MODEL_PARAMETERS, **LOCAL_API_PARAMETERS)
```


```python
# sanity check
test_predictor = IntentClassificationModule()
with dspy.context(lm=llama_70b):
    sample_input = ft_trainset[15]
    print(sample_input)
    print(test_predictor(**sample_input.inputs()).label)
```


```python
# Lets see how the vanilla program performs on our small labeled dataset
# vanilla_program = IntentClassificationModule()
# with dspy.context(lm=llama_70b):
#     print("Evaluating the vanilla program on the devset using llama 70B...")
#     true_labeled_eval = evaluate_devset(vanilla_program)
```


Now we are ready to optimize the pipeline. We want to optimize the 70B pipeline in order to get the best possible data to then train our 8B model.


```python
# Optimization hyperparameters
from dspy.teleprompt.random_search import BootstrapFewShotWithRandomSearch

# Define the hyperparameters for prompt optimization
MAX_BOOTSTRAPPED_DEMOS = 3
MAX_LABELED_DEMOS = 3
NUM_CANDIDATE_PROGRAMS = 6
OPTIMIZER_NUM_TRAIN = 100
OPTIMIZER_NUM_VAL = 300
```


```python
# We have added this flag to save you some compute and time while running the notebook
# TODO: do prompt optimization on the 100 labeled examples
# COMPILE_PROGRAM = False
# EVAL_PROGRAM = True

# # Compile the optimizer and evaluate
# with dspy.context(lm=llama_70b):
#     vanilla_program = IntentClassificationModule()
#     if COMPILE_PROGRAM:
#         bfrs_base_program = bfrs_optimizer.compile(vanilla_program, trainset=po_trainset, valset=po_devset)
#         bfrs_base_program.save(f"b25_70b_31_bfrs_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}.json")
#     else:
#         bfrs_base_program = IntentClassificationModule()
#         bfrs_base_program.load(f"b25_70b_31_bfrs_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}.json")
#     if EVAL_PROGRAM:
#         llama_70b_bfrs_eval = evaluate_devset(bfrs_base_program)

```

### Bootstrap Data


In this section, we bootstrap data for fine-tuning. In the code block below, we are deciding which program should be used to collect the bootstraps. We are setting this to the prompt optimized program, but one could also set this to the vanilla program, though doing so would lead to lower quality bootstraps.


```python
# Lets do some data analysis and cleaning
# First we will convert the data to a pandas dataframe
import pandas as pd
from dspy.teleprompt.finetune_teleprompter import bootstrap_data, convert_to_module_level_message_data

# TODO: WORKING HERE

# For realism of this scenario, we are going to delete all our labels except for our test set(which is cheating and we wouldn't have in production) and our 100 true labeled examples
def delete_labels(dataset):
    for example in dataset:
        if "label" in example:
            del example["label"]
    return dataset

ft_trainset_to_label = delete_labels(ft_trainset)
with dspy.context(lm=llama_70b):
    collected_data = bootstrap_data(vanilla_program, ft_trainset_to_label, num_threads=NUM_THREADS, max_errors=10000)
    print(collected_data[0])
    collected_data_filtered = [x for x in collected_data if x["prediction"]["label"] in labels_in_use]

    # Convert collected_data to a pandas DataFrame
    dataset = convert_to_module_level_message_data(collected_data_filtered, program=vanilla_program, exclude_demos=True)
    
print(dataset[0])
print(len(dataset))
```


```python
print("Data points removed:", len(collected_data) - len(collected_data_filtered))
# look at all the data that was removed to see if any can be cleaned
filtered_data = [x for x in collected_data if x["prediction"]["label"] not in labels_in_use]
bad_labels = [x["prediction"]["label"] for x in filtered_data]
print(bad_labels)
```


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


```python
# If we had an eval set, we would add it here, but we are using all of our examples for finetuning
dataset_filenames = {f"trainset_data_banking.jsonl": dataset}

for filename, data in dataset_filenames.items():
    messages_format = [{"messages": item} for item in data]
    write_jsonl(filename, messages_format)
```

# Fine-tuning

We will use LLM Forge to fine-tune the 1B model.

In order to do this, we need to format our data into the correct format (Follows OpenAI messaging format placed in a jsonl file).

We initially saved the data into a json file in prompt-completion format.

In order to prepare for finetuning, we need to do three steps:
1. Format the data into the correct format and verify that the data is valid
2. Upload the data to GCP
3. Generate the compute configuration file

After the compute configuration file is generated, we can submit the job to LLM Forge, using either the command line or using the anyscale jobs sdk.
TODO: Add the anyscale jobs sdk submit method

Be sure to checkout the fine-tuning documentation for the latest on how to use our [API](https://docs.anyscale.com/llms/finetuning/intro) and additional [capabilities](https://docs.anyscale.com/category/fine-tuning-beta/).

We'll fine-tune our LLM by choosing a set of configurations. We have created recipes for different LLMs in the [`training configs`](configs/training/lora/llama-3-8b.yaml) folder which can be used as is or modified for experiments. These configurations provide flexibility over a broad range of parameters such as model, data paths, compute to use for training, number of training epochs, how often to save checkpoints, padding, loss, etc. We also include several [DeepSpeed](https://github.com/microsoft/DeepSpeed) [configurations](configs/deepspeed/zero_3_offload_optim+param.json) to choose from for further optimizations around data/model parallelism, mixed precision, checkpointing, etc.

We also have recipes for [LoRA](https://arxiv.org/abs/2106.09685) (where we train a set of small low ranked matrices instead of the original attention and feed forward layers) or full parameter fine-tuning. We recommend starting with LoRA as it's less resource intensive and quicker to train.


```python
from dsp.modules.lm import TrainingMethod
from dspy.clients.lm import LM
from dspy.clients.anyscale import FinetuneJobAnyScale

train_path = f"trainset_data_banking.jsonl"
eval_path = None
method = TrainingMethod.SFT

kwargs = {
    "hyperparameters": {
        "num_devices": 4,
        "trainer_resources": None,
        "worker_resources": None,
        "generation_config": {
            "prompt_format": {
                "system": "<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>",
                "user": "<|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|>",
                "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{instruction}<|eot_id|>",
                "trailing_assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "bos": "<|begin_of_text|>",
                "system_in_user": False,
                "default_system_message": ""
            },
        },
        "learning_rate": 3e-5,
        "num_epochs": 6,
        "train_batch_size_per_device": 32
    },
    "use_lora": True,
    # TODO: I think this needs to be set dynamically
    # "lora_dynamic_folder": "dspy/lora_weights/prodjob_qmulcjw4x8z599m8hkyja8tbmi/meta-llama/Llama-3.2-1B-Instruct"
}

finetuneable_lm = dspy.LM(model="meta-llama/Llama-3.2-1B-Instruct")
finetuning_job = await finetuneable_lm.finetune(method, train_path, eval_path, provider="anyscale", train_kwargs=kwargs)
model_names = []
async for model_name in finetuning_job:
    model_names.append(model_name)
    

```


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    Cell In[1], line 1
    ----> 1 from dsp.modules.lm import TrainingMethod
          2 from dspy.clients.lm import LM
          3 from dspy.clients.anyscale import FinetuneJobAnyScale


    ImportError: cannot import name 'TrainingMethod' from 'dsp.modules.lm' (/home/ray/default/dspy-1/dsp/modules/lm.py)



```python
# TODO: Currently I am manually updating HF_TOKEN
# Also copying the prodjob into the serving file
```

# Evaluation

Throughout this section, anything using 8B model (or technically 70B too) should use the new evaluate with ray data batch offline(or technically online) inference.

Probably worth testing offline with 8x8 threads vs just 64 threads to see if it makes a meaningful difference.

## Performance comparisons

- 70B
- 70B BSFS
- 8B
- 8B BSFT
- 8B BSFT + BSFS

The first model to run is the 8B model in order to collect a baseline of performance.

You can run the local VLLM instance with the following command:

Make sure to set your HF_TOKEN and HF_HOME environment variables

For Anyscale, putting models into /mnt/local_storage is a typical pattern.


`vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000 --pipeline_parallel_size 4 --enable_prefix_caching`

Lets break down what this command does:
- `vllm serve` is the command to run the VLLM server
- `meta-llama/Meta-Llama-3.1-8B-Instruct` is the model to run
- `--port 8000` is the port to run the server on
- `--pipeline_parallel_size 4` is the number of pipeline parallel size to run the server with. We are using 4 because we have 4 GPUs all of which can hold an instance of the model.
- `--enable_prefix_caching` is the flag to enable the prefix caching. This will store and reuse the beginnings of prompts to avoid repeating the same computation. This is especially useful for DSPy since we are almost always using prompts with the same beginning parts in the form of few shot demonstrations.


```python
print(model_names)
```


```python
# TODO IMPORTANT IMPORTANT IMPORTANT
import json
if False:
    if model_names is not None:
        with open("model_names.json", "w") as f:
            json.dump(model_names, f)
else:
    with open("model_names.json", "r") as f:
        model_names = json.load(f)

```


```python
# Command for easy copying: 
llama_1b = dspy.LM(model="openai/meta-llama/Llama-3.2-1B-Instruct", **LOCAL_API_PARAMETERS, **MODEL_PARAMETERS)
finetuned_llamas_1b = {f: dspy.LM(model="openai/" + f, **LOCAL_API_PARAMETERS, **MODEL_PARAMETERS) for f in model_names}
all_llamas = {**finetuned_llamas_1b, "base": llama_1b}
```


```python
# # Sanity check that the finetuned models are working
llama1 = list(finetuned_llamas_1b.values())[3]
with dspy.context(lm=llama1):
    print(llama1.model)
    test_predictor = IntentClassificationModule()
    sample_input = ft_trainset[11]
    print(sample_input)
    print(test_predictor(**sample_input.inputs()).label)
```

Now let's try optimizing the program with the finetuned model

Now we know how well the base pipeline performs, let's run prompt optimization on the pipeline in order to juice up the performance.

Let's go over what the hyperparameters mean:
- MAX_BOOTSTRAPPED_DEMOS: DSPy will "bootstrap" the program by collecting examples at each step that are successful and reusing those in the pipeline. This means that it will automatically collect and add chains of thought to the pipeline.
- MAX_LABELED_DEMOS: DSPy will also insert some labeled demonstrations from the training set. These would be unmodified examples from the training set that are just using the gold answer.
- NUM_CANDIDATE_PROGRAMS: This is the number of candidate programs that the optimizer will generate. The actual number of programs that are created is this plus three, as DSPy will also try a program with no examples, a program with TODO (check)
- OPTIMIZER_NUM_TRAIN and OPTIMIZER_NUM_VAL: These are the number of examples that the optimizer will use for training and validation. Note that we will be taking the "validation" set from the trainset so as the actual validation set is untouched.


```python
# Initialize the optimizer
bfrs_optimizer = BootstrapFewShotWithRandomSearch(
    metric=metric,
    max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS,
    max_labeled_demos=MAX_LABELED_DEMOS,
    num_candidate_programs=NUM_CANDIDATE_PROGRAMS,
    num_threads=NUM_THREADS,
    max_errors=10000
)
```


```python
DEV_SIZE = 1000
```


```python
def collected_data_to_example(data):
    return dspy.Example(text=data["example"]["text"], label=data["prediction"]["label"]).with_inputs("text")

collected_data_examples = [collected_data_to_example(x) for x in collected_data_filtered]

devset_synthetic = collected_data_examples[:DEV_SIZE]
ft_optimizer_devset = collected_data_examples[DEV_SIZE:DEV_SIZE+OPTIMIZER_NUM_VAL]
ft_optimizer_trainset = collected_data_examples[DEV_SIZE+OPTIMIZER_NUM_VAL:]


print(len(devset_synthetic), len(ft_optimizer_trainset), len(ft_optimizer_devset))
print(devset_synthetic[0])
```


```python
COMPILE_PROGRAM = True

ft_results = {}
for folder, llama in all_llamas.items():
    print("Evaluating", llama.model)
    ft_results[folder] = {}
    with dspy.context(lm=llama):
        evaluate_devset = Evaluate(devset=devset_synthetic, metric=metric, num_threads=NUM_THREADS, display_progress=True, max_errors=10000)
        
        vanilla_program = IntentClassificationModule()
        devset_result = evaluate_devset(vanilla_program)
        ft_results[folder]["vanilla"] = {"devset": devset_result}

        if COMPILE_PROGRAM:
            bfrs_finetuned_program = bfrs_optimizer.compile(vanilla_program, trainset=ft_optimizer_trainset, valset=ft_optimizer_devset)
            bfrs_finetuned_program.save(f"simpleintent_1b_32_ft_bfrs_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}_{folder.split('/')[-1]}.json")
        else:
            bfrs_finetuned_program = IntentClassificationModule()
            bfrs_finetuned_program.load(f"simpleintent_1b_32_ft_bfrs_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}_{folder.split('/')[-1]}.json")
        
        llama_8b_bfrs_finetuned_eval = evaluate_devset(bfrs_finetuned_program)
        ft_results[folder]["bfrs"] = {"devset": llama_8b_bfrs_finetuned_eval, "true_labels": None, "testset": None}
        print(f"result for {folder}: {llama_8b_bfrs_finetuned_eval}, None, None")

```


```python
if True:
    import json
    with open("ft_results.json", "w") as f:
        json.dump(ft_results, f)
else:
    ft_results = json.load(open("ft_results.json"))
ft_results

```


```python
# Now we need to evaluate the test set
best_non_base_model = max([x for x in ft_results.keys() if x != "base"], key=lambda x: ft_results[x]["bfrs"]["devset"])
print("Best non-base model:", best_non_base_model)
base_and_best = {"base": all_llamas["base"], best_non_base_model: all_llamas[best_non_base_model]}

for folder, llama in base_and_best.items():
    print("Evaluating", folder)
    vanilla_program = IntentClassificationModule()
    
    with dspy.context(lm=llama):
        testset_result_vanilla = evaluate_testset(vanilla_program)
        ft_results[folder]["vanilla"]["testset"] = testset_result_vanilla
        vanilla_program.load(f"simpleintent_1b_32_ft_bfrs_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}_{folder.split('/')[-1]}.json")
        testset_result = evaluate_testset(vanilla_program)
        ft_results[folder]["bfrs"]["testset"] = testset_result


```


```python
ft_results
```


```python
import json
if False:
    ft_results = json.load(open("ft_results.json"))
    print(ft_results)
else:
    with open("ft_results.json", "w") as f:
        json.dump(ft_results, f)

```


```python
ft_results
```


```python
import json
import matplotlib.pyplot as plt
import numpy as np

# Prepare data for plotting
models = []
vanilla_devset = []
bfrs_devset = []
vanilla_testset = []
bfrs_testset = []

for model, results in ft_results.items():
    if model == "base":
        models.append("base")
    else:
        models.append("Epoch " + model.split(':')[1].split('-')[1])  # Extract epoch information
    vanilla_devset.append(results['vanilla']['devset'])
    bfrs_devset.append(results['bfrs']['devset'])
    vanilla_testset.append(results['vanilla'].get('testset', None))
    bfrs_testset.append(results['bfrs'].get('testset', None))

# Sort the data by epoch, keeping "base" at the beginning
sorted_data = sorted(zip(models, vanilla_devset, bfrs_devset, vanilla_testset, bfrs_testset),
                     key=lambda x: (x[0] != "base", x[0]))
models, vanilla_devset, bfrs_devset, vanilla_testset, bfrs_testset = zip(*sorted_data)

for i in range(len(models)):
    print(models[i], "vanilla_devset", vanilla_devset[i], "bfrs_devset", bfrs_devset[i], "vanilla_testset", vanilla_testset[i], "bfrs_testset", bfrs_testset[i])

# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Adjust bar positions and width
x = np.arange(len(models))
width = 0.35

# Plot bars for Dev Set (top graph)
ax1.bar(x - width/2, vanilla_devset, width, label='No Prompt Optimization', color='skyblue')
ax1.bar(x + width/2, bfrs_devset, width, label='BootstrapFewShotRandomSearch', color='lightgreen')

# Add value labels on top of each bar for Dev Set
for i, v in enumerate(vanilla_devset):
    ax1.text(i - width/2, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
for i, v in enumerate(bfrs_devset):
    ax1.text(i + width/2, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

# Customize the Dev Set plot
ax1.set_ylabel('Dev Set Scores')
ax1.set_title('Model Performance Comparison Across Epochs (Synthetic Dev Set; N=1000)')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# Find the highest devset score and its corresponding model
highest_devset_score = max(bfrs_devset)
highest_score_model = models[bfrs_devset.index(highest_devset_score)]

print(highest_devset_score, highest_score_model, bfrs_devset[models.index(highest_score_model)])

# Prepare data for the bottom graph
# Prepare data for the bottom graph (Test Set)
base_vanilla_testset = vanilla_testset[models.index("base")]
base_bfrs_testset = bfrs_testset[models.index("base")]

best_model_index = bfrs_devset.index(highest_devset_score)
best_vanilla_testset = vanilla_testset[best_model_index]
best_bfrs_testset = bfrs_testset[best_model_index]

print("best_vanilla_testset", best_vanilla_testset, "best_bfrs_testset", best_bfrs_testset)
print("base_vanilla_testset", base_vanilla_testset, "base_bfrs_testset", base_bfrs_testset)

# Plot bars for Test Set (bottom graph)
models_to_plot = ["Base Model", f"Best Model ({highest_score_model})"]
x_test = np.arange(len(models_to_plot))

ax2.bar(x_test - width/2, [base_vanilla_testset, best_vanilla_testset], width, label='No Prompt Optimization', color='skyblue')
ax2.bar(x_test + width/2, [base_bfrs_testset, best_bfrs_testset], width, label='BootstrapFewShotRandomSearch', color='lightgreen')

# Add value labels on top of each bar for Test Set
for i, v in enumerate([base_vanilla_testset, best_vanilla_testset]):
    ax2.text(i - width/2, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
for i, v in enumerate([base_bfrs_testset, best_bfrs_testset]):
    ax2.text(i + width/2, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

# Customize the Test Set plot
ax2.set_ylabel('Test Set Scores')
ax2.set_title('Model Performance Comparison (Real Test Set; N=1000)')
ax2.set_xticks(x_test)
ax2.set_xticklabels(models_to_plot)
ax2.legend()
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

```

Lastly, lets give the base 8B model a fair chance by prompt optimizing it.


```python
# Now we can compare all iterations of this pipeline
print(f"Results for HotPotQA fine-tuning LLaMa 8B with a starting trainset")
print(f"    70B model (vanilla program): {llama_70b_base_eval}")
print(f"    70B model (bfrs program): {llama_70b_bfrs_eval}")
print(f"    8B model (vanilla program): {vanilla_8b_base_eval}")
print(f"    8B model (bfrs program): {llama_8b_bfrs_eval}")
print(f"    8B model (finetuned program): {llama_8b_finetuned_eval}")
print(f"    8B model (finetuned bfrs program): {llama_8b_bfrs_finetuned_eval}")
print(f"    8B model (finetuned mipro program): {llama_8b_ft_mipro_eval}")
```


```python
raise NotImplementedError("Stop here")
```

# TODO - Serving

This is the second biggest unknown
I imagine it to be easy, but crazier things have happened

I need to keep a reference or link to the LLM forge job inside the LM.finetune method

how do I get the ray llm image!

We'll start by running the rayllm CLI command below to start the workflow to generate the service yaml configuration:
```bash
mkdir /home/ray/default/deploy/services
cd /home/ray/default/deploy/services
rayllm gen-config 
```

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/e2e-dspy-workflow/assets/cli.png" width=500 alt="todo! get this inage of what I need to serve">


<b style="background-color: yellow;">&nbsp;🛑 IMPORTANT&nbsp;</b>: Please `Terminate` your service from the Service page to avoid depleting your free trial credits.


```python
# Clean up
!python src/clear_cell_nums.py
!find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
!find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
!rm -rf __pycache__ data .HF_TOKEN deploy/services
```

# MARK: Ensemble - To Be Removed


```python
from dspy.teleprompt import Ensemble

# testing ensemble
top_programs = []

for folder, llama in all_llamas.items():
    with dspy.context(lm=llama):
        vanilla_program = IntentClassificationModule()
        bfrs_finetuned_program = bfrs_optimizer.compile(vanilla_program, trainset=optimizer_trainset, valset=optimizer_valset)
        devset_result = evaluate_devset(bfrs_finetuned_program)

        def wrapped_program(*args, **kwargs):
            with dspy.context(lm=llama):
                return bfrs_finetuned_program(*args, **kwargs)
        top_programs.append((wrapped_program, llama, devset_result))

top_3_devset = sorted(top_programs, key=lambda x: x[2], reverse=True)[:3]

teleprompter = Ensemble(reduce_fn=dspy.majority, size=None)
# print(top_3_devset)
programs = [p[0] for p in top_3_devset]
# print(programs)
compiled_program = teleprompter.compile(programs)
# # print(compiled_program)
eval_result = evaluate_devset(compiled_program)
eval_testset = evaluate_testset(compiled_program)
print(f"result for best_ensemble: {eval_result}, {eval_true_labels}, {eval_testset}")
ft_results[folder]["best_ensemble"] = {"devset": eval_result, "true_labels": eval_true_labels, "testset": eval_testset}
```

# MARK: Prediction no CoT - To Be Removed


```python
dspy.settings.configure(experimental=True)

from dspy.teleprompt.finetune_teleprompter import bootstrap_data, bootstrap_data_for_round, convert_to_module_level_message_data    

class IntentClassificationPredictModule(dspy.Module):
    def __init__(self):
        self.intent_classifier = dspy.Predict(IntentClassification)
        self.valid_labels = set(["activate_my_card", "cancel_transfer", "cash_withdrawal_charge", "declined_card_payment", "declined_cash_withdrawal", "direct_debit_payment_not_recognised", "extra_charge_on_statement", "pending_card_payment", "pending_top_up", "Refund_not_showing_up", "request_refund", "reverted_card_payment", "transaction_charged_twice", "transfer_fee_charged", "transfer_not_received_by_recipient", "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal"])

    def forward(self, text):
        prediction = self.intent_classifier(intent=text)
        sanitized_prediction = dspy.Prediction(label=prediction.label.lower().strip().replace(" ", "_"))
        # if sanitized_prediction.label not in self.valid_labels:
        #     for label in self.valid_labels:
        #         if label in sanitized_prediction.label:
        #             sanitized_prediction.label = label
        #             break
        #     # this means that the prediction was not in the valid labels
        #     # Could do edit distance or something more sophisticated here
        #     # but for now just take the first
        #     sanitized_prediction.label = self.valid_labels[0]
        return sanitized_prediction

def convert_examples_to_messages(examples):
    manual_traces = []
    module = IntentClassificationPredictModule()
    for example in examples:
        example["intent"] = example["text"]
        example = example.with_inputs("intent")
        manual_traces.append((module.intent_classifier, example.inputs(), example))

    data = []
    # traces are (pred, inputs, outputs)
    adapter = dspy.ChatAdapter()
    for pred, inputs, outputs in manual_traces:
        messages = adapter.format(pred.signature, [], inputs)
        formatted_completion = adapter.format_completion(pred.signature, outputs)
        messages.append({"role": "assistant", "content": formatted_completion})
        data.append({"messages": messages})

    return data

train_path = f"ft_trainset_data_banking_no_cot_{len(ft_trainset)}.jsonl"
eval_path = f"ft_valset_data_banking_no_cot_{len(devset)}.jsonl"

write_jsonl(train_path, convert_examples_to_messages(ft_trainset))
write_jsonl(eval_path, convert_examples_to_messages(devset))

```


```python
from dsp.modules.lm import TrainingMethod


method = TrainingMethod.SFT

kwargs = {
    "hyperparameters": {
        "num_devices": 4,
        "trainer_resources": None,
        "worker_resources": None,
        "generation_config": {
            "prompt_format": {
                "system": "<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>",
                "user": "<|start_header_id|>user<|end_header_id|>\n{instruction}<|eot_id|>",
                "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{instruction}<|eot_id|>",
                "trailing_assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
                "bos": "<|begin_of_text|>",
                "system_in_user": False,
                "default_system_message": ""
            },
        },
        "learning_rate": 3e-5,
        "num_epochs": 6,
        "train_batch_size_per_device": 32
    },
    "use_lora": True,
    # TODO: I think this needs to be set dynamically
    # "lora_dynamic_folder": "dspy/lora_weights/prodjob_qmulcjw4x8z599m8hkyja8tbmi/meta-llama/Llama-3.2-1B-Instruct"
}

SKIP_FT = False
if not SKIP_FT:
    # TODO: Get job working with LLMForge
    student_llama_1b = dspy.TrainableAnyscale(model="meta-llama/Llama-3.2-1B-Instruct")
    future = student_llama_1b.get_finetune(method, train_path, eval_path, **kwargs)
    checkpoint_names = future.result()

    model_names = checkpoint_names
```


```python
import json

if True:
    with open("model_names_predict.json", "r") as f:
        model_names = json.load(f)
else:
    with open("model_names_predict.json", "w") as f:
        json.dump(checkpoint_names, f)
    # model_names = checkpoint_names

```


```python
# Command for easy copying: 

llama_1b = dspy.LM(model="openai/meta-llama/Llama-3.2-1B-Instruct", **LOCAL_API_PARAMETERS, **MODEL_PARAMETERS)
finetuned_llamas_1b = {f: dspy.LM(model="openai/" + f, **LOCAL_API_PARAMETERS, **MODEL_PARAMETERS) for f in model_names}
all_llamas = {**finetuned_llamas_1b, "base": llama_1b}
```


```python
print([x.model for x in all_llamas.values()])

```


```python
# collected_data_filtered[0]
def collected_data_to_example(data):
    return dspy.Example(text=data["example"]["text"], label=data["prediction"]["label"]).with_inputs("text")

collected_data_examples = [collected_data_to_example(x) for x in collected_data_filtered]
# collected_data_examples[0]

devset_synthetic = collected_data_examples[:DEV_SIZE]
ft_optimizer_devset = collected_data_examples[DEV_SIZE:DEV_SIZE+OPTIMIZER_NUM_VAL]
ft_optimizer_trainset = collected_data_examples[DEV_SIZE+OPTIMIZER_NUM_VAL:]

evaluate_devset = Evaluate(devset=devset_synthetic, metric=metric, num_threads=NUM_THREADS, display_progress=True, max_errors=10000)

print(len(devset_synthetic), len(ft_optimizer_trainset), len(ft_optimizer_devset))
print(devset_synthetic[0])
```


```python
COMPILE_PROGRAM = True

dspy.settings.configure(experimental=True, lm=None)

ft_results = {}
for folder, llama in all_llamas.items():
    ft_results[folder] = {}
    vanilla_program = IntentClassificationModule()
    with dspy.context(lm=llama):
        devset_result = evaluate_devset(vanilla_program)
        ft_results[folder]["vanilla"] = {"devset": devset_result, "testset": None}

        if COMPILE_PROGRAM:
            bfrs_finetuned_program = bfrs_optimizer.compile(vanilla_program, trainset=ft_optimizer_trainset, valset=ft_optimizer_devset)
            bfrs_finetuned_program.save(f"simpleintent_predict_1b_32_ft_bfrs_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}_{folder.split('/')[-1]}.json")
        else:
            bfrs_finetuned_program = IntentClassificationModule()
            bfrs_finetuned_program.load(f"simpleintent_predict_1b_32_ft_bfrs_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}_{NUM_CANDIDATE_PROGRAMS}_{folder.split('/')[-1]}.json")
        
        llama_8b_bfrs_finetuned_eval = evaluate_devset(bfrs_finetuned_program)
        ft_results[folder]["bfrs"] = {"devset": llama_8b_bfrs_finetuned_eval, "true_labels": None, "testset": None}
        print(f"result for {folder}: {llama_8b_bfrs_finetuned_eval}, None, None")
```


```python
original_ft_results = json.load(open("ft_results.json", "r"))
combined_ft_results = {"cot": original_ft_results, "no_cot": ft_results}

combined_ft_results
```


```python
import json
import matplotlib.pyplot as plt
import numpy as np

# Prepare data for plotting
models = []
vanilla_devset = []
bfrs_devset = []
vanilla_testset = []
bfrs_testset = []

for model, results in ft_results.items():
    if model == "base":
        models.append("base")
    else:
        models.append("Epoch " + model.split(':')[1].split('-')[1])  # Extract epoch information
    vanilla_devset.append(results['vanilla']['devset'])
    bfrs_devset.append(results['bfrs']['devset'])
    vanilla_testset.append(results['vanilla']['testset'])
    bfrs_testset.append(results['bfrs']['testset'])

# Sort the data by epoch, keeping "base" at the beginning
sorted_data = sorted(zip(models, vanilla_devset, bfrs_devset, vanilla_testset, bfrs_testset),
                     key=lambda x: (x[0] != "base", x[0]))
models, vanilla_devset, bfrs_devset, vanilla_testset, bfrs_testset = zip(*sorted_data)

for i in range(len(models)):
    print(models[i], "vanilla_devset", vanilla_devset[i], "bfrs_devset", bfrs_devset[i], "vanilla_testset", vanilla_testset[i], "bfrs_testset", bfrs_testset[i])

# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Adjust bar positions and width
x = np.arange(len(models))
width = 0.35

# Plot bars for Dev Set (top graph)
ax1.bar(x - width/2, vanilla_devset, width, label='Vanilla Dev', color='skyblue')
ax1.bar(x + width/2, bfrs_devset, width, label='BFRS Dev', color='lightgreen')

# Add value labels on top of each bar for Dev Set
for i, v in enumerate(vanilla_devset):
    ax1.text(i - width/2, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
for i, v in enumerate(bfrs_devset):
    ax1.text(i + width/2, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

# Customize the Dev Set plot
ax1.set_ylabel('Dev Set Scores')
ax1.set_title('Model Performance Comparison Across Epochs (Dev Set)')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# Find the highest devset score and its corresponding model
highest_devset_score = max(bfrs_devset)
highest_score_model = models[bfrs_devset.index(highest_devset_score)]

print(highest_devset_score, highest_score_model, bfrs_devset[models.index(highest_score_model)])

# Prepare data for the bottom graph
# Prepare data for the bottom graph (Test Set)
base_vanilla_testset = vanilla_testset[models.index("base")]
base_bfrs_testset = bfrs_testset[models.index("base")]

best_model_index = bfrs_devset.index(highest_devset_score)
best_vanilla_testset = vanilla_testset[best_model_index]
best_bfrs_testset = bfrs_testset[best_model_index]

print("best_vanilla_testset", best_vanilla_testset, "best_bfrs_testset", best_bfrs_testset)
print("base_vanilla_testset", base_vanilla_testset, "base_bfrs_testset", base_bfrs_testset)

# Plot bars for Test Set (bottom graph)
models_to_plot = ["Base Model", f"Best Model ({highest_score_model})"]
x_test = np.arange(len(models_to_plot))

ax2.bar(x_test - width/2, [base_vanilla_testset, best_vanilla_testset], width, label='Vanilla Test', color='coral')
ax2.bar(x_test + width/2, [base_bfrs_testset, best_bfrs_testset], width, label='BFRS Test', color='lightseagreen')

# Add value labels on top of each bar for Test Set
for i, v in enumerate([base_vanilla_testset, best_vanilla_testset]):
    ax2.text(i - width/2, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
for i, v in enumerate([base_bfrs_testset, best_bfrs_testset]):
    ax2.text(i + width/2, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)

# Customize the Test Set plot
ax2.set_ylabel('Test Set Scores')
ax2.set_title('Model Performance Comparison (Test Set)')
ax2.set_xticks(x_test)
ax2.set_xticklabels(models_to_plot)
ax2.legend()
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

```


```python
# # {'cot': {'meta-llama/Llama-3.2-1B-Instruct:epochs-2-total-trained-steps-69': {'vanilla': {'devset': 26.2,
#     'testset': 28.4},
#    'bfrs': {'devset': 32.4, 'true_labels': None, 'testset': 32.4}},
#   'meta-llama/Llama-3.2-1B-Instruct:epochs-5-total-trained-steps-138': {'vanilla': {'devset': 28.8,
#     'testset': 32.3},
#    'bfrs': {'devset': 42.0, 'true_labels': None, 'testset': 43.1}},
#   'meta-llama/Llama-3.2-1B-Instruct:epochs-4-total-trained-steps-115': {'vanilla': {'devset': 28.2,
#     'testset': 32.5},
#    'bfrs': {'devset': 37.2, 'true_labels': None, 'testset': 38.6}},
#   'meta-llama/Llama-3.2-1B-Instruct:epochs-3-total-trained-steps-92': {'vanilla': {'devset': 30.2,
#     'testset': 34.1},
#    'bfrs': {'devset': 38.4, 'true_labels': None, 'testset': 37.5}},
#   'meta-llama/Llama-3.2-1B-Instruct:epochs-1-total-trained-steps-46': {'vanilla': {'devset': 15.6,
#     'testset': 14.6},
#    'bfrs': {'devset': 27.6, 'true_labels': None, 'testset': 29.0}},
#   'meta-llama/Llama-3.2-1B-Instruct:epochs-0-total-trained-steps-23': {'vanilla': {'devset': 22.6,
#     'testset': 20.4},
#    'bfrs': {'devset': 29.0, 'true_labels': None, 'testset': 32.1}},
#   'base': {'vanilla': {'devset': 1.4, 'testset': 1.9},
#    'bfrs': {'devset': 27.8, 'true_labels': None, 'testset': 28.9}}},
#  'no_cot': {'meta-llama/Llama-3.2-1B-Instruct:epochs-0-total-trained-steps-29': {'vanilla': {'devset': 21.4,
#     'testset': 22.2},
#    'bfrs': {'devset': 34.8, 'true_labels': None, 'testset': 37.9}},
#   'meta-llama/Llama-3.2-1B-Instruct:epochs-5-total-trained-steps-174': {'vanilla': {'devset': 43.8,
#     'testset': 45.1},
#    'bfrs': {'devset': 43.8, 'true_labels': None, 'testset': 45.1}},
#   'meta-llama/Llama-3.2-1B-Instruct:epochs-1-total-trained-steps-58': {'vanilla': {'devset': 14.6,
#     'testset': 16.3},
#    'bfrs': {'devset': 34.4, 'true_labels': None, 'testset': 36.2}},
#   'meta-llama/Llama-3.2-1B-Instruct:epochs-2-total-trained-steps-87': {'vanilla': {'devset': 37.8,
#     'testset': 41.8},
#    'bfrs': {'devset': 37.8, 'true_labels': None, 'testset': 41.8}},
#   'meta-llama/Llama-3.2-1B-Instruct:epochs-4-total-trained-steps-145': {'vanilla': {'devset': 45.0,
#     'testset': 46.3},
#    'bfrs': {'devset': 45.0, 'true_labels': None, 'testset': 46.3}},
#   'meta-llama/Llama-3.2-1B-Instruct:epochs-3-total-trained-steps-116': {'vanilla': {'devset': 45.2,
#     'testset': 45.9},
#    'bfrs': {'devset': 45.2, 'true_labels': None, 'testset': 45.9}},
#   'base': {'vanilla': {'devset': 9.4, 'testset': 8.6},
#    'bfrs': {'devset': 36.4, 'true_labels': None, 'testset': 38.8}}}}



import matplotlib.pyplot as plt
import numpy as np

def extract_data(results, key):
    return {
        model: {
            'vanilla': data['vanilla'][key],
            'bfrs': data['bfrs'][key]
        }
        for model, data in results.items()
    }

cot_devset = extract_data(combined_ft_results['cot'], 'devset')
no_cot_devset = extract_data(combined_ft_results['no_cot'], 'devset')
print(no_cot_devset)

# Prepare data for plotting
models = list(cot_devset.keys())
x = np.arange(len(models))
width = 0.2

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))

# Top graph
ax1.bar(x - 1.5*width, [cot_devset[m]['vanilla'] for m in models], width, label='CoT Vanilla')
ax1.bar(x - 0.5*width, [cot_devset[m]['bfrs'] for m in models], width, label='CoT BFRS')
ax1.bar(x + 0.5*width, [no_cot_devset[m]['vanilla'] for m in models], width, label='No CoT Vanilla')
ax1.bar(x + 1.5*width, [no_cot_devset[m]['bfrs'] for m in models], width, label='No CoT BFRS')

ax1.set_ylabel('Devset Score')
ax1.set_title('CoT vs No CoT: Vanilla and BFRS on Devset')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right')
ax1.legend()

# Find best models
best_cot = max(cot_devset.items(), key=lambda x: x[1]['bfrs'])[0]
best_no_cot = max(no_cot_devset.items(), key=lambda x: x[1]['bfrs'])[0]

# Bottom graph
models = ['Best CoT', 'Best No CoT']
x = np.arange(len(models))

cot_data = ft_results['cot'][best_cot]
no_cot_data = ft_results['no_cot'][best_no_cot]

ax2.bar(x - 1.5*width, [cot_data['vanilla']['devset'], no_cot_data['vanilla']['devset']], width, label='Vanilla Devset')
ax2.bar(x - 0.5*width, [cot_data['bfrs']['devset'], no_cot_data['bfrs']['devset']], width, label='BFRS Devset')
ax2.bar(x + 0.5*width, [cot_data['vanilla']['testset'], no_cot_data['vanilla']['testset']], width, label='Vanilla Testset')
ax2.bar(x + 1.5*width, [cot_data['bfrs']['testset'], no_cot_data['bfrs']['testset']], width, label='BFRS Testset')

ax2.set_ylabel('Score')
ax2.set_title('Best CoT vs Best No CoT: Devset and Testset Results')
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()

plt.tight_layout()
plt.show()

```


```python
# MARK: Mipro
# from dspy.teleprompt import MIPROv2

# eval_kwargs = dict(display_progress=True, display_table=0, num_threads=NUM_THREADS)
# teleprompter = MIPROv2(prompt_model=llama_70b, task_model=llama_70b, metric=metric, num_candidates=10, init_temperature=0.9, verbose=True)

# COMPILE_PROGRAM = True
# if COMPILE_PROGRAM:
#     with dspy.context(lm=llama_70b):
#         compiled_program = teleprompter.compile(vanilla_program, trainset=optimizer_trainset, valset=optimizer_valset, num_batches=30, max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS,max_labeled_demos=MAX_LABELED_DEMOS, eval_kwargs=eval_kwargs, requires_permission_to_run=False)
#         compiled_program.save(f"t2sql_70b_31_MIPROv2_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}.json")
# else:
#     compiled_program = TextToSQLModule()
#     compiled_program.load(f"t2sql_70b_31_MIPROv2_{MAX_BOOTSTRAPPED_DEMOS}_{MAX_LABELED_DEMOS}.json")

# with dspy.context(lm=llama_70b):
#     llama_70b_mipro_eval = evaluate_devset(compiled_program)
```
