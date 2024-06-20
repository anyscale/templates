# Fine-tuning on datasets with long context
**⏱️ Time to complete**: 5 minutes

This guide demonstrates how to prepare a dataset that results in particularely long context lengths. Make sure you have gone over the [basic fine-tuning guide](../../README.md) before going over this cookbook.

In the following example, we filter a hugginface dataset and assemble it into the right format.
You case use this as a template for creating your own datasets.


```python
import json
import pandas as pd
from transformers import AutoTokenizer

# Fill in your personal hugginface token with access to the tokenizer (You can use a similar tokenizer as a work-around)
HHUGGINFACE_TOKEN = ...
# The name of the model you want to fine-tune with. We use this only for tokenization so models with the same tokenizer are interoperable here.
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# Your fine-tuning context length
MAX_CONTEXT_LENGTH = 16384
# Depending on your model, tokenized messages will have special tokens such as a "beginning of sequence" or "system message" token added.
# The size of this "safety buffer" should be larger than what you expect these additional tokens to be in sum.
# 500 is a conservative size for a single-turn user-assistant conversation.
SAFETY_BUFFER = 500
# Design this to fit your dataset. This will help your model learn and converge to a better solution.
SYSTEM_MESSAGE = "You are an expert for patent law who generates abstracts from patents. Base your answer solely on the provided patent."
# Construct these dataframe depending on where you get your dataset from
TRAIN_DF = pd.read_csv("hf://datasets/Trelis/big_patent_60k_characters/train.csv")
TEST_DF = pd.read_csv("hf://datasets/Trelis/big_patent_60k_characters/test.csv")

# Fit this to how you want to construct your messages from your dataset. Pay attention to the names of columns from the dataset here.
def to_messages_dict(d: dict):
    """Assembles a single example of the dataset for fine-tuning."""
    return {"messages": [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": d["description"]},  {"role": "assistant", "content": d["abstract"]}]}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HHUGGINFACE_TOKEN)

def is_too_long(messages: dict):
    """Filters out rows that exceed MAX_CONTEXT_LENGTH in their total length"""
    return sum([len(tokenizer(m["content"])["input_ids"]) for m in messages["messages"]]) + SAFETY_BUFFER > MAX_CONTEXT_LENGTH

for frame, output_file in [(TRAIN_DF, "train.jsonl"), (TEST_DF, "test.jsonl")]:
    with open(output_file, 'w') as f:
        for _, row in frame.iterrows():
            messages = to_messages_dict(row.to_dict())
            if not is_too_long(messages):
                json_str = json.dumps(messages)
                f.write(json_str + '\n')
```

We can now use this dataset to fine-tune an LLM that helps us with creating abstracts from patents.
The fine-tuned model will have a context-length of 16384 tokens when fine-tuned on Anyscale Endpoints or the Anyscale platform.


## FAQ:

### How do I find out how many tokens the examples in my dataset have?

If you fine-tune with this template, you will find exact statistics at the beginning of your fine-tuning job.
Exact numbers are hard to compute in advance. To get a rough idea, you can instantiate a tokenizer for your model and feed it a few samples from your dataset. Remember that one example consists of system message, a prompt, and an answer. There may also be [online services](https://belladoreai.github.io/llama3-tokenizer-js/example-demo/build/) that can help.

### What if my dataset results in examples that are longer that the native context length of the model?

Some datasets have examples that, when tokenize, exceed the context length of your LLM.
Anyscale Endpoints and the Anyscale platform support extending the native context length of LLMs. For example, for Llama 3 8B, we support fine-tuning with up to 32768 tokens (see [Anyscale Endpoints](https://docs.anyscale.com/canary/endpoints/fine-tuning/supported-models/) docs for a list).
You ca read more aobut model quality considerations in [our blog](https://www.anyscale.com/blog/fine-tuning-llms-for-longer-context-and-better-rag-systems).


