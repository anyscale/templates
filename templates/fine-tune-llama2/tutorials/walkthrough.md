# What is happening under the hood?

## Downloading the pre-trained checkpoint on to all GPU nodes. 

The pre-trained models for these models is quite large (12.8G for 7B model and 128G for 70B model). In order to make loading these models faster, we have mirrored the weights on to an AWS S3 bucket which can result in up 10GB/s download speed if the aws configs are setup correctly. 

## Cloud storage

Similarly the checkpoints during training can be quite large and we would like to be able to save those checkpoints to the familiar huggingface format so that we can serve it conveniently. The fine-tuning script in this template uses Ray Train Checkpointing to sync the checkpoints created by each node back to a centralized cloud storage on AWS S3. The final file structure for each checkpoint will have a look similar to the following structure:

```
aws s3 ls s3://<bucket_path>/checkpoint_00000

├── .is_checkpoint
├── .metadata.pkl
├── .tune_metadata
├── _metadata.meta.pkl
├── _preprocessor
├── _preprocessor.meta.pkl
├── added_tokens.json
├── config.json
├── generation_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer.model
└── tokenizer_config.json
```

After training we can use the [Llama-2 Serving template](#) to deploy our fine-tuned LLM by providing the checkpoint path stored on cloud directly.

## Creating the dataset

The main fine-tuning script is written in a general format that would require you to provide a `jsonl` file for train and test datasets in addition to a `json` file listing the special tokens used in your dataset. 

For example each row in your dataset might be formated like the following:

```
{"input": "<ASSISTANT>How can I help you?</ASSISTANT><USER>how is the weather?</USER>}
```

And the special tokens can be:

```
{"tokens": ["<ASSISTANT>", "</ASSISTANT>", "<USER>", "</USER>"]}
```

Depending on the dataset you want to fine-tune on, the tokenization and dataset pre-processing will likely need to be adjusted. The current code is configured to train on the Grade School Math 8k (GSM8K) dataset. By running the code below we create three files that are needed to launch the training script with. 

```
python create_dataset.py

>>> data/train.jsonl # 7.4k training data
>>> data/test.jsonl # 1.3k test data
>>> tokens.json # a list of special tokens
```

This dataset is trained with a context length of 512 which includes excessive padding to keep all samples limited to 512 tokens. This means that the training dataset has 3.5 M tokens.

## Launching fine-tuning

The script is written using Ray Train + Deepspeed integration via accelerate API. The script is general enough that it can be used to fine-tune all released sizes of Llama-2 models. 

The command for seeing all the options is:

```
python finetune_hf_llm.py --help
```

This script was tested across three model sizes on the following cluster configurations on Anyscale platform. 


| Model Size | Base HF Model ID             | Batch size per device | GPUs           | Time per epoch (min.) |
|------------|------------------------------|-----------------------|----------------|-----------------------|
| 7B         | `meta-llama/Llama-2-7b-hf`   | 16                    | 16x A10G (24G) | ~14 min.              |
| 13B        | `meta-llama/Llama-2-13b-hf`  | 16                    | 16x A10G (24G) | ~26 min.              |
| 70B        | `meta-llama/Llama-2-70b-hf`  | 8                     | 32x A10G (24G) | ~190 min.             |


To launch a full fine-tuning you can use the following command:

```
./run_llama_ft.sh --size=7b
```