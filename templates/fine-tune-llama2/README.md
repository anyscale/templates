# Fine-tuning Llama-2 models with Deepspeed, Accelerate, and Ray Train
This template shows you how to fine-tune Llama-2 models. 

## Step 1: Test the fine-tuning logic
Open a terminal in VS Code (ctl+shift+`). Choose the model size you want (7b, 13b, or 70b) and test the fine-tuning logic with dummy data, Grade School Math 8k (GSM8K) dataset.
```
./run_llama_ft.sh --size=7b --as-test
```

## Step 2: Add your own data
- Replace the contents in `./data/train.jsonl` with your own training data
- (Optional) Replace the contents in `./data/test.jsonl` with your own test data if any.
- (Optional) Add special token in `./data/tokens.json` if any.

## Step 3: Test the fine-tuning logic with your own data
Test the fine-tuning logic again with your own data.
```
./run_llama_ft.sh --size=7b --as-test
```

## Step 4: Kick off the fine-tuning
Start the full parameter fine-tuning for the model you want.
```
./run_llama_ft.sh --size=7b
```

Model checkpoints will be stored under `{user's first name}/ft_llms_with_deepspeed/` in the cloud stroage bucket created for your Anyscale account. The full path will be printed in the output after the training is completed.

------

## What's next.   

Vola! You have fine-tuned your own Llama-2 models. Want more than this? Check out advanced tutorials below 

- [Comprehensive walkthrough](./tutorials/walkthrough.md)
- [Fine-tune Llama-2 with LoRA adatpers](./tutorials/lora.md)