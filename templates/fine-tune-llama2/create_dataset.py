from datasets import load_dataset
import json
import os

# This script creates a fine-tuning dataset from the gsm8k dataset (https://huggingface.co/datasets/gsm8k).
# It also defines an additional tokens.json with tokens used to structure the data. The tokens should be treated as special tokens by the model. You can use this script as a starting point to create your own dataset.
dataset = load_dataset("gsm8k", "main")

dataset_splits = {"train": dataset["train"], "test": dataset["test"]}


def main():
    if not os.path.exists("data"):
        os.mkdir("data")

    with open("data/tokens.json", "w") as f:
        tokens = {}
        tokens["tokens"] = ["<START_Q>", "<END_Q>", "<START_A>", "<END_A>"]
        f.write(json.dumps(tokens))

    for key, ds in dataset_splits.items():
        with open(f"data/{key}.jsonl", "w") as f:
            for item in ds:
                newitem = {}
                newitem["input"] = (
                    f"<START_Q>{item['question']}<END_Q>"
                    f"<START_A>{item['answer']}<END_A>"
                )
                f.write(json.dumps(newitem) + "\n")


if __name__ == "__main__":
    main()
