from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
from torch.utils.flop_counter import FlopCounterMode
from llmforge.loss import LMLoss
from llmforge.file_transfer import ModelDownloader
from peft.tuners.lora.layer import Embedding as LoRAEmbeddingLayer


from argparse import ArgumentParser
args = ArgumentParser()
args.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
args.add_argument("--lora", default=False, action="store_true")
args.add_argument("--batch_size", type=int, default=2)
args.add_argument("--device", type=str, default="cuda")
args.add_argument("--seq_len", type=int, default=2048)
args.add_argument("--gradient_checkpointing", default=False, action="store_true")
args.add_argument("--dataset_path", type=str, default="alpaca_long_subset.jsonl")
args = args.parse_args()

# load model in bf16
downloader = ModelDownloader(
    model_id=args.model,
)
pretrained_path = downloader.download()
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=pretrained_path, torch_dtype=torch.bfloat16
)
dataset = datasets.load_dataset("json", data_files=args.dataset_path)
dataset = dataset["train"]

tokenizer = AutoTokenizer.from_pretrained(args.model, truncate = True, padding = "max_length")

if args.lora:
    import peft
    lora_config = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "modules_to_save": [],
    }
    lora_config = peft.LoraConfig(**lora_config)
    model = peft.get_peft_model(model, lora_config)

if args.gradient_checkpointing:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if not isinstance(model.get_input_embeddings(), LoRAEmbeddingLayer):
        # If LoRA is applied to neither the embeddings nor the head, we need to enable input requires_grads
        # for gradient checkpointing to work.
        # This is needed because torch requires at least one input and output to a checkpointed parameter to have requires_grad=True.
        # See https://pytorch.org/docs/stable/checkpoint.html
        model.enable_input_require_grads()
model = model.to(args.device)
model.train()


flop_counter = FlopCounterMode(mods=model, display=False, depth=None)

i = 0
batch = {"input_ids": [], "attention_mask": []}
for message in dataset["messages"]:
    batch_tokens = tokenizer(message[0]["content"] + message[1]["content"] , return_tensors="pt")
    batch["input_ids"].append(batch_tokens["input_ids"][:, :args.seq_len])
    batch["attention_mask"].append(batch_tokens["attention_mask"][:, :args.seq_len])
    if i == args.batch_size - 1:
        break
    i += 1

batch["input_ids"] = torch.cat(batch["input_ids"]).to(model.device)
batch["attention_mask"] = torch.cat(batch["attention_mask"]).to(model.device)
batch["labels"] = batch["input_ids"].clone()

with flop_counter:
    output = model(**batch)
    loss = LMLoss()(output, batch)
    loss.backward()
total_flops =  flop_counter.get_total_flops()
model.zero_grad(set_to_none=True)  # More efficient than zero_grad()

del output
del batch

print(f"{args.model} TFLOP: {total_flops / 1e12}")
print("Divide by time per step to get TFLOPs")
