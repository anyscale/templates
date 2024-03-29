import numpy as np
import pandas as pd
import os
import ray
from datasets import load_dataset
import ray.data
from transformers import AutoTokenizer
from ray.data.preprocessors import BatchMapper
from transformers import Trainer, TrainingArguments
from transformers import (
    GPTJForCausalLM,
    AutoTokenizer,
    default_data_collator,
)
from transformers.utils.logging import disable_progress_bar, enable_progress_bar
import torch

from ray import tune
from ray.air import session
from ray.train.huggingface import TransformersTrainer
from ray.air.config import ScalingConfig
from ray.air.config import RunConfig
from ray.data.preprocessors import Chain
import evaluate

#### Set up global variables.  We will use 16 workers, each being assigned 1 GPU and 8 CPUs.
model_name = 'EleutherAI/gpt-j-6b'
use_gpu = True
num_workers = 16
cpus_per_worker = 8



#---------------------EDIT AND UPDATE WITH YOUR DATASET HERE---------------------#
current_dataset = load_dataset("tiny_shakespeare")
ray_datasets = ray.data.from_huggingface(current_dataset)
# Replace this with your own training dataset loading
train_ds = ray_datasets["train"]
#  Replace this with your own evaluation dataset loading
eval_ds = ray_datasets["validation"]
#---------------------EDIT AND UPDATE WITH YOUR DATASET HERE---------------------#


# Because the dataset is represented by a single large string, we will need to do some preprocessing.
# For that, we will define two Ray AIR Preprocessors using the BatchMapper API, allowing us to define functions that will be applied on batches of data.
# The split_text function will take the single string and split it into separate lines, removing empty lines and character names ending with ‘:’ (eg. ‘ROMEO:’).
# The tokenize function will take the lines and tokenize them using the 🤗 Tokenizer associated with the model, ensuring each entry has the same length (block_size) by padding and truncating.
block_size = 512

def split_text(batch: pd.DataFrame) -> pd.DataFrame:
    text = list(batch["text"])
    flat_text = "".join(text)
    split_text = [
        x.strip()
        for x in flat_text.split("\n")
        if x.strip() and not x.strip()[-1] == ":"
    ]
    return pd.DataFrame(split_text, columns=["text"])


def tokenize(batch: pd.DataFrame) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    ret = tokenizer(
        list(batch["text"]),
        truncation=True,
        max_length=block_size,
        padding="max_length",
        return_tensors="np",
    )
    ret["labels"] = ret["input_ids"].copy()
    return dict(ret)


splitter = BatchMapper(split_text, batch_format="pandas")
tokenizer = BatchMapper(tokenize, batch_format="pandas")

# We can now configure Ray AIR's ray.train.huggingface.TransformersTrainer to perform distributed fine-tuning of the model.
# In order to do that, we specify a trainer_init_per_worker function, which creates a 🤗 Transformers Trainer that will be distributed by Ray using Distributed Data Parallelism (using PyTorch Distributed backend internally).
# This means that each worker will have its own copy of the model, but operate on different data, At the end of each step, all the workers will sync gradients.

# Because GPT-J is a relatively large model, it may not be possible to fit it on smaller GPU types (<=16 GB GRAM).
#  To deal with that issue, we can use DeepSpeed, a library to optimize the training process and allow us to (among other things) offload and partition optimizer and parameter states, reducing GRAM usage.
# Furthermore, DeepSpeed ZeRO Stage 3 allows us to load large models without running out of memory.



def trainer_init_per_worker(train_dataset, eval_dataset=None, **config):
    # Use the actual number of CPUs assigned by Ray
    os.environ["OMP_NUM_THREADS"] = str(
        session.get_trial_resources().bundles[-1].get("CPU", 1)
    )
    # Enable tf32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True

    batch_size = config.get("batch_size", 4)
    epochs = config.get("epochs", 2)
    warmup_steps = config.get("warmup_steps", 0)
    learning_rate = config.get("learning_rate", 0.00002)
    weight_decay = config.get("weight_decay", 0.01)

    deepspeed = {
        "fp16": {
            "enabled": "auto",
            "initial_scale_power": 8,
        },
        "bf16": {"enabled": "auto"},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
            },
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "gather_16bit_weights_on_model_save": True,
            "round_robin_gradients": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }

    print("Preparing training arguments")
    training_args = TrainingArguments(
        "output",
        per_device_train_batch_size=batch_size,
        logging_steps=1,
        save_strategy="no",
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        label_names=["input_ids", "attention_mask"],
        num_train_epochs=epochs,
        push_to_hub=False,
        disable_tqdm=True,  # declutter the output a little
        fp16=True,
        gradient_checkpointing=True,
        deepspeed=deepspeed,
    )
    disable_progress_bar()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model")

    model = GPTJForCausalLM.from_pretrained(model_name, use_cache=False)
    model.resize_token_embeddings(len(tokenizer))

    print("Model loaded")

    enable_progress_bar()

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    return trainer

# With our trainer_init_per_worker complete, we can now instantiate the ray.train.huggingface.TransformersTrainer.
# Aside from the function, we set the scaling_config, controlling the amount of workers and resources used, and the datasets we will use for training and evaluation.

# We pass the preprocessors we have defined earlier as an argument, wrapped in a ray.data.preprocessors.chain.Chain.
# The preprocessor will be included with the returned ray.air.checkpoint.Checkpoint, meaning it will also be applied during inference.

trainer = TransformersTrainer(
    trainer_init_per_worker=trainer_init_per_worker,
    trainer_init_config={
        "batch_size": 16,  # per device
        "epochs": 1,
    },
    scaling_config=ScalingConfig(
        num_workers=num_workers,
        use_gpu=use_gpu,
        resources_per_worker={"GPU": 1, "CPU": cpus_per_worker},
    ),
     run_config=RunConfig(
        storage_path="/mnt/user_storage",
        sync_config=tune.SyncConfig(syncer=None),
    ),

    datasets={"train": train_ds, "evaluation": eval_ds},
    preprocessor=Chain(splitter, tokenizer),
)

#Finally, we call the ray.train.huggingface.TransformersTrainer.fit method to start training with Ray AIR.
# We will save the ray.air.Result object to a variable so we can access metrics and checkpoints.
results = trainer.fit()
