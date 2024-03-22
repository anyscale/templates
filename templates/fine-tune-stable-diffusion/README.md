# Fine-tuning Stable Diffusion XL with Ray Train

⏱️ Time to complete: 15 min

This template shows you how to do [Dreambooth](https://arxiv.org/abs/2208.12242) fine-tuning, which is a method of personalizing a stable diffusion model on a few examples (3~5) of a subject.

In this tutorial, you will learn about:
1. How to easily scale out an existing HuggingFace `diffusers` example to run on a Ray cluster with minimal modifications.
2. Basic features of [Ray Train](https://docs.ray.io/en/latest/train/train.html) such as specifying the number of training workers and the desired accelerator type.
3. Anyscale's smart instance selection and autoscaling that makes it simple to scale up your training workload to any size.

## Step 1: Install python dependencies

The application requires a few extra Python dependencies. Install them using `pip` and they'll be automatically installed on remote workers when they're launched!


```python
# !pip install -U accelerate==0.28.0 diffusers==0.27.2 peft==0.10.0 transformers==4.39.1
```

## Step 2: Set up a dataset of your subject

First, provide some pictures of your subject.

We'll use a sample dog dataset to demonstrate, but you can populate `SUBJECT_IMAGES_DIR` with pictures of your own subject.
Fine-tuning works best if your images are all cropped to a square with your subject in the center!

A few notes on these constants that you can modify when training on your own custom subject:
* `SUBJECT_TOKEN` is the a unique token that you will teach the model to correspond to your subject. This can be is any token that does not appear much in normal text.
    * Think of it as the name of your subject that the diffusion model will learn to recognize. Feel free to leave it as `sks`.
    * When generating images, make sure to include `sks` in your prompt -- otherwise the model will just generate any random dog, not the dog that we fine-tuned it on!
* `SUBJECT_CLASS` is the category that your subject falls into.
    * For example, if you have a human subject, the class could be `"man"` or `"woman"`.
    * This class combined with the `SUBJECT_TOKEN` can be used in a prompt to convey the meaning: "a dog named sks".
* `SUBJECT_IMAGES_DIR` contains the training data of our subject used for fine-tuning.
    * **This should stay in `/mnt/cluster_storage` so that all distributed workers can access the data!**


```python
SUBJECT_TOKEN = "sks"
SUBJECT_CLASS = "dog"
SUBJECT_IMAGES_DIR = "/mnt/cluster_storage/subject_images"
```


```python
# Download the sample dog dataset -- feel free to comment this out.
from huggingface_hub import snapshot_download

snapshot_download(
    "diffusers/dog-example",
    local_dir=SUBJECT_IMAGES_DIR, repo_type="dataset",
    ignore_patterns=".gitattributes",
)
```

Take a look at the dataset!


```python
from IPython.display import Image, display
from pathlib import Path

display(*[Image(filename=image_path, width=250) for image_path in Path(SUBJECT_IMAGES_DIR).iterdir()])
```

Let's come up with some prompts to test our model on after fine-tuning. Notice the `{SUBJECT_TOKEN} {SUBJECT_CLASS}` included in each of them.

You can change these to be more fitting for your subject.


```python
PROMPTS = [
    f"{SUBJECT_TOKEN} {SUBJECT_CLASS} at the beach",
    f"{SUBJECT_TOKEN} {SUBJECT_CLASS} in a bucket",
    f"{SUBJECT_TOKEN} {SUBJECT_CLASS} sleeping soundly",
    f"{SUBJECT_TOKEN} {SUBJECT_CLASS} as a superhero",
]
PROMPTS
```

## Step 3: Run fine-tuning with Ray Train + HuggingFace Accelerate

Next, let's launch the distributed fine-tuning job.

We will use the training script provided by the [HuggingFace diffusers Dreambooth fine-tuning example](https://github.com/huggingface/diffusers/blob/d7634cca87641897baf90f5a006f2d6d16eac6ec/examples/dreambooth/README_sdxl.md) with very slight modifications.

See `train_dreambooth_lora_sdxl.py` for the training script. The example does fine-tuning with [Low Rank Adaptation](https://arxiv.org/abs/2106.09685) (LoRA), which is a method that freezes most layers but injects a small set of trainable layers that get added to existing layers. This method greatly reduces the amount of training state in GPU memory and reduces the checkpoint size, while maintaining the fine-tuned model quality.

This script is built on HuggingFace Accelerate, and we will show how easy it is to run an existing training script on a Ray cluster with Ray Train.

### Parse training arguments

The `diffusers` script is originally launched via the command line. Here, we'll launch it with Ray Train instead and pass in the parsed command line arguments, in order to make as few modifications to the training script as possible.


```python
import os
from train_dreambooth_lora_sdxl import parse_args

# [Optional] Setup wandb to visualize generated samples during fine-tuning.
# os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY"
os.environ["WANDB_API_KEY"] = "afee4ae3e5b07d9f76117a8ad9c62e930cd7a63d"

# See `parse_args` in train_dreambooth_lora_sdxl.py to see all the possible configurations.
cmd_line_args = [
    f"--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
    f"--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
    f"--instance_data_dir={SUBJECT_IMAGES_DIR}",
    "--output_dir=/mnt/local_storage/lora-trained-xl",
    "--mixed_precision=fp16",
    # A neutral prompt that serves as the caption for the subject image during training.
    f"--instance_prompt=a photo of {SUBJECT_TOKEN} {SUBJECT_CLASS}",
    "--resolution=1024",
    "--train_batch_size=1",
    "--gradient_accumulation_steps=1",
    "--learning_rate=1e-4",
    "--lr_scheduler=constant",
    "--lr_warmup_steps=0",
    "--max_train_steps=100",
    "--checkpointing_steps=100",
    # Use the first prompt as a sample to generate during training.
    f"--validation_prompt={PROMPTS[0]}",
    "--validation_epochs=25",
    "--seed=0",
] + (["--report_to=wandb"] if os.environ.get("WANDB_API_KEY") else [])

TRAINING_ARGS = parse_args(input_args=cmd_line_args)
```

### Launch distributed training with Ray Train

To run distributed training, we'll use a `ray.train.torch.TorchTrainer` to request GPU workers and connect them together in a distributed worker group. Then, when the workers run the training script, HuggingFace Accelerate detects this distributed process group and sets up the model to do data parallel training.

A few notes:
* `ray.init(runtime_env={"env_vars": ...})` sets the environment variables on all workers in the cluster -- setting the environment variable in this notebook on the head node is not enough in a distributed setting.
* `train_fn_per_worker` is the function that will run on all distributed training workers. In this case, it's just a light wrapper on top of the `diffusers` example script that copies the latest checkpoint to shared cluster storage.
* `ScalingConfig` is the configuration that determines how many workers and what kind of accelerator to use for training. Once the training is launched, **Anyscale will automatically scale up nodes to meet this resource request!**

The result of this fine-tuning will be a fine-tuned LoRA model checkpoint at `MODEL_CHECKPOINT_PATH`.


```python
MODEL_CHECKPOINT_PATH = "/mnt/cluster_storage/checkpoint-final"
```


```python
import os
import shutil

import ray.train
from ray.train.torch import TorchTrainer

from train_dreambooth_lora_sdxl import main


# Set the HuggingFace model cache to a shared location
# so that model loading time is faster after the first time.
os.environ["HF_HOME"] = "/mnt/cluster_storage/hf_cache"

# Set environment variables across the entire cluster.
ray.init(
    runtime_env={
        "env_vars": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "WANDB_API_KEY": os.environ.get("WANDB_API_KEY"),
        }
    },
    ignore_reinit_error=True,
)


def train_fn_per_worker(config: dict):
    # See train_dreambooth_lora_sdxl.py for all of the training details.
    final_checkpoint_path = main(config["args"])

    if final_checkpoint_path is not None:
        destination_path = config["model_checkpoint_path"]
        shutil.copytree(final_checkpoint_path, destination_path, dirs_exist_ok=True)
        print(f"Copied the checkpoint to {destination_path} for later use!")


trainer = TorchTrainer(
    train_fn_per_worker,
    train_loop_config={
        # Pass command line arguments from the driver to the `config` dict of the `train_fn_per_worker`
        "args": TRAINING_ARGS,
        # This is where we can access the fine-tuned model checkpoint later.
        "model_checkpoint_path": MODEL_CHECKPOINT_PATH,
    },
    scaling_config=ray.train.ScalingConfig(
        # Do data parallel training with A10G GPU workers
        num_workers=4, use_gpu=True, accelerator_type="A10G"
    ),
)

```


```python
# Launch the training.
trainer.fit()
```

## Step 3: Generate some images with your fine-tuned model!

Finally, let's generate some images!

We'll launch 2 remote GPU tasks to generate images from the `PROMPTS` we defined earlier, one using just the base model and one that loads our fine-tuned LoRA weights. Let's compare them to see the results of fine-tuning!

Note: If your cluster has already scaled down from the training job due to the workers being idle, then this step might take a little longer to relaunch new GPU workers.


```python
import ray
from utils import generate

[base_model_images, finetuned_images] = ray.get([
    generate.remote(prompts=PROMPTS, args=TRAINING_ARGS),
    generate.remote(
        prompts=PROMPTS, args=TRAINING_ARGS, model_checkpoint_path=MODEL_CHECKPOINT_PATH
    )
])
```


```python
print("\n".join(base_model_images))
display(*[Image(filename=image_path, width=250) for image_path in base_model_images])
```


```python
print("\n".join(finetuned_images))
display(*[Image(filename=image_path, width=250) for image_path in finetuned_images])
```

## Summary

Congrats, you've fine-tuned Stable Diffusion XL!

As a recap, this notebook:
1. Installed cluster-wide dependencies.
2. Scaled out fine-tuning to many GPU workers.
3. Compared the generated output results before and after fine-tuning.

As a next step, you can take the fine-tuned model checkpoint and use it to serve the model. See the tutorial on serving stable diffusion on the home page to get started!


```python

```
