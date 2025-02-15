# Fine-tuning Stable Diffusion XL with Ray Train

**⏱️ Time to complete**: 10 min

This template shows you how to do [Dreambooth](https://arxiv.org/abs/2208.12242) fine-tuning, which is a method of personalizing a stable diffusion model on a few examples (3~5) of a subject.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/fine-tune-stable-diffusion/assets/finetune-sample-results.png"/>

In this tutorial, you will learn about:
1. How to easily scale out an existing HuggingFace `diffusers` example to run on a Ray cluster with minimal modifications.
2. Basic features of [Ray Train](https://docs.ray.io/en/latest/train/train.html) such as specifying the number of training workers and the desired accelerator type.
3. Anyscale's smart instance selection and autoscaling that makes it simple to scale up your training workload to any size.

## Step 1: Install Python dependencies

The application requires a few extra Python dependencies. Install them using `pip`. When launching remote workers, Anyscale automatically installs the dependencies on them.


```python
!pip install -U accelerate==0.28.0 diffusers==0.27.2 peft==0.10.0 transformers==4.39.1
```

## Step 2: Set up a dataset of your subject

First, provide some images of the subject you want to fine-tune on.

This example uses a sample dog dataset to demonstrate, but you can use pictures of your own subject.
Fine-tuning works best if your images are all cropped to a square with your subject in the center.

A few notes on these constants that you can modify when training on your own custom subject:
* `SUBJECT_TOKEN` is the a unique token that you teach the model to correspond to your subject. This token can be any token that doesn't appear much in normal text.
    * Think of it as the name of your subject that the diffusion model learns to recognize. You can leave it as `sks`.
    * When generating images, make sure to include `sks` in your prompt--otherwise the model generates any random dog, not the dog that you fine-tuned it on.
* `SUBJECT_CLASS` is the category that your subject falls into.
    * For example, if you have a human subject, the class could be `"man"` or `"woman"`.
    * Use this class in combination with the `SUBJECT_TOKEN` in a prompt to convey the meaning: "a dog named sks".
* Put training images of your subject in `SUBJECT_IMAGES_PATH` to upload later to cloud storage so that all worker nodes can access the dataset.
    * The easiest way to use your own images is to drag files into a folder in the VS Code file explorer, then moving the folder to `SUBJECT_IMAGES_PATH` in the command line. For example, `mv ./images /mnt/local_storage/subject_images`.


```python
SUBJECT_TOKEN = "sks"
SUBJECT_CLASS = "dog"
SUBJECT_IMAGES_PATH = "/mnt/local_storage/subject_images"
```


```python
# Copy the sample dog dataset to the subject images path--feel free to comment this out.
!mkdir -p {SUBJECT_IMAGES_PATH} && cp ./assets/dog/*.jpeg {SUBJECT_IMAGES_PATH}
```

Take a look at the dataset.


```python
from IPython.display import Image, display
from pathlib import Path

display(*[Image(filename=image_path, width=250) for image_path in Path(SUBJECT_IMAGES_PATH).iterdir()])
```

Next, upload the dataset to cloud storage so that Anyscale can download it on each worker node at the start of training.


```python
import os
from utils import upload_to_cloud

DATA_CLOUD_PATH = os.environ["ANYSCALE_ARTIFACT_STORAGE"] + "/subject_images"
upload_to_cloud(local_path=SUBJECT_IMAGES_PATH, cloud_uri=DATA_CLOUD_PATH)
print("Uploaded data to: ", DATA_CLOUD_PATH)
```

Create some prompts to test our model on after fine-tuning. Notice that every prompt includes the `{SUBJECT_TOKEN} {SUBJECT_CLASS}`.

You can change these to be more applicable for your subject.


```python
PROMPTS = [
    f"{SUBJECT_TOKEN} {SUBJECT_CLASS} at the beach",
    f"{SUBJECT_TOKEN} {SUBJECT_CLASS} in a bucket",
    f"{SUBJECT_TOKEN} {SUBJECT_CLASS} sleeping soundly",
    f"{SUBJECT_TOKEN} {SUBJECT_CLASS} as a superhero",
]
PROMPTS
```

## Step 3: Run fine-tuning with Ray Train and Hugging Face Accelerate

Next, launch the distributed fine-tuning job.

Use the training script provided by the [Hugging Face diffusers Dreambooth fine-tuning example](https://github.com/huggingface/diffusers/blob/d7634cca87641897baf90f5a006f2d6d16eac6ec/examples/dreambooth/README_sdxl.md) with very slight modifications.

See `train_dreambooth_lora_sdxl.py` for the training script. The example does fine-tuning with [Low Rank Adaptation](https://arxiv.org/abs/2106.09685) (LoRA), which is a method that freezes most layers but injects a small set of trainable layers that get added to existing layers. This method greatly reduces the amount of training state in GPU memory and reduces the checkpoint size, while maintaining the fine-tuned model quality.

This script uses Hugging Face Accelerate, and this example shows that it's easy to scale out an existing training script on a Ray cluster with Ray Train.

### Parse training arguments

The original example launches the `diffusers` script at the command line. This example launches it with Ray Train instead and passes in the parsed command line arguments, in order to make as few modifications to the training script as possible.

The settings and hyperparameters below are taken from the [Hugging Face example](https://github.com/huggingface/diffusers/blob/d7634cca87641897baf90f5a006f2d6d16eac6ec/examples/dreambooth/README_sdxl.md).


```python
from train_dreambooth_lora_sdxl import parse_args

# [Optional] Setup wandb to visualize generated samples during fine-tuning.
# os.environ["WANDB_API_KEY"] = "YOUR_WANDB_API_KEY"

# See `parse_args` in train_dreambooth_lora_sdxl.py to see all the possible configurations.
cmd_line_args = [
    f"--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
    f"--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
    f"--instance_data_dir={SUBJECT_IMAGES_PATH}",
    "--output_dir=/mnt/local_storage/lora-trained-xl",
    "--mixed_precision=fp16",
    # A neutral prompt that serves as the caption for the subject image during training.
    f"--instance_prompt=a photo of {SUBJECT_TOKEN} {SUBJECT_CLASS}",
    "--resolution=1024",
    # The global batch size is: num_workers * train_batch_size * gradient_accumulation_steps
    # Define the number of workers later in the TorchTrainer.
    "--train_batch_size=1",  # This is the batch size *per* worker.
    "--gradient_accumulation_steps=1",
    "--learning_rate=1e-4",
    "--lr_scheduler=constant",
    "--lr_warmup_steps=0",
    "--max_train_steps=100",
    "--checkpointing_steps=100",
    # Use the first prompt as a sample to generate during training.
    "--seed=0",
] + (
    [
        f"--validation_prompt={PROMPTS[0]}",
        "--validation_epochs=25",
        "--report_to=wandb",
    ]
    if os.environ.get("WANDB_API_KEY")
    else []
)

TRAINING_ARGS = parse_args(input_args=cmd_line_args)
```

### Launch distributed training with Ray Train

To run distributed training, use a `ray.train.torch.TorchTrainer` to request GPU workers and connect them together in a distributed worker group. Then, when the workers run the training script, Hugging Face Accelerate detects this distributed process group and sets up the model to do data parallel training.

A few notes:
* `ray.init(runtime_env={"env_vars": ...})` sets the environment variables on all workers in the cluster -- setting the environment variable in this notebook on the head node is not enough in a distributed setting.
* `train_fn_per_worker` is the function that will run on all distributed training workers. In this case, it's just a light wrapper on top of the `diffusers` example script that copies the latest checkpoint to shared cluster storage.
* `ScalingConfig` is the configuration that determines how many workers and what kind of accelerator to use for training. Once the training is launched, **Anyscale will automatically scale up nodes to meet this resource request!**

The result of this fine-tuning is a fine-tuned LoRA model checkpoint at `MODEL_CHECKPOINT_PATH`.


```python
MODEL_CHECKPOINT_PATH = os.environ["ANYSCALE_ARTIFACT_STORAGE"] + "/checkpoint-final"

print("Final checkpoint will be uploaded to: ", MODEL_CHECKPOINT_PATH)
```


```python
import ray
import ray.train
from ray.train.torch import TorchTrainer

from train_dreambooth_lora_sdxl import main
from utils import (
    download_from_cloud,
    upload_to_cloud,
    get_a10g_or_equivalent_accelerator_type,
)


# Set environment variables across the entire cluster.
ENV_VARS = {"HF_HOME": "/mnt/local_storage/huggingface"}

WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
if WANDB_API_KEY:
    ENV_VARS["WANDB_API_KEY"] = WANDB_API_KEY

ray.shutdown()
ray.init(runtime_env={"env_vars": ENV_VARS})


def train_fn_per_worker(config: dict):
    download_from_cloud(cloud_uri=DATA_CLOUD_PATH, local_path=SUBJECT_IMAGES_PATH)

    # See train_dreambooth_lora_sdxl.py for all of the training details.
    final_checkpoint_path = main(config["args"])

    # Upload final checkpoint to cloud. (Only the rank 0 worker will return a path here.)
    if final_checkpoint_path is not None:
        upload_to_cloud(
            local_path=final_checkpoint_path, cloud_uri=MODEL_CHECKPOINT_PATH
        )
        print("Final checkpoint has been uploaded to: ", MODEL_CHECKPOINT_PATH)


trainer = TorchTrainer(
    train_fn_per_worker,
    # Pass command line arguments to the `config` dict of the `train_fn_per_worker`
    train_loop_config={"args": TRAINING_ARGS},
    scaling_config=ray.train.ScalingConfig(
        # Do data parallel training with GPU workers
        # Request A10G GPUs (or L4 GPUs if running on GCP)
        num_workers=4,
        use_gpu=True,
        accelerator_type=get_a10g_or_equivalent_accelerator_type(),
    ),
)

# Launch the training.
trainer.fit()
print("Finished fine-tuning!")
```

## Step 3: Generate some images with your fine-tuned model.

Finally, generate some images!

Launch 2 remote GPU tasks to generate images from the `PROMPTS` you defined earlier, one using just the base model and one that loads the fine-tuned LoRA weights. Compare them to see the results of fine-tuning.

Note: If Anyscale already scaled down your cluster from the training job due to the workers being idle, then this step might take a little longer to relaunch new GPU workers.


```python
import ray
from utils import generate

[base_model_images, finetuned_images] = ray.get(
    [
        generate.remote(prompts=PROMPTS, args=TRAINING_ARGS),
        generate.remote(
            prompts=PROMPTS,
            args=TRAINING_ARGS,
            model_checkpoint_path=MODEL_CHECKPOINT_PATH,
        ),
    ]
)
```

### Images generated with the finetuned model

These images should resemble your subject. If the generated image quality isn't satisfactory, see to the tips in [this blog post](https://huggingface.co/blog/dreambooth#tldr-recommended-settings) to tweak your hyperparameters.


```python
from IPython.display import display

display(*finetuned_images)
```

### Images generated with the base model for comparison


```python
# Uncomment below to show the images generated by the base model
# for a comparison of generate images before and after fine-tuning.

# display(*base_model_images)
```

## Summary

At this point, you've fine-tuned Stable Diffusion XL.

As a recap, this notebook:
1. Installed cluster-wide dependencies.
2. Scaled out fine-tuning to multiple GPU workers.
3. Compared the generated output results before and after fine-tuning.

As a next step, you can take the fine-tuned model checkpoint and use it to serve the model. See the tutorial on serving stable diffusion on the home page to get started.
