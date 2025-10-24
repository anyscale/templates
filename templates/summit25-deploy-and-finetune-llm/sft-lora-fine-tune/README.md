# Supervised Fine-Tuning (SFT) at scale with LoRA

This guide provides a step-by-step workflow for supervised fine-tuning the [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) model on Anyscale clusters. You use LLaMA-Factory as the training framework and `LoRA`, a Parameter-Efficient Fine-Tuning (PEFT) method, to reduce memory requirements and enable efficient training.

SFT is a technique to adapt a pre-trained model to specific tasks. By showing the model high-quality examples of instructions and their desired outputs, you teach it to follow new instructions more accurately.

## Step 1: Set up your environment

### Dependencies
First, ensure your environment has the correct libraries. Start with a pre-built container image and install LLaMA-Factory on top of it.

Recommended container image:
```bash
anyscale/ray-llm:2.48.0-py311-cu128
```

Execute the following commands to install the required packages and optional tools for experiment tracking and faster model downloads:


```bash
%%bash
# Install the specific version of LLaMA-Factory
pip install -q llamafactory@git+https://github.com/hiyouga/LLaMA-Factory.git@v0.9.3

# (Optional) Install DeepSpeed for large-scale training
pip install -q deepspeed==0.16.9

# (Optional) For visualizing training metrics and logs
pip install -q tensorboard==2.20.0

# (Optional) For accelerated model downloads from Hugging Face
pip install -q hf_transfer==0.1.9
```

### Model and compute resources

This example uses a small model and a single GPU to demonstrate the workflow. However, this setup is designed to be highly scalable. You can easily adapt it for larger models by increasing the number of GPUs and enabling DeepSpeed.

| Item | Value |
|------|-------|
| **Base model** | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| **Worker Nodes** | 1 × L4 |

## Step 2: Prepare the dataset

### Understand the dataset
This tutorial uses [`glaive_toolcall_en_demo`](https://huggingface.co/datasets/zuol/glaive_toolcall_en_demo/tree/main), a dataset designed to teach models how to use tools (also known as function calling).

This dataset contains conversational examples where the model needs to interact with external tools. Each entry includes:
* `conversations`: A turn-by-turn log between a human and the gpt assistant.
* `tools`: A JSON schema describing the functions the model can call.

**Note**: The `conversations` may include special turns like function_call (the model deciding to call a tool) and observation (the result returned from the tool). This structure is ideal for teaching the model sophisticated tool-use behavior. To maintain role alignment in ShareGPT format, you must follow a strict turn order: `human` and `observation` (tool output) must appear in odd-numbered positions, while `gpt` and `function_call` must appear in even-numbered positions. The model learns to generate the content in the `gpt` and `function_call` turns.

**Dataset example**

```json
{
"conversations": [
    {
    "from": "human",
    "value": "Hi, I have some ingredients and I want to cook something. Can you help me find a recipe?"
    },
    {
    "from": "gpt",
    "value": "Of course! I can help you with that. Please tell me what ingredients you have."
    },
    {
    "from": "human",
    "value": "I have chicken, bell peppers, and rice."
    },
    {
    "from": "function_call",
    "value": "{\"name\": \"search_recipes\", \"arguments\": {\"ingredients\": [\"chicken\", \"bell peppers\", \"rice\"]}}"
    },
    {
    "from": "observation",
    "value": "{\"recipes\": [{\"name\": \"Chicken and Bell Pepper Stir Fry\", \"ingredients\": [\"chicken\", \"bell peppers\", \"rice\"], \"instructions\": \"Cut the chicken into small pieces. Slice the bell peppers. Cook the rice. Stir fry the chicken and bell peppers. Serve over rice.\"}, {\"name\": \"Chicken and Rice Casserole\", \"ingredients\": [\"chicken\", \"bell peppers\", \"rice\"], \"instructions\": \"Cook the chicken and rice separately. Mix them together with the bell peppers in a casserole dish. Bake until golden brown.\"}]}"
    },
    {
    "from": "gpt",
    "value": "I found two recipes for you. The first one is \"Chicken and Bell Pepper Stir Fry\". The instructions are: Cut the chicken into small pieces. Slice the bell peppers. Cook the rice. Stir fry the chicken and bell peppers. Serve over rice. The second one is \"Chicken and Rice Casserole\". The instructions are: Cook the chicken and rice separately. Mix them together with the bell peppers in a casserole dish. Bake until golden brown. Which one would you like to try?"
    },
    {
    "from": "human",
    "value": "They both sound delicious, but I think I'll try the stir fry. Can you order the ingredients for me?"
    },
    {
    "from": "gpt",
    "value": "I'm sorry, but as an AI, I don't have the capability to perform external tasks such as ordering ingredients. However, I can help you find more recipes or provide cooking tips if you need."
    }
],
"tools": "[{\"name\": \"search_recipes\", \"description\": \"Search for recipes based on ingredients\", \"parameters\": {\"type\": \"object\", \"properties\": {\"ingredients\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}, \"description\": \"The ingredients to search for\"}}, \"required\": [\"ingredients\"]}}]"
}
```

### Register the dataset

To specify new datasets that are accessible across Ray worker nodes, you must first add a **`dataset_info.json`** to **[storage shared across nodes](https://docs.anyscale.com/configuration/storage#shared)** such as `/mnt/cluster_storage`. This configuration file acts as a central registry for all your datasets. It maps a custom name to your dataset file location, format, and column structure. 

If you plan to run SFT fine-tuning on the `glaive_toolcall_en_demo` dataset, first complete the setup steps below. Ensure that you place the dataset files in a storage location that all workers can access (for example, a shared mount or object storage). Avoid storing large files on the head node.

`dataset_info.json`
```json
{
  "my_glaive_toolcall_en_demo": {
      "file_name": "/mnt/cluster_storage/glaive_toolcall_en_demo.json",
      "formatting": "sharegpt",
      "columns": {
          "messages": "conversations",
          "tools": "tools"
      }
  }
}
```

For a more detailed dataset preparation and formatting guide, see [Choose your data format](https://docs.anyscale.com/llm/fine-tuning/data-preparation#data-format).


```bash
%%bash
# Make sure all files are accessible to worker nodes
# Create a copy of the data in /mnt/cluster_storage
wget https://anyscale-public-materials.s3.us-west-2.amazonaws.com/llm-finetuning/llama-factory/datasets/sharegpt/glaive_toolcall_en_demo.json -O /mnt/cluster_storage/glaive_toolcall_en_demo.json
# Create a copy of the dataset registry in /mnt/cluster_storage
cp dataset_info.json /mnt/cluster_storage/
```

## Step 3: Create the fine-tuning config (SFT with LoRA)

Next, create the main YAML configuration file—the master recipe for your fine-tuning job. It specifies the base model, the fine-tuning method (LoRA), the dataset, training hyperparameters, cluster resources, and more.

**Important notes:**
- **Access and paths:** The YAML only needs to be on the **head node**, but any referenced paths (`dataset_dir`, `output_dir`) must reside on storage **reachable by all workers** (for example, `/mnt/cluster_storage/`).
- **Gated models:** If your base model has gated access (for example, Llama) on Hugging Face, set `HF_TOKEN` in the runtime environment.
- **GPU selection:** The config sets `accelerator_type` to `L4`, but you can switch to other GPUs depending on your cloud availability.

### Configure LLaMA-Factory with Ray

**Note**: To customize the training configuration, edit `sft_lora.yaml`.

```yaml
# sft_lora.yaml

### model
model_name_or_path: Qwen/Qwen2.5-0.5B-Instruct
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: all

### deepspeed
# deepspeed: /mnt/cluster_storage/ds_z3_config.json # Enable for larger models

### dataset
dataset: my_glaive_toolcall_en_demo
dataset_dir: /mnt/cluster_storage

template: qwen
cutoff_len: 1024
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 16

### output
output_dir: qwen2.5_0.5b_lora_sft
logging_steps: 5
save_steps: 10          # For tensorboard logging purpose too. Can increase if not using tensorboard
plot_loss: true
report_to: tensorboard # or none

### train
per_device_train_batch_size: 2 # Adjust this depending on your GPU memory and sequence length
gradient_accumulation_steps: 4
num_train_epochs: 2.0
learning_rate: 1.0e-4
bf16: true
lr_scheduler_type: cosine
warmup_ratio: 0.1
ddp_timeout: 180000000

### ray
ray_run_name: qwen2.5_0.5b_lora_sft
ray_storage_path: /mnt/cluster_storage/
ray_num_workers: 1  # Number of GPUs to use
resources_per_worker:
  GPU: 1
  accelerator_type:L4: 0.001            # Use this to simply specify a GPU type (not guaranteed on the same node).
  # anyscale/accelerator_shape:1xL4: 0.001  # Use this to specify a specific node shape.
  # See https://docs.ray.io/en/master/ray-core/accelerator-types.html#accelerator-types for a full list of accelerator types.
ray_init_kwargs:
  runtime_env:
    env_vars:
      # If using gated models like meta-llama/Llama-3.1-8B-Instruct
      # HF_TOKEN: <your_huggingface_token>
      # If hf_transfer is installed
      HF_HUB_ENABLE_HF_TRANSFER: '1'
```

### (Optional) Scaling up with DeepSpeed

While this example runs on a single GPU, the workflow is designed to scale to much larger models using multiple GPUs. For models that don't fit on a single GPU, you can use DeepSpeed, an optimization library that distributes the model and training computation across multiple GPUs. Higher ZeRO stages (1→3) and enabling CPU offload reduce GPU VRAM usage, but might cause slower training.

To enable DeepSpeed, create a separate Deepspeed config in the **[storage shared across nodes](https://docs.anyscale.com/configuration/storage#shared)**. and reference it from your main training yaml config with:

```yaml
deepspeed: /mnt/cluster_storage/ds_z3_config.json
```

Below is a sample ZeRO-3 config:

`ds_z3_config.json`
```json
{
"train_batch_size": "auto",
"train_micro_batch_size_per_gpu": "auto",
"gradient_accumulation_steps": "auto",
"gradient_clipping": "auto",
"zero_allow_untested_optimizer": true,
"fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
},
"bf16": {
    "enabled": "auto"
},
"zero_optimization": {
    "stage": 3,
    "overlap_comm": false,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
}
}
```

For even greater memory savings, you can combine ZeRO Stage 2 or 3 with *CPU offload*, which moves the partitioned states out of GPU VRAM and into CPU RAM. While this allows you to train massive models on limited hardware, it increases I/O and significantly slows down training. See `deepspeed-configs/ds_z3_offload_config.json` for reference.

For a more detailed guide on acceleration and optimization methods including DeepSpeed on Ray, see [Speed and memory optimizations](https://docs.anyscale.com/llm/fine-tuning/speed-and-memory-optimizations).


```bash
%%bash
# Create a copy of the DeepSpeed configuration file in /mnt/cluster_storage
cp deepspeed-configs/ds_z3_config.json /mnt/cluster_storage/
```

## Step 4: Train and monitor

With all configurations in place, you can launch fine-tuning in one of two ways:

### Option A: Run from a workspace (quick start)

The `USE_RAY=1` prefix tells LLaMA-Factory to run in distributed mode on the Ray cluster attached to your workspace.


```bash
%%bash
USE_RAY=1 llamafactory-cli train sft_lora.yaml
```

### Option B: Run as an Anyscale job (production)

For longer or production runs, submit the training as an **Anyscale job**. Jobs run outside your interactive session for better stability, retries, and durable logs. You package LLaMA-Factory and other libraries in a container image and launch with a short job config. See [Run LLaMA-Factory as an Anyscale job](https://docs.anyscale.com/llm/fine-tuning/llamafactory-jobs) for the step-by-step guide.

### Tracking with TensorBoard
If you enabled TensorBoard logging (`report_to: tensorboard` in your YAML), you can watch metrics (for example, training loss) update live and compare multiple runs with the same run name side-by-side.

- **While the job is running:** LLaMA-Factory prints a ready-to-run command that starts with `tensorboard --logdir`. Open a new terminal and run it. For example:
  ```bash
  tensorboard --logdir /tmp/ray/session_*/artifacts/*/qwen2.5_0.5b_lora_sft/driver_artifacts
  ```

- **After the job:** Point TensorBoard at `{ray_storage_path}/{ray_run_name}/`. Each `TorchTrainer_*` subfolder holds event files for a single run. Using the parent folder aggregates all runs for easy comparison.
  ```bash
  tensorboard --logdir /mnt/cluster_storage/qwen2.5_0.5b_lora_sft
  ```

In your Anyscale workspace, look for the open **port 6006** labeled **TensorBoard** to view the dashboards.

![Anyscale workspace showing open ports with TensorBoard on port 6006](https://anyscale-public-materials.s3.us-west-2.amazonaws.com/llm-finetuning/llama-factory/open-ports.png)

**TensorBoard example**

![TensorBoard](https://anyscale-public-materials.s3.us-west-2.amazonaws.com/llm-finetuning/llama-factory/3.2.1/ray-summit-tensorboard.png)

For a more detailed guide on tracking experiments with other tools such as Weights & Biases or MLflow, see [Observability and tracking](https://docs.anyscale.com/llm/fine-tuning/observability-and-tracking).

## Step 5: Locate checkpoints

Ray Train writes checkpoints under `ray_storage_path/ray_run_name`. In this example run, the path is: `/mnt/cluster_storage/qwen2.5_0.5b_lora_sft`.

Inside, you see a **trainer session** directory named like:
`TorchTrainer_8c6a5_00000_0_2025-09-09_09-53-45/`.

- Ray Train creates `TorchTrainer_*` **when the trainer starts**; the suffix encodes a short run ID and the **start timestamp**.
- Within that directory, Ray Train names checkpoints `checkpoint_000xxx/`, where the number is the saved ordered checkpoints.

Control the save cadence with `save_strategy` and `save_steps`. For instructions on how to resume interrupted training with `resume_from_checkpoint` and more, see [Understand the artifacts directory](https://docs.anyscale.com/llm/fine-tuning/checkpointing#artifacts-directory).

## Step 6: Export the model

If you use LoRA, you can keep the base model and adapters separate for [multi-LoRA deployment](https://docs.anyscale.com/llm/serving/multi-lora) or [merge the adapters](https://docs.anyscale.com/llm/fine-tuning/checkpointing#merge-lora) into the base model for low-latency inference. 

For full fine-tuning or freeze-tuning, export the fine-tuned model directly.

You may optionally apply [post-training quantization](https://docs.anyscale.com/llm/fine-tuning/checkpointing#ptq) on merged or full models before serving.
