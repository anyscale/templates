# Bring your own weights 
**⏱️ Time to complete**: 10 minutes

This guide focuses on how you can bring weights of a model similar in architecture to the Llama or Mistral family of models to fine-tune on the Anyscale Platform. Specifically, we will fine-tune the [Meta Llama Guard 2 model](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-guard-2/). Make sure you have gone over the [basic fine-tuning guide](../../README.md) before going over this cookbook. 


# Table of Contents
1. [Model weights stored in remote storage](#model-weights-stored-in-remote-storage)
    - [Public models](#public-model-weights)
    - [Private models](#private-model-weights)
2. [Model weights stored locally](#model-weights-stored-locally)
3. [Specifying the right model ID and prompt format](#specifying-the-right-model-ID-and-prompt-format)

## Example YAML

Along with the training and validation file paths, we specify the model that is similar in architecture `model_id` and point to the location of the model weights in an S3 bucket `initial_base_model_ckpt_path`. Additionally, given that this model could have a different prompt format, we add it to the configuration YAML as `prompt_format `.

```yaml
model_id: meta-llama/Meta-Llama-3-8B
initial_base_model_ckpt_path: s3://my-bucket/llama-guard-2
train_path: s3://air-example-data/gsm8k/train.jsonl # <-- change this to the path to your training data
valid_path: s3://air-example-data/gsm8k/test.jsonl # <-- change this to the path to your validation data. This is optional
generation_config:
  prompt_format:
context_length: 512
num_devices: 16 
num_epochs: 10
train_batch_size_per_device: 8
eval_batch_size_per_device: 16
learning_rate: 5e-6
padding: "longest" 
num_checkpoints_to_keep: 1
dataset_size_scaling_factor: 10000
output_dir: /mnt/local_storage
deepspeed:
  config_path: deepspeed_configs/zero_3_offload_optim+param.json
dataset_size_scaling_factor: 10000 
flash_attention_2: true
trainer_resources:
  memory: 53687091200
worker_resources:
  accelerator_type:A10G: 0.001
```

# Model Weights stored in remote storage 

## Public models
For models configured for public access, you simply need to add the URI of the location of the model weights in your training YAML. We support loading from data stored on S3 and GCS.


## Private models
For private models, you have two options: 

### Option 1: Configure permissions directly in your cloud account
The most convenient option is to provide read permissions for your Anyscale workspace to the specific bucket holding your model weights. You can follow our guide to do so [here](https://docs.anyscale.com/configuration/cloud-storage-buckets#access-private-cloud-storage).


### Option 2: Sync data into default cloud storage provided by Anyscale
The other option you have is to sync your mdoel weights into Anyscale-provided storage and then continue with fine-tuning. Let's consider private models in AWS S3. First, we'll need to configure your workspace to be able to access these model weights. We recommend that you simply export relevant environment variables directly (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, etc) into your current terminal session. Once that is done, you can copy over your model weight into 
the [default object storage bucket](https://docs.anyscale.com/platform/workspaces/workspaces-storage#object-storage-s3-or-gcs-buckets) provided by Anyscale (`$ANYSCALE_ARTIFACT_STORAGE`). That way, across runs/ workspace restarts, you don't have to repeat this process (compared to just downloading the files into your workspace).
1. First, download the model weights into your workspace:  
    ```bash
    aws s3 sync s3://<bucket_name>/<path_to_model_weights>/ mymodels/
    ```
2. The default object storage bucket configured for you in your workspace uses Anyscale-managed credentials internally. It is recommended to reset the credentials you provided so as to not interfere with the Anyscale-managed access setup. For example, if your Anyscale hosted cloud is on AWS, then adding your AWS credentials to your private bucket means that `aws` can't access the default object storage bucket (`$ANYSCALE_ARTIFACT_STORAGE`) anymore. Thus, reset your credentials by simply setting the relevant environment variables to the empty string.
3. Next, you can upload your model weights to `$ANYSCALE_ARTIFACT_STORAGE` via the relevant cli (AWS S3/ GCS depending on your Anyscale Cloud). For example:

    GCP: 
    ```bash
    gcloud storage cp -r mymodels/ $ANYSCALE_ARTIFACT_STORAGE/mymodels/
    ```

    AWS:
    ```bash
    aws s3 sync mymodels/ $ANYSCALE_ARTIFACT_STORAGE/mymodels/
    ``` 

4. Finally, you can update the `initial_base_model_ckpt_path` in your training config YAML.

# Model weights stored locally (--- to be edited)

For local files you have two options: 
1. Upload to remote storage and follow the instructions above (the more reliable option for large datasets). 
2. Upload directly to your Anyscale workspace: This is the simplest option for small files. You can use the UI in your VSCode window (simply right click -> upload files/folder) and upload your training files. This data needs to be placed in the shared cluster storage `/mnt/cluster_storage` so that it's accessible by all the worker nodes. (For more on workspace storage, see our guide [here](https://docs.anyscale.com/platform/workspaces/workspaces-storage/)). For example, let's say I uploaded a folder `my_files` with the following structure:

    ```
    myfiles/  
    ├── train.jsonl
    └── val.jsonl
    ```

    I would now do:

    ```bash
    mv myfiles /mnt/cluster_storage
    ```

    Next, update your training config YAML to point to the right training and validation files. 

    ```yaml
    train_path: /mnt/cluster_storage/myfiles/train.jsonl
    valid_path: /mnt/cluster_storage/myfiles/test.jsonl
    ```

**Note:** If you anticipate to use the same dataset files for multiple runs/ across workspace sessions, you should upload the files to `$ANYSCALE_ARTIFACT_STORAGE`.

# Specifying the right model ID and prompt format

The overarching idea is that specifying a model ID will provide context to the architecture of the LLM, which will guide how the model will be further trained. It won't necessarily mean that the model that is specified here is the model that will be fine-tuned. For that, we will rely on the weights that are provided. Knowing that the Llama Guard 2 model shares the same architecture as the Llama 3 8B model (or any other Llama model for that matter) makes it a suitable choice for the model ID.

Different LLMs are trained with varifying prompt formats. Leveraging this format and continuing with it make the finetuning more effective. 

