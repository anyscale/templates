# Bring your own data 
**⏱️ Time to complete**: 10 minutes

This guide focuses on how you can bring your own data to fine-tune your model on the Anyscale Platform. Make sure you have gone over the [basic fine-tuning guide](../../README.md) before going over this cookbook.


# Table of Contents
1. [Data stored in remote storage](#data-stored-in-remote-storage)
    - [Public datasets](#public-datasets)
    - [Private datasets](#private-datasets)
2. [Data stored locally](#data-stored-locally)

## Example YAML

We specify training and validation file paths in the `train_path` and `valid_path` entries in the config file as shown in the example YAML below. Validation file path is optional.

```yaml
model_id: meta-llama/Meta-Llama-3-8B 
train_path: s3://air-example-data/gsm8k/train.jsonl # <-- change this to the path to your training data
valid_path: s3://air-example-data/gsm8k/test.jsonl # <-- change this to the path to your validation data. This is optional
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

# Data stored in remote storage 

## Public datasets
For datasets configured for public access, you simply need to add the relevant training and validation file URI in your training YAML. We support loading from data stored on S3 and GCS.


## Private datasets
With private data, you have two options: 

### Option 1: Configure permissions directly in your cloud account
The most convenient option is to provide read permissions for your Anyscale workspace for the specific bucket. You can follow our guide to do so [here](https://docs.anyscale.com/configuration/cloud-storage-buckets#access-private-cloud-storage).


### Option 2: Sync data into default cloud storage provided by Anyscale
The other option you have is to sync your data into Anyscale-provided storage and then continue with fine-tuning. Let's consider private data on AWS S3. First, we'll need to configure your workspace to be able to access the data. We recommend that you simply export relevant environment variables directly (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, etc) into your current terminal session.  With that aside, what we want to do is to move this data into 
the [default object storage bucket](https://docs.anyscale.com/platform/workspaces/workspaces-storage#object-storage-s3-or-gcs-buckets) provided by Anyscale (`$ANYSCALE_ARTIFACT_STORAGE`). That way, across runs/ workspace restarts, you don't have to repeat this process (compared to just downloading the files into your workspace).
1. First, download the data into your workspace:  
    ```bash
    aws s3 sync s3://<bucket_name>/<path_to_data_dir>/ myfiles/
    ```
2. The default object storage bucket configured for you in your workspace uses Anyscale-managed credentials internally. It is recommended to reset the credentials you provided so as to not interfere with the Anyscale-managed access setup. For example, if your Anyscale hosted cloud is on AWS, then adding your AWS credentials to your private bucket means that `aws` can't access the default object storage bucket (`$ANYSCALE_ARTIFACT_STORAGE`) anymore. Thus, reset your credentials by simply setting the relevant environment variables to the empty string.
3. Next, you can upload your data to `$ANYSCALE_ARTIFACT_STORAGE` via the relevant cli (AWS S3/ GCS depending on your Anyscale Cloud). For example:

    GCP: 
    ```bash
    gcloud storage cp -r myfiles/ $ANYSCALE_ARTIFACT_STORAGE/myfiles/
    ```

    AWS:
    ```bash
    aws s3 sync myfiles/ $ANYSCALE_ARTIFACT_STORAGE/myfiles/
    ``` 

4. Finally, you can update the training and validation paths in your training config YAML.

# Data stored locally

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

