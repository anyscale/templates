# Bring your own data 
**⏱️ Time to complete**: 10 minutes

This guide focuses on how you can bring your own data to fine-tune your model on the Anyscale Platform. Make sure you've gone over the [basic fine-tuning guide](../../README.md) before going over this cookbook.




# Data stored in remote storage 

## Public datasets
For datasets configured for public access, you simply need to add the relevant training and validation file URI in your training YAML. We support loading from data stored on S3 and GCS.

## Private datasets
With private data, you have two options: 

### Option 1: Configure permissions directly in your cloud account
The most convenient option is to provide read permissions for your Anyscale workspace for the specific bucket. You can follow our guide to do so [here](https://docs.anyscale.com/configuration/cloud-storage-buckets#access-private-cloud-storage).


### Option 2: Sync data into default cloud storage provided by Anyscale
The other option you have is to sync your data into Anyscale-provided storage and then continue with fine-tuning. Let's consider private data on AWS S3. First, we'll need to configure your workspace to be able to access the data. Run `aws configure` or directly export relevant environment variables directly (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, etc) in your current terminal session. We recommend that you simply use environment variables since it's easier to revert this operation (you will see why later).  With that aside, what we want to do is to move this data into 
the [default object storage bucket](https://docs.anyscale.com/platform/workspaces/workspaces-storage#object-storage-s3-or-gcs-buckets) provided by Anyscale (`$ANYSCALE_ARTIFACT_STORAGE`). That way, across runs/ workspace restarts, you don't have to repeat this process.
1. First, download the data into your workspace:  
    ```bash
    aws s3 sync s3://<bucket_name>/<path_to_data_dir>/ myfiles/
    ```
2. The default object storage bucket configured for you in your workspace uses Anyscale-managed credentials internally. It is recommended to reset the credentials you provided so as to not interfere with the Anyscale-managed access setup. For example, if your Anyscale hosted cloud is on AWS, then adding your AWS credentials means that `aws` can't access the default object storage bucket anymore. Thus, reset your credentials through the same method you used. If you simply exported relevant environment variables, it's simple to reset them (set to empty string). If you used `aws configure`, you'll need to delete the config and credential files `~/.aws/config` and `~/.aws/credentials`. 
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

