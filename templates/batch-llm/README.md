# Batch LLM Inference using Anyscale Jobs and vLLM

This template shows you how to:
* Run a batch LLM inference workload using [vLLM](https://docs.vllm.ai/en/latest) and [Ray Data](https://docs.ray.io/en/latest/data/data.html)
* Deploy the batch LLM workload as an Anyscale Service

**Note**: This template can be run from this Workspace.

## Review the Jupyter Notebook

## Run as an Anyscale Job
This Workspace Template includes an example `batch_llm_job.yaml` which can be used to run this batch inference as an Anyscale Job.

[Anyscale Jobs](https://docs.anyscale.com/jobs/get-started) provide key features such as:

* Automated failure handling
* Automated email alerting
* Record and persist outputs such as logs

### Step 1: Review the python code

* Open the `batch_llm.py` file
* This template is using the prompts in an S3 bucket in AWS. The prompts are in the format:
```
I always wanted to be a...
The best way to learn a new language is...
```
* This can be updated to any similarly formatted dataset.


## Step 2: Run as an Anyscale Job

To run the as an Ansycale Job from this Workspace, run the following command:
```bash
anyscale job submit batch_llm_job.yaml --follow
```

This will copy the existing python and re-use the [Cluster Environment](https://docs.anyscale.com/configure/dependency-management/cluster-environments) and [Compute Config](https://docs.anyscale.com/configure/compute-configs/overview) to run the job.

## Notes

This template uses a Compute Config Smart Instance Manager (SIM) feature called Min/Max. This allows SIM to dynamically choose which nodes in a given family to launch, with a target of 10 GPUs being available for the cluster.
