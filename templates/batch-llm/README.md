# Batch LLM Inference using Anyscale Jobs and vLLM

This template shows you how to:
* Run a LLM inference workload using [vLLM](https://docs.vllm.ai/en/latest) and [Ray Data](https://docs.ray.io/en/latest/data/data.html) on offline large-scale input data.

**Note**: This template can be run from this Workspace.

## How to decide between online VS offline inference for LLM
Online LLM inference (e.g. Anyscale Endpoint) should be used when you want to get real-time response for prompt. Use online inference when you want to optimize latency of inference to be as quick as possible.

On the other hand, offline LLM inference (e.g. Anyscale Job here) should be used when you want to get
reponses for a large number of prompts within an end-to-end time requirement (e.g. minutes to hours granularity). Use offline inference when you want to optimize throughput of inference to use resource
(e.g. GPU) as much as possible on large-scale input data.

## Run notebook on sample input data
Open `batch_llm.ipynb` to play around batch inference on sample input data.
After running the notebook, you should know the basics and APIs to run batch inference.

## Run as a Python script in Workspace
Open `batch_llm.py` to run batch inference on input data hosted on AWS S3.
To be able to run the script, you need to set your Hugging Face token `HF_TOKEN` to access models on Hugging Face.

```py
export HF_TOKEN="..."
python batch_llm.py
```

After running the script on Workspace, you should know the end-to-end workflow to run batch inference on your input data, and store the output data out on cloud storage.

## Run as an Anyscale Job
This Workspace Template includes an example `batch_llm_job.yaml` which can be used to run this batch inference script `batch_llm.py` as an Anyscale Job.

[Anyscale Jobs](https://docs.anyscale.com/jobs/get-started) provide key features such as:

* Automated failure handling
* Automated email alerting
* Record and persist outputs such as logs

To run the as an Ansycale Job from this Workspace, run the following command:
```bash
anyscale job submit batch_llm_job.yaml --follow
```

This will copy the existing python and re-use the [Cluster Environment](https://docs.anyscale.com/configure/dependency-management/cluster-environments) and [Compute Config](https://docs.anyscale.com/configure/compute-configs/overview) to run the job.
