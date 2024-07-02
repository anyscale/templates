# Running LLM finetuning template as an Anyscale job
**⏱️ Time to complete**: 10 minutes

While for developer velocity it is preferable to use Workspace to run python scripts, for automation and launching jobs from your laptop without having to spin up a Workspace it is recommended to run the fine-tuning workloads via isolated [Anyscale jobs](https://docs.anyscale.com/platform/jobs/).

We have to specify the command that needs to run (ex. [COMMAND][ARGS]) along with the requirements (ex. docker image, additional, pip packages, etc.) in a job yaml and then call `anyscale job submit` to launch the job on Anyscale.

> Note: Executing an Anyscale Job within a Workspace will ensure that files in the current working directory are available for the Job (unless excluded with --exclude). But we can also load files from anywhere (ex. Github repo, S3, etc.) if we want to launch a Job from anywhere.


```python
!cat ./job_configs/job.yaml
```

    name: "llmforge-job"
    entrypoint: "llmforge anyscale finetune config/llama-3-8b.yaml"
    image_uri: localhost:5555/anyscale/llm-forge:0.5.0.1-ngmM6BdcEdhWo0nvedP7janPLKS9Cdz2
    requirements: []
    max_retries: 0
    working_dir: "."

A few notes:

- `entrypoint` is basically the command we want to run. Pay attention to the relative file location (config/llama-3-8b.yaml) and the `working_dir`. Inside `llama-3-8b.yaml` we are also referencing a relative path to `config/zero_3_offload_optim+param.json` which works because of specifying the `working_dir` to be the current dir `.`.
- `image_uri` refers to the image that has llmforge installed. On a the finetuning template the latest released image is automatically listed. For the full list of versions and their uris visit [llmforge versions](../../README.md#llmforge-versions). If you run this job from a workspace the image_uri will be inherited from the workspace image.
- `max_retries`: setting this to zero makes sure we do not keep retrying if the job fails. We should retry only when the job is flaky (maybe due to resource constraints, etc.)
- `working_dir`: Setting working_dir to current dir is necessary when you are submitting this job from outside of workspace (Your laptop or CI/CD pipelines)


```python
!anyscale job submit --config-file ./job_configs/job.yaml
```

    [1m[36mOutput[0m[0m
    [0m[1m[36m(anyscale +0.8s)[0m [0m[0m[0m[0mSubmitting job with config JobConfig(name='llmforge-job', image_uri='localhost:5555/anyscale/llm-forge:0.5.0.1-ngmM6BdcEdhWo0nvedP7janPLKS9Cdz2', compute_config=None, env_vars=None, py_modules=None, cloud=None, project=None, ray_version=None).[0m
    [0m[1m[36m(anyscale +7.0s)[0m [0m[0m[0m[0mUploading local dir '.' to cloud storage.[0m
    [0m[1m[36m(anyscale +8.8s)[0m [0m[0m[0m[0mJob 'llmforge-job' submitted, ID: 'prodjob_dd6654glhg3h6t6sulq9y9c6aw'.[0m
    [0m[1m[36m(anyscale +8.8s)[0m [0m[0m[0m[0mView the job in the UI: https://console.anyscale.com/jobs/prodjob_dd6654glhg3h6t6sulq9y9c6aw[0m
    [0m[1m[36m(anyscale +8.8s)[0m [0m[0m[0m[0mUse `--wait` to wait for the job to run and stream logs.[0m
    [0m[0m

As the job runs we can go to the provided URL (`console.anyscale.con/jobs/prod_job...`) to monitor the logs and metrics related to the job.
