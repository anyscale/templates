# Running LLM finetuning template as an Anyscale job
**⏱️ Time to complete**: 10 minutes

While for developer velocity it is preferable to use Workspace to run python scripts, for automation and launching jobs from your laptop without having to spin up a workspace it is recommended to run the fine-tuning workloads via isolated [Anyscale jobs](https://docs.anyscale.com/platform/jobs/). You may also want to launch production long running jobs through workspace since workspace setup might be set as ephemeral. 

For specifying a job, you have to specify the command that needs to run (ex. [COMMAND][ARGS]) along with the requirements (ex. docker image, additional, pip packages, etc.) in a job yaml and then call `anyscale job submit` to launch the job on Anyscale.

> **Note**: Executing an Anyscale Job within a Workspace will ensure that files in the current working directory are available for the Job (unless excluded with --exclude). But we can also load files from anywhere (ex. Github repo, S3, etc.) if we want to launch a Job from anywhere.

### Launching from workspace


```python
!cat ./job_configs/job_workspace.yaml
```

    name: "llmforge-job"
    entrypoint: "llmforge anyscale finetune config/llama-3-8b.yaml"
    max_retries: 0


### Launching from outside of workspace (local laptop or CI/CD pipelines)


```python
!cat ./job_configs/job_client.yaml
```

    name: "llmforge-job"
    entrypoint: "llmforge anyscale finetune config/llama-3-8b.yaml"
    image_uri: <replace_with_llmforge_image_uri_value>
    max_retries: 0
    working_dir: "."


These available settings can be found on Anyscale jobs API [guide](https://docs.anyscale.com/reference/job-api/#jobconfig). A few notes:

- `entrypoint` is basically the command we want to run. Pay attention to the relative file location (config/llama-3-8b.yaml) and the `working_dir`. Inside `llama-3-8b.yaml` we are also referencing a relative path to `config/zero_3_offload_optim+param.json`. This works because we specify the `working_dir` to be the current directory `.` when submitting the job from client side. If submitting from the workspace the `~/default` directory is treated as `working_dir`.
- `image_uri` refers to the image that has llmforge installed. On the finetuning template the latest released image is automatically listed. For the full list of versions and their uris visit [llmforge versions](../../README.md#llmforge-versions). If you run this job from a workspace the image_uri will be inherited from the workspace image.
- `max_retries`: setting this to zero makes sure we do not keep retrying if the job fails. We should retry only when the job is flaky (maybe due to resource constraints, etc.)
- `working_dir`: Setting working_dir to the current directory is necessary when you are submitting this job from outside of workspace (Your laptop or CI/CD pipelines)


```python
!anyscale job submit --config-file ./job_configs/job_workspace.yaml
```

    [1m[36mOutput[0m[0m
    [0m[1m[36m(anyscale +0.8s)[0m [0m[0m[0m[0mSubmitting job with config JobConfig(name='llmforge-job', image_uri=None, compute_config=None, env_vars=None, py_modules=None, cloud=None, project=None, ray_version=None).[0m
    [0m[1m[36m(anyscale +3.3s)[0m [0m[0m[0m[0mUploading local dir '.' to cloud storage.[0m
    [0m[1m[36m(anyscale +4.9s)[0m [0m[0m[0m[0mIncluding workspace-managed pip dependencies.[0m
    [0m[1m[36m(anyscale +5.3s)[0m [0m[0m[0m[0mJob 'llmforge-job' submitted, ID: 'prodjob_i3s5rcw8kd1eh6vr7rekr4ckrx'.[0m
    [0m[1m[36m(anyscale +5.3s)[0m [0m[0m[0m[0mView the job in the UI: https://console.anyscale.com/jobs/prodjob_i3s5rcw8kd1eh6vr7rekr4ckrx[0m
    [0m[1m[36m(anyscale +5.3s)[0m [0m[0m[0m[0mUse `--wait` to wait for the job to run and stream logs.[0m
    [0m[0m

As the job runs we can go to the provided URL (`console.anyscale.con/jobs/prod_job...`) to monitor the logs and metrics related to the job.

> **Hint**: To provide `WANDB_API_KEY` you can use `env_vars` in the job specification yaml.


