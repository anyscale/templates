# Running Experiments in Parallel with Tune

Ray Tune lets you easily run experiments in parallel across a cluster.

In this tutorial, you will learn:
1. How to set up a Ray Tune app to run an parallel grid sweep across a cluster.
2. Basic Ray Tune features, including stats reporting and storing results.
3. Monitoring cluster parallelism and execution using the Ray dashboard.

**Note**: This tutorial is run within a workspace. Please overview the ``Introduction to Workspaces`` template first before this tutorial.

## Grid search hello world

Let's start by running a quick "hello world" that runs a few variations of a function call across a cluster. It should take about 10 seconds to run:


```python
from ray import tune

def f(config):
    print("hello world from variant", config["x"])
    return {"my_result_metric": config["x"] ** 2}

tuner = tune.Tuner(f, param_space={"x": tune.grid_search([0, 1, 2, 3, 4])})
results = tuner.fit()
print(results)
```

### Interpreting the results

You should see during the run a table of the trials created by Tune. One trial is created for each individual value of `x` in the grid sweep. The table shows where the trial was run in the cluster, how long the trial took, and reported metrics:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-tune/assets/tune-status.png" width=800px/>

On completion, it returns a `ResultGrid` object that captures the experiment results. This includes the reported trial metrics, the path where trial results are saved:

```py
ResultGrid<[
  Result(
    metrics={'my_result_metric': 0},
    path='/home/ray/ray_results/f_2024-02-27_11-40-53/f_1e2c4_00000_0_x=0_2024-02-27_11-40-56',
    filesystem='local',
    checkpoint=None
  ),
  ...
```

 Note that the filesystem of the result says "local", which means results are written to the workspace local disk. We'll cover how to configure [Tune storage](https://docs.ray.io/en/latest/tune/tutorials/tune-storage.html) for a distributed run later in this tutorial.

### Viewing trial outputs

To view the stdout and stderr of the trial, use the ``Logs`` tab in the Workspace UI. Navigate to the log page and search for "hello", and you'll be able to see the logs printed for each trial run in the cluster:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-tune/assets/tune-logs.png" width=800px/>

Tune also saves a number of input and output metadata files for each trial to storage, you can view them by querying the returned result object:
- ``params.json``: The input parameters of the trial
    - ``params.pkl`` pickle form of the parameters (for non-JSON objects)
- ``result.json``: Log of intermediate and final reported metrics
    - ``progress.csv``: CSV form of the results
    - ``events.out.tfevents``: TensorBoard form of the results


```python
import os

# Print the list of metadata files from trial 0 of the previous run.
os.listdir(results[0].path)
```

## Configuring a larger-scale run

Next, we'll configure Tune for a larger-scale run on a multi-node cluster. We'll customize the following parameters:
- Resources to request for each trial
- Saving results to cloud storage

The code below walks through how to do this in Tune. Go ahead and run the cell, it will take a few minutes to complete on a multi-node cluster:


```python
from ray import tune, train
import os
import time

# Do a large scale run with 100 trials, each of which takes 60 seconds to run
# and requests two CPU slots from Ray.
# For example, each trial could be training a variation of a model.
NUM_TRIALS = 100
TIME_PER_TRIAL = 60
CPUS_PER_TRIAL = 2

# Define where results are stored. We'll use the Anyscale artifact storage path to
# save results to cloud storage.
STORAGE_PATH = os.environ["ANYSCALE_ARTIFACT_STORAGE"] + "/tune_results"

def f(config):
    # Import model libraries, etc...
    # Load data and train model code here...
    time.sleep(TIME_PER_TRIAL)

    # Return final stats. You can also return intermediate progress
    # using ray.train.report() if needed.
    # To return your model, you could write it to storage and return its
    # URI in this dict, or return it as a Tune Checkpoint:
    # https://docs.ray.io/en/latest/tune/tutorials/tune-checkpoints.html
    return {"my_result_metric": config["x"] ** 2, "other_data": ...}

# Define trial parameters as a single grid sweep.
trial_space = {
    # This is an example parameter. You could replace it with filesystem paths,
    # model types, or even full nested Python dicts of model configurations, etc.,
    # that enumerate the set of trials to run.
    "x": tune.grid_search(range(NUM_TRIALS)),
}

# Can customize resources per trial, including CPUs and GPUs.
f_wrapped = tune.with_resources(f, {"cpu": CPUS_PER_TRIAL})

# Start a Tune run and print the output.
tuner = tune.Tuner(
    f_wrapped,
    param_space=trial_space,
    run_config=train.RunConfig(storage_path=STORAGE_PATH),
)
results = tuner.fit()
print(results)
```

## Monitoring Tune execution in the cluster

Let's observe how the above run executed in the Ray cluster for the workspace. To do this, go to the "Ray Dashboard" tab in the workspace UI.

First, let's view the run in the Jobs sub-tab and click through to into the job view. Here, you can see an overview of the job, and the status of the individual actors Tune has launched to parallelize the job:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-tune/assets/tune-jobs-1.png" width=800px/>

You can further click through to the actors sub-page and view the status of individual running actors. Inspect trial logs, CPU profiles, and memory profiles using this page:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-tune/assets/tune-jobs-2.png" width=800px/>

Finally, we can observe the holistic execution of the job in the cluster in the Metrics sub-tab. When running the above job on a 36-CPU cluster, we can see that Tune was able to launch ~16 concurrent actors for trial execution, with each actor assigned 2 CPU slots as configured:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-tune/assets/tune-metrics.png" width=800px/>


That concludes our overview of Ray Tune in Anyscale. To learn more about advanced features of Tune and how it can improve your experiment management lifecycle, check out the [Ray Tune docs](https://docs.ray.io/en/latest/tune/index.html).

## Summary

This notebook:
- Run a basic parallel grid sweep experiment in a workspace.
- Showed how to configure Ray Tune's storage and scheduling options.
- Demoed how to debug an experiment run in the cluster using observability tools.


