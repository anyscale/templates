# Introduction to Anyscale Jobs

Submit your machine learning apps as [Anyscale Jobs](https://docs.anyscale.com/preview/preview/platform/jobs/) for scalability, programmability, fault tolerance, and persisting outputs like logs.

**⏱️ Time to complete**: 10 min

After implementing and testing your machine learning workloads, it’s time to move them into production. An Anyscale Job packages your application code, dependencies, and compute configurations.

This example takes you through a common development to production workflow with Anyscale Jobs:

1. Development
    a. Run an app in a workspace.
2. Production
    a. Submit the app to Anyscale Jobs.
    b. View the output.
3. Submit the job externally from another machine.

## Walkthrough

First, let's run the following app first interactively in the current workspace.

This template includes a simple processing job in **./main.py** that runs a few Ray tasks. Run the cell below in the workspace, you should see it print the result after a few seconds.


```python
# First install the necessary `emoji` dependency.
!pip install emoji
```


```python
# Then run the Ray app script.
!python main.py
```

Next, let's try submitting the app to Anyscale Jobs. Within a workspace, you can use the "anyscale job submit" (job runs will be managed by Anyscale Jobs) functionality for this.

The following cell should also run to completion within a few minutes and print the same result. Note however that the Ray app was not run within the workspace cluster (you can check the ``Ray Dashboard`` to verify). It was submitted to Anyscale for execution on a new Ray cluster.


```python
# Second, submit the Ray app for execution on a new Ray cluster.
# The execution will be managed by Anyscale Jobs.
!anyscale job submit --name my-job --wait -- python main.py

# Tip: You can run any Ray app as a job by prefixing its entrypoint with "anyscale job submit --".
```

### Job UI Overview

You can view active and historical job runs at (`Home > Jobs`). Click into the job run created by the above cell to inspect its results.

You should see the job state and its output on the overview page.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-jobs/assets/anyscale-job.png" height=400px>

## Submitting a Job programmatically

In the above section, you submitted a Job from a workspace. By default, Jobs submitted from workspaces inherit the dependencies and compute config of the workspace.

You can also submit jobs from other machines, using the Anyscale CLI. 

Copy ``main.py`` to an empty folder on your **laptop**, and then run the following on your **laptop** to try this out:

```bash
# Make sure we have anyscale CLI installed.
$ pip install -U anyscale

# Note: outside of workspaces, you must specify required files via --working-dir.
$ anyscale job submit --working-dir=. --wait -- python main.py
```

Jobs submitted externally will run with the Anyscale default autoscaling compute config and dependencies. To override these settings, use the ``--config-file``, ``--image-uri``, or ``--containerfile`` flags.

This concludes the Anyscale Jobs tutorial. To learn more about how to configure Anyscale Jobs, see the [Anyscale documentation](https://docs.endpoints.anyscale.com/preview/).

## Summary

This notebook:
- Ran a simple Ray app in the local workspace.
- Submitted the same Ray app as an Anyscale Job.
- Walked through how to submit the same Job externally from a different machine.
