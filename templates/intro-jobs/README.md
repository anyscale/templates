# Introduction to Jobs

**⏱️ Time to complete**: 10 min

This tutorial shows you how to:
1. Run a Ray app non-interactively in Anyscale as an "Anyscale Job".
2. Configure and debug Anyscale Jobs.
3. Submit jobs from Anyscale Workspaces as well as other other machines.

**Note**: This tutorial runs within a workspace. Please overview the `Introduction to Workspaces` template first before this tutorial.

## Key features of Anyscale Jobs

Typically, we recommend running batch Ray apps as Anyscale Jobs when moving to production. Like workspaces, Anyscale Jobs run with their own Ray cluster, so you can run the exact same Ray program in a workspace as a Job too.

Key features of Anyscale Jobs:
- Programmatic submission API
- Automated failure handling
- Automated email alerting
- Record and persist outputs such as logs


**Note**: Ray also has an internal concept of a "Ray job", which is created when running a Ray app. Anyscale Jobs, Workspaces, and Services all launch Ray jobs internally.

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

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-jobs/assets/anyscale-job.png" height=400px />

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

This concludes the Anyscale Jobs tutorial. To learn more about how to configure Anyscale Jobs, see the [Anyscale documentation](https://docs.anyscale.com/platform/jobs/).

## Summary

This notebook:
- Ran a simple Ray app in the local workspace.
- Submitted the same Ray app as an Anyscale Job.
- Walked through how to submit the same Job externally from a different machine.
