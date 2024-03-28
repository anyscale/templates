# Introduction to Jobs

This tutorial shows you how to:
1. Run a Ray app non-interactively in Anyscale as an "Anyscale Job".
2. Configure and debug Anyscale Jobs.

**Note**: This tutorial is run within a workspace. Please overview the `Introduction to Workspaces` template first before this tutorial.

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

Next, let's try submitting the app to Anyscale Jobs. Within a workspace, you can use the "ray job submit" (job runs will be managed by Anyscale Jobs) functionality for this.

The following cell should also run to completion within a few minutes and print the same result. Note however that the Ray app was not run within the workspace cluster (you can check the ``Ray Dashboard`` to verify). It was submitted to Anyscale for execution on a new Ray cluster.


```python
# Second, submit the Ray app for execution on a new Ray cluster.
# The execution will be managed by Anyscale Jobs.
!ray job submit --wait -- python main.py

# Tip: You can run any Ray app as a job by prefixing its entrypoint with "ray job submit --".
```

### Job UI Overview

You can view active and historical job runs at (`Home > Jobs`). Click into the job run created by the above cell to inspect its results.

You should see the job state and its output on the overview page.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-jobs/assets/anyscale-job.png" height=400px>

This concludes the Anyscale Jobs tutorial. To learn more about how to configure Anyscale Jobs, see the [Anyscale documentation](https://docs.endpoints.anyscale.com/preview/).

## Summary

This notebook:
- Ran a simple Ray app in the local workspace.
- Submitted the same Ray app as an Anyscale Job.
