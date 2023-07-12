# Distributed Training With PyTorch on Fashion MNIST 

In this tutorial you will train models on the Fashion MNIST dataset using PyTorch.


| Details | Description |
| ---------------------- | ----------- |
| Summary | This tutorial demonstrates how to set up distributed training with PyTorch using the MNIST dataset and run on Anyscale|
| Time to Run | Less than 5 minutes |
| Compute Requirements | We recommend at least 1 GPU node. The default will scale up to 3 worker nodes each with 1 NVIDIA T4 GPU. |
| Cluster Environment | This template uses a docker image built on top of the latest Anyscale-provided Ray image using Python 3.10, which comes with PyTorch: [`anyscale/ray-ml:2.5.1-py310-gpu`](https://docs.anyscale.com/reference/base-images/overview). See the appendix below for more details. |

## Running the tutorial
### Background
The tutorial is run from an Anyscale Workspace.  Anyscale Workspaces is a fully managed development environment that enables ML practitioners to build distributed Ray applications and advance from research to development to production easily, all within a single environment. The Workspace provides developer friendly tools like VSCode and Jupyter backed by a remote scaling Ray Cluster for development.

Anyscale requires 2 configs to start up a Workspace Cluster:
1. A cluster environment that handles dependencies.
2. A compute configuration that determines how many nodes of each type to bring up. This also configures how many nodes are available for autoscaling.

Those have been set by default in this tutorial but can be edited and updated if needed and is covered in the appendix.

### Run
The tutorial includes a pyton script with the code to do distributed training.  You can execute this training script directly from the workspace terminal.

To run:
```bash
python pytorch.py
```

You'll see training iterations and metrics as the training executes.

### Monitor
After launching the script, you can look at the Ray dashboard. It can be accessed from the Workspace home page and enables users to track things like CPU/GPU utilization, GPU memory usage, remote task statuses, and more!

![Dash](https://github.com/anyscale/templates/releases/download/media/workspacedash.png)

[See here for more extensive documentation on the dashboard.](https://docs.ray.io/en/latest/ray-observability/getting-started.html)

### Model Saving
The model will be saved in the Anyscale Artifact Store, which is automatically set up and configured with your Anyscale deployment.

For every Anyscale Cloud, a default object storage bucket is configured during the Cloud deployment. All the Workspaces, Jobs, and Services Clusters within an Anyscale Cloud have permission to read and write to its default bucket.

Use the following environment variables to access the default bucket:

1. ANYSCALE_CLOUD_STORAGE_BUCKET: the name of the bucket.
2. ANYSCALE_CLOUD_STORAGE_BUCKET_REGION: the region of the bucket.
3. ANYSCALE_ARTIFACT_STORAGE: the URI to the pre-generated folder for storing your artifacts while keeping them separate them from Anyscale-generated ones.


You can view the saved model and artifacts by running:
```bash
aws s3 ls $ANYSCALE_ARTIFACT_STORAGE
```

Or, if you are on GCP:
```bash
gsutil ls $ANYSCALE_ARTIFACT_STORAGE
```
Authentication is automatcially handled by default.

### Submit as Anyscale Production Job
From within your Anyscale Workspace, you can run your script as an Anyscale Job. This might be useful if you want to run things in production and have a long running job. You can test that each Anyscale Job will spin up its own cluster (with the same compute config and cluster environment as the Workspace) and run the script.  The Anyscale Job will automatically retry in event of failure and provides monitoring via the Ray Dashboard and Grafana. 

To submit as a Production Job you can run:

```bash
anyscale job submit -- python pytorch.py
```

[You can learn more about Anyscale Jobs here.](https://docs.anyscale.com/productionize/jobs/get-started)

### Next Steps

#### Training on your own data: Modifying the Script 
You can easily modify the script in VSCode or Jupyter to use your own data, add data pre-processing logic, or change the model architecture!  Read more about loading data with Ray [from your file store or database here](https://docs.ray.io/en/latest/data/loading-data.html). 

Once the code is updated, run the same command as before to kick off your training job:
```bash
anyscale job submit -- python pytorch.py
```