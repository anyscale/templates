import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Using Triton Inference Server with Anyscale Services

**⏱️ Time to complete**: 30 minutes on AWS and 1 hour on GCP

This guide develops and deploys a Stable Diffusion 1.5 service as a Triton Server app
in Anyscale.

This tutorial shows how to:
1. Build a Docker image for Triton Server to run on the Anyscale platform.
2. Compile a model using Triton's Python backend in Anyscale Workspaces.
3. Run Triton Server on Ray Serve locally in Anyscale Workspaces.
4. Deploy the app using Anyscale Services.

**Note**: This guide doesn't substitute the official Triton documentation.
For more information, see
[NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html).

**Note**: This tutorial is using a GPU head node because it includes a step to
compile the model. GPU instances can take longer to start and cost more than CPU
instances. If you already compiled your model and stored it in cloud storage, you
can start the head node on a CPU instance instead.


## Build a Docker image for Triton Server to run on the Anyscale platform

The tutorial starts by building a Docker image that can run properly on the Anyscale
platform. You don't need to build the image from scratch. The image in this
tutorial is the same image that the workspace uses. This section is informational.
Some of the tutorial code is from Nvidia's tutorials:
[TenserRT Stable Diffusion](https://github.com/NVIDIA/TensorRT/blob/release/10.0/demo/Diffusion/README.md),
[Triton building complex pipeline](https://github.com/triton-inference-server/tutorials/blob/r24.04/Conceptual_Guide/Part_6-building_complex_pipelines/README.md),
and [Triton Ray Serve Deployment](https://github.com/triton-inference-server/tutorials/tree/r24.04/Triton_Inference_Server_Python_API/examples/rayserve).
To learn more about TensorRT and Triton, see their official documentation and tutorial.

The Nvidia tutorial has an image to build the model and another image to serve the
model. This tutorial has a docker image that pulls in all the necessary dependencies.
This image compiles the model, allows you to do local development in Anyscale
Workspaces, and deploys using Anyscale Services.

Anyscale uses the full `Dockerfile` to build the image and provides it in the workspace.
View the file by clicking on `Dockerfile` in the file explorer. It shows the image
requirements for this tutorial.

## Compile a model using Triton's Python backend in Anyscale Workspaces

Unlike Nvidia's tutorial, this tutorial doesn't include models in the Docker image.
Instead, in this section, you build the model and upload the model to cloud
storage such as AWS S3 or GCP Storage for serving later.

Run this code to start a Triton server using the `/tmp/workspace/diffusion-models`
directory as the model repository.


```python
import tritonserver

model_repository = ["/tmp/workspace/diffusion-models"]

triton_server = tritonserver.Server(
    model_repository=model_repository,
    model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
)
triton_server.start(wait_until_ready=True)
```

Compile the model using Triton's Python backend. The compile takes 10-15 minutes on a
T4 GPU and 8-10 minutes on an A10G GPU. The model saves TensorRT engine artifacts
in the `model_repository` directory. Keep in mind that the model compiling needs to be
in the same type of GPU you plan to serve the model on.



```python
import time
import datetime


print(f"start time: {datetime.datetime.now()}")
t0 = time.time()
"""
The line below executes TensorRT's diffusion backend to compile the model.
It loads the weights from Hugging Face. Export the model into ONNX format
into the model repository directory. Then compile the model into a TensorRT engine
and store the artifacts in the model repository directory. For more details,
see the `stable_diffusion_pipeline.py` file in the `diffusion` directory.
"""
model = triton_server.load("stable_diffusion_1_5")
duration = time.time() - t0
print(f"Total duration: {duration}s")

# Unload the model and the server to free the memory.
triton_server.unload(model, wait_until_unloaded=True)
triton_server.stop()
```

After the model compile completes, the previous step generates some .py and onnx files
in the same model repository. You only need to upload both the model config file
`config.pbtxt` and the TensorRT engine artifacts from the model repository. Anyscale
provides an environment variable `ANYSCALE_ARTIFACT_STORAGE` for customers to store
model artifacts. To learn more about the storage, see
[Object Storage (S3 or GCS buckets)](https://docs.anyscale.com/1.0.0/services/storage/#object-storage-s3-or-gcs-buckets).

Use one of the following tabs to upload the model:


<Tabs>
    <TabItem value="AWS" label="AWS" default>

```python
aws s3 cp /tmp/workspace/diffusion-models/stable_diffusion_1_5/config.pbtxt $ANYSCALE_ARTIFACT_STORAGE/triton_model_repository/stable_diffusion_1_5/config.pbtxt
aws s3 cp /tmp/workspace/diffusion-models/stable_diffusion_1_5/1/1.5-engine-batch-size-1/ $ANYSCALE_ARTIFACT_STORAGE/triton_model_repository/stable_diffusion_1_5/1/1.5-engine-batch-size-1/ --recursive
```

  </TabItem>
  <TabItem value="GCP" label="GCP">

```python
gcloud storage cp /tmp/workspace/diffusion-models/stable_diffusion_1_5/config.pbtxt $ANYSCALE_ARTIFACT_STORAGE/triton_model_repository/stable_diffusion_1_5/config.pbtxt
gcloud storage cp /tmp/workspace/diffusion-models/stable_diffusion_1_5/1/1.5-engine-batch-size-1/ $ANYSCALE_ARTIFACT_STORAGE/triton_model_repository/stable_diffusion_1_5/1/1.5-engine-batch-size-1/ --recursive
```

  </TabItem>
</Tabs>

## Run Triton Server on Ray Serve locally in Anyscale Workspaces

The `triton_app.py` in this workspace demonstrates how to use a remote model repository
to start Triton serve, load the specific Stable Diffusion model, run inference with
Triton, and serve the response with Ray Serve. View the file by clicking on
`triton_app.py` in the file explorer. You can also do prompt engineering, apply
business logic, or do model composition with Ray Serve before returning the response
as image. Run the follow code to start Triton Server with Ray Serve.


```python
serve run triton_app:triton_deployment --non-blocking
```

The cluster autoscaler starts a GPU worker node. The model downloads from the cloud
storage location where you just uploaded the model artifacts, and then loads into
the Triton Server and serve an endpoint with Ray Serve. It might take few minutes to
start the server.

Once you see the message "Deployed app 'default' successfully", you can run the
following command to query the endpoint and save the image to a local file.


```python
curl "http://localhost:8000/generate?prompt=dogs%20in%20new%20york,%20realistic,%204k,%20photograph" > dogs_photo.jpg
```

The following is an example of a generated image:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/triton_services/assets/dogs_photo.jpg"/>

## Deploy the app using Anyscale Services

After you complete local development in the workspace and are ready to move to
production, you can deploy the service with Anyscale Services by running the following
command:



```python
anyscale service deploy --name "triton-stable-diffusion" triton_app:triton_deployment
```

This command starts a fresh cluster, and deploys the same code you ran locally to a
service. The cluster should contain two nodes, a head node, and a worker node, where
the replica is running on the worker. You should see output similar to the following:

```commandline
(anyscale +1.5s) Starting new service 'triton-stable-diffusion'.
(anyscale +2.1s) Uploading local dir '.' to cloud storage.
(anyscale +2.9s) Including workspace-managed pip dependencies.
(anyscale +3.8s) Service 'triton-stable-diffusion' deployed.
(anyscale +3.8s) View the service in the UI: 'https://console.anyscale.com/services/service2_s8cwtlwwvukzxzd256z1wyqmj9'
(anyscale +3.8s) Query the service once it's running using the following curl command:
(anyscale +3.8s) curl -H 'Authorization: Bearer pnnHyxUG_v6hzLbUn7LLmgNjF5g3t0XAxa0TXoRFV6g' https://triton-stable-diffusion-bxauk.cld-kvedzwag2qa8i5bj.s.anyscaleuserdata.com/
```

Click the link to the services UI to check the status. When it's status is `running`,
run the following command to test the endpoint. Make sure to change the bearer token
and the base URL to the values printed in the deployment output. This command queries
the newly deployed service and stores the generated image locally.


```python
curl -H "Authorization: Bearer pnnHyxUG_v6hzLbUn7LLmgNjF5g3t0XAxa0TXoRFV6g" \
    "https://triton-stable-diffusion-bxauk.cld-kvedzwag2qa8i5bj.s.anyscaleuserdata.com/generate?prompt=dogs%20in%20new%20york,%20realistic,%204k,%20photograph" \
    > dogs_photo_service.jpg
```

The following is an example of a generated image:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/triton_services/assets/dogs_photo_service.jpg"/>
