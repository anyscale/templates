# Using Triton Inference Server on Anyscale Services

**⏱️ Time to complete**: 30 minutes on AWS and 1 hour on GCP

This guide walks you through develop and deploy Triton Server applications in Anyscale
though running a Stable Diffusion 1.5 service.

This tutorial shows:
1. Build Docker image for Triton Server to run in Anyscale platform.
2. Compile model using Triton's Python backend on Anyscale Workspaces.
3. Run Triton Server on Ray Serve locally on Anyscale Workspaces.
4. Deploy the application on Anyscale Services.

**Note**: This guide isn't meant to substitute with the official Triton documentation.
For more information, refer to the
[Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html).

## Build Docker image for Triton Server to run in Anyscale platform

The tutorial starts by building a Docker image that can run properly in Anyscale
platform. You don't need to build the image from scratch. The image used to run this
tutorial is already used in the workspace. This section is just for informational
purposes. Some of the code used here are from Nvidia's tutorials from
[TenserRT Stable Diffusion](https://github.com/NVIDIA/TensorRT/blob/release/10.0/demo/Diffusion/README.md),
[Triton building complex pipeline](https://github.com/triton-inference-server/tutorials/blob/r24.04/Conceptual_Guide/Part_6-building_complex_pipelines/README.md),
and [Triton Ray Serve Deployment](https://github.com/triton-inference-server/tutorials/tree/r24.04/Triton_Inference_Server_Python_API/examples/rayserve).
If you need to learn more about TensorRT and Triton, refer to their official
documentation and tutorial.

In Nvidia's tutorial, they have an image to build the model and another image to serve
the model. In this tutorial, the docker image pulls in all the necessary dependencies.
This image compiles the model, allow you to do local development in Anyscale Workspace,
and deploys to Anyscale Services.

The full `Dockerfile` used to build the image is already provided in the workspace.
You can view the file by clicking on the `Dockerfile` in the file explorer. It shows
what's required in the image to run this tutorial.

## Compile model using Triton's Python backend on Anyscale Workspaces

Unlike Nvidia's tutorial, this tutorial doesn't include models in the Docker image.
Instead, in this section, you are building the model and upload the model to a cloud
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

Compile the model using Triton's Python backend. This takes 10-15 minutes on a
T4 GPU and 8-10 minutes on an A10G GPU. The model saves in the `model_repository`
directory as TensorRT engine artifacts. Also keep in mind the model compiling
needs to in the same type of GPU you are planning to serve the model on.



```python
import time
import datetime


print(f"start time: {datetime.datetime.now()}")
t0 = time.time()
model = triton_server.load("stable_diffusion_1_5")
duration = time.time() - t0
print(f"Total duration: {duration}s")

# Unload the model and the server to free the memory.
triton_server.unload(model, wait_until_unloaded=True)
triton_server.stop()
```

After the model compile, you can upload the model to a cloud storage. You need to
upload both the model config file `config.pbtxt` and the TensorRT engine model
directory. Anyscale provides a environment variable `ANYSCALE_ARTIFACT_STORAGE` for
customers to store model artifacts. Use one of the following to upload the model.


```python
# If you are running in AWS.
aws s3 cp /tmp/workspace/diffusion-models/stable_diffusion_1_5/config.pbtxt $ANYSCALE_ARTIFACT_STORAGE/triton_model_repository/stable_diffusion_1_5/config.pbtxt
aws s3 cp /tmp/workspace/diffusion-models/stable_diffusion_1_5/1/1.5-engine-batch-size-1/ $ANYSCALE_ARTIFACT_STORAGE/triton_model_repository/stable_diffusion_1_5/1/1.5-engine-batch-size-1/ --recursive
```


```python
# If you are running in GCP.
gcloud storage cp /tmp/workspace/diffusion-models/stable_diffusion_1_5/config.pbtxt $ANYSCALE_ARTIFACT_STORAGE/triton_model_repository/stable_diffusion_1_5/config.pbtxt
gcloud storage cp /tmp/workspace/diffusion-models/stable_diffusion_1_5/1/1.5-engine-batch-size-1/ $ANYSCALE_ARTIFACT_STORAGE/triton_model_repository/stable_diffusion_1_5/1/1.5-engine-batch-size-1/ --recursive
```

## Run Triton Server on Ray Serve locally on Anyscale Workspaces

The `triton_app.py` included in this workspace demonstrates using
remote model repository to start triton serve, loading the specific Stable Diffusion
model, run inference with Triton, and serving the response through Ray Serve.
You can view the file by clicking on the `triton_app.py` in the file explorer. In
addition, you can also do prompt engineering, apply business logic, or doing model
composition with Ray Serve before returning the response as image. Run the follow code
to start Triton Server with Ray Serve.


```python
serve run triton_app:triton_deployment --non-blocking
```

Cluster autoscaler starts a GPU worker node. The model downloads from the
cloud storage location where you just uploaded the model artifacts, and then loaded into
the Triton Server and serve the endpoint with Ray Serve. It might take few minutes to
start the server.

Once you see the message "Deployed app 'default' successfully." You can run the
following command to query the endpoint and save the image to a local file.


```python
curl "http://localhost:8000/generate?prompt=dogs%20in%20new%20york,%20realistic,%204k,%20photograph" > dogs_photo.jpg
```

An example of generated image looks like the following

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/triton_services/assets/dogs_photo.jpg"/>

## Deploy the application on Anyscale Services

Once you completed local development on the workspace and ready to move to production,
you can deploy the service onto Anyscale Services by running the following command.



```python
anyscale service deploy --name "triton-stable-diffusion" triton_app:triton_deployment
```

This command starts a fresh cluster, and deploy the same code you just ran locally
to a service. The cluster should contain two nodes, a head node, and a worker node,
where the replica is running on the worker. You can see output similar to the following

```commandline
(anyscale +1.5s) Starting new service 'triton-stable-diffusion'.
(anyscale +2.1s) Uploading local dir '.' to cloud storage.
(anyscale +2.9s) Including workspace-managed pip dependencies.
(anyscale +3.8s) Service 'triton-stable-diffusion' deployed.
(anyscale +3.8s) View the service in the UI: 'https://console.anyscale.com/services/service2_s8cwtlwwvukzxzd256z1wyqmj9'
(anyscale +3.8s) Query the service once it's running using the following curl command:
(anyscale +3.8s) curl -H 'Authorization: Bearer pnnHyxUG_v6hzLbUn7LLmgNjF5g3t0XAxa0TXoRFV6g' https://triton-stable-diffusion-bxauk.cld-kvedzwag2qa8i5bj.s.anyscaleuserdata.com/
```

You can click in the link to the services UI to check the status. Once it's in the
running status, you can run the following command to test the endpoint. Make sure to
change the bearer token and the base URL to the one showed from the deployment output.
This command queries against the newly deployed service and store the generated image
locally.


```python
curl -H "Authorization: Bearer pnnHyxUG_v6hzLbUn7LLmgNjF5g3t0XAxa0TXoRFV6g" \
    "https://triton-stable-diffusion-bxauk.cld-kvedzwag2qa8i5bj.s.anyscaleuserdata.com/generate?prompt=dogs%20in%20new%20york,%20realistic,%204k,%20photograph" \
    > dogs_photo_service.jpg
```

An example of generated image looks like the following

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/triton_services/assets/dogs_photo_service.jpg"/>
