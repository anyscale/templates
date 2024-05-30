# Using Triton Inference Server on Anyscale Services

**⏱️ Time to complete**: 30 minutes (1 hour on GCP)

This guide walks you through develop and deploy Triton Server applications in Anyscale
though running a Stable Diffusion 1.5 service. 

In this tutorial, you will learn:
1. Build Docker image for Triton Server to run in Anyscale platform.
2. Compile model using Triton's Python backend on Anyscale Workspaces.
3. Run Triton Server locally on Anyscale Workspaces. 
4. Deploy the application on Anyscale Services.

**Note**: This guide is not meant to substitute with the official Triton documentation.
For more information, please refer to the
[Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html).

## Build Docker image for Triton Server to run in Anyscale platform

Let's start by building a Docker image that can run properly in Anyscale platform. You
do not need to build the image from scratch. The image used to run this tutorial is
already used in the workspace. This section is just for informational purposes.
Some of the code used here are taken piecewise from Nvidia's tutorials from [TenserRT Stable Diffusion](https://github.com/NVIDIA/TensorRT/blob/release/10.0/demo/Diffusion/README.md),
[Triton building complex pipeline](https://github.com/triton-inference-server/tutorials/blob/r24.04/Conceptual_Guide/Part_6-building_complex_pipelines/README.md),
and [Triton Ray Serve Deployment](https://github.com/triton-inference-server/tutorials/tree/r24.04/Triton_Inference_Server_Python_API/examples/rayserve).
If you need to learn more on TensorRT and Triton, please refer to their official
documentation and tutorial.

In Nvidia's tutorial, they have an image that is used to build the model and
another image that is used to serve the model. In this tutorial, we will pull in all the
necessary dependencies into one image. This image will be used to compile the model, 
do local development in the Anyscale Workspace, and deploy to Anyscale Services.

The full `Dockerfile` used to build the image is already provided in the workspace.
You can view the file by clicking on the `Dockerfile` in the file explorer. It starts
with Nvidia's Triton serve base image `nvcr.io/nvidia/tritonserver:24.04-py3`. This
can be updated to the later version depending on when you are running this tutorial in
the future.

```Dockerfile
FROM nvcr.io/nvidia/tritonserver:24.04-py3
```

Then it installs the necessary for dependencies for compile and serve Stable Diffusion
models.

```Dockerfile
RUN pip install --no-cache-dir tritonclient[all]==2.45.0 torch==2.3.0 diffusers==0.23.1 \
    onnx==1.14.0 onnx-graphsurgeon==0.5.2 polygraphy==0.49.9 transformers==4.31.0 scipy==1.13.0 \
    nvtx==0.2.10 accelerate==0.30.1 optimum[onnxruntime]==1.19.2 nvidia-ammo==0.9.4
```

The next section installs Triton's Python API and pulls in `model.py` and `config.pbtxt`
from `triton-inference-server/tutorials` repository to serve Stable Diffusion 1.5 model.

```Dockerfile
RUN git clone https://github.com/triton-inference-server/tutorials.git -b r24.04 --single-branch /tmp/tutorials
RUN pip --no-cache-dir install /tmp/tutorials/Triton_Inference_Server_Python_API/deps/tritonserver-2.41.0.dev0-py3-none-any.whl[all]
RUN mkdir -p /opt/tritonserver/backends/diffusion
RUN cp /tmp/tutorials/Popular_Models_Guide/StableDiffusion/backend/diffusion/model.py /opt/tritonserver/backends/diffusion/model.py
RUN mkdir -p /tmp/workspace/diffusion-models/stable_diffusion_1_5/1
RUN cp /tmp/tutorials/Popular_Models_Guide/StableDiffusion/diffusion-models/stable_diffusion_1_5/config.pbtxt /tmp/workspace/diffusion-models/stable_diffusion_1_5/config.pbtxt
```

Then it installs TensorRT and copy the Diffusion backend to be used in Triton. Also
note the backend code consists of loading the Stable Diffusion model, compile into
TensorRT, and store the TensorRT to Triton's model repository. If you are interested
how that works, you can take a look at the backend diffusion directory.

```Dockerfile
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt==10.0.1
RUN git clone https://github.com/NVIDIA/TensorRT.git -b v10.0.1 --single-branch /tmp/TensorRT
RUN cp -rf /tmp/TensorRT/demo/Diffusion /opt/tritonserver/backends/diffusion/
```

All the above steps are specifically for building and serving the Stable Diffusion 1.5
model. If you are working on different models, you can change them accordingly.

The last section of the `Dockerfile` is to set up Anyscale related dependencies for it
to run on the Anyscale platform. You can keep this section as is regardless which model
you are working with.

## Compile model using Triton's Python backend on Anyscale Workspaces

Unlike Nvidia's tutorial, we do not include models in the Docker image. Instead, in this
section we will build the model and upload the model to a cloud storage such as AWS S3
or GCP Cloud Storage for serving later.

Run this code to start a Triton server using the tmp directory as the model repository.

```python
import tritonserver

model_repository = ["/tmp/workspace/diffusion-models"]

triton_server = tritonserver.Server(
    model_repository=model_repository,
    model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
)
triton_server.start(wait_until_ready=True)
```

Compile the model using Triton's Python backend. This will take 10-15 minutes on a
T4 GPU and 8-10 minutes on an A10 GPU. The model will be compiled and saved in the
`model_repository` directory as TensorRT engine. Also keep in mind the model has to be
built in the same type of GPU you are planning to serve the model on. 

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

After the model is compiled, we can upload the model to a cloud storage. We need to
upload both the model config file `config.pbtxt` and the TensorRT engine model
directory. Anyscale provides a environment variable `ANYSCALE_ARTIFACT_STORAGE` that
can be used to store model artifacts. Use one of the following to upload the model.

```python
# If running in AWS
!aws s3 cp /tmp/workspace/diffusion-models/stable_diffusion_1_5/config.pbtxt $ANYSCALE_ARTIFACT_STORAGE/triton_model_repository/stable_diffusion_1_5/config.pbtxt
!aws s3 cp /tmp/workspace/diffusion-models/stable_diffusion_1_5/1/1.5-engine-batch-size-1/ $ANYSCALE_ARTIFACT_STORAGE/triton_model_repository/stable_diffusion_1_5/1/1.5-engine-batch-size-1/ --recursive
```

```python
# If running in GCP
!gcloud storage cp /tmp/workspace/diffusion-models/stable_diffusion_1_5/config.pbtxt $ANYSCALE_ARTIFACT_STORAGE/triton_model_repository/stable_diffusion_1_5/config.pbtxt
!gcloud storage cp /tmp/workspace/diffusion-models/stable_diffusion_1_5/1/1.5-engine-batch-size-1/ $ANYSCALE_ARTIFACT_STORAGE/triton_model_repository/stable_diffusion_1_5/1/1.5-engine-batch-size-1/ --recursive
```

## Run Triton Server locally on Anyscale Workspaces

The `Deployment.py` is included in this workspace for you. This file demonstrates using
remote model repository to start triton serve, loading the specific Stable Diffusion
model, run inference with Triton, and serving the response through Ray Serve.
You can view the file by clicking on the `Deployment.py` in the file explorer. In
addition, you can also do prompt engineering, apply business logics, or doing model
composition with Ray Serve before returning the response as image. Run the follow code
to start Triton Server with Ray Serve.

```python
!serve run deployment:triton_deployment --non-blocking
```

A GPU worker node will be started by autoscaler. The model will be downloaded from the
cloud storage location where we just uploaded the model artifacts, and then loaded into
the Triton Server and serve the endpoint via Ray Serve. It might take few minutes to
start the server. 

Once you see the message "Deployed app 'default' successfully.". You can run the
following command to query the endpoint and save the image to a local file.

```python
!curl "http://localhost:8000/generate?prompt=dogs%20in%20new%20york,%20realistic,%204k,%20photograph" > dogs_photo.jpg
```

An example of generated image will look like the following

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/triton_services/assets/dogs_photo.jpg"/>

## Deploy the application on Anyscale Services

The `config.yaml` is also included in the workspace for you. You can view the file by
clicking on the `config.yaml` in the file explorer. Once you completed local
development on the workspace and ready to move to production, you can deploy the
service onto Anyscale Services by running the following command.

```python
!anyscale service deploy -f config.yaml --name "triton-stable-diffusion"
```

This command will start a fresh cluster, and deploy the same code you just ran locally
to a service. The cluster should contain two nodes, a head node, and a worker node,
where the replica will be running on the worker. You will see output like the following

```commandline
(anyscale +1.5s) Starting new service 'triton-stable-diffusion'.
(anyscale +2.1s) Uploading local dir '.' to cloud storage.
(anyscale +2.9s) Including workspace-managed pip dependencies.
(anyscale +3.8s) Service 'triton-stable-diffusion' deployed.
(anyscale +3.8s) View the service in the UI: 'https://console.anyscale.com/services/service2_s8cwtlwwvukzxzd256z1wyqmj9'
(anyscale +3.8s) Query the service once it's running using the following curl command:
(anyscale +3.8s) curl -H 'Authorization: Bearer pnnHyxUG_v6hzLbUn7LLmgNjF5g3t0XAxa0TXoRFV6g' https://triton-stable-diffusion-bxauk.cld-kvedzwag2qa8i5bj.s.anyscaleuserdata.com/
```

You can click on the link to the services UI to check the status. Once it's in the
running status, you can run the following command to test the endpoint. Make sure to
change the bearer token and the base URL to the one showed from the above deploy output.
This command will query against the newly deployed service and store the generated image
locally.

```python
!curl -H "Authorization: Bearer pnnHyxUG_v6hzLbUn7LLmgNjF5g3t0XAxa0TXoRFV6g" \
    "https://triton-stable-diffusion-bxauk.cld-kvedzwag2qa8i5bj.s.anyscaleuserdata.com/generate?prompt=dogs%20in%20new%20york,%20realistic,%204k,%20photograph" \
    > dogs_photo_service.jpg

```

An example of generated image will look like the following

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/triton_services/assets/dogs_photo_service.jpg"/>
