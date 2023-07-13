# Anyscale Service - Image classification

This template provides an example of an image classification service using a Resnet-50 model. It showcases the flexibility of Ray Serve to run any model backend or optimizers by offering 3 different examples -
1. PyTorch,
2. [ONNX Runtime](https://onnxruntime.ai/) a cross-platform inference accelerator,
3. Using [Torch-TensorRT](https://pytorch.org/TensorRT/) to leverage NVIDIA’s [TensorRT](https://developer.nvidia.com/tensorrt) Deep Learning Optimizer and Runtime for optimizing model inference. 

Running a model using ONNX or Torch-TensorRT does require some model preparation steps. These are included in the `__init__()` methods but may also be done offline. 

## Running locally on an Anyscale Workspace

After launching this template, within the Workspace terminal you can run following command to start a serve application.

`serve run pt-resnet:model`

Once the serve application has started successfully, open another terminal and test it using:

`python query.py`

The script runs 100 queries. We recommend to repeat the same with TensorRT and ONNX to compare performance and choose the best one for your application.

## Roll out as Anyscale service

You can also use the `*-service.yaml` files to deploy an Anyscale Service. Run the following command:

`anyscale service rollout -f trt-service.yaml --name {ENTER_SERVICE_NAME}`

You should see the terminal output URL to view & manage your Service within a few seconds. 
