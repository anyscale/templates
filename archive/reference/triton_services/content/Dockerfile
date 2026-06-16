# Start with latest Nvidia's Triton serve base image
FROM nvcr.io/nvidia/tritonserver:25.08-py3

#### Start of Stable Diffusion modifications ####
# Install tokenizers first with latest version to ensure pre-built wheels
RUN pip install --no-cache-dir tokenizers

# Install dependencies related to build and serve Stable Diffusion.
# Align versions with Triton 25.08 official release
RUN pip install --no-cache-dir tritonclient[all]==2.60.0 torch==2.3.0 diffusers==0.23.1 \
    onnx==1.16.0 onnx-graphsurgeon==0.5.6 polygraphy==0.49.9 transformers>=4.35.0 scipy==1.13.0 \
    nvtx==0.2.10 accelerate==0.30.1 optimum[onnxruntime]==1.21.0 \
    --extra-index-url https://pypi.nvidia.com nvidia-modelopt[torch] tensorrt

# Install Triton's Python API and pull in `model.py` and `config.pbtxt` from
# `triton-inference-server/tutorials` repository to serve Stable Diffusion 1.5 model.
RUN git clone https://github.com/triton-inference-server/tutorials.git -b r25.08 --single-branch /tmp/tutorials
RUN pip --no-cache-dir install /tmp/tutorials/Triton_Inference_Server_Python_API/deps/tritonserver-*.whl[all] || echo "Warning: tritonserver wheel not found, using system tritonclient"
RUN mkdir -p /opt/tritonserver/backends/diffusion
RUN cp /tmp/tutorials/Popular_Models_Guide/StableDiffusion/backend/diffusion/model.py /opt/tritonserver/backends/diffusion/model.py
RUN mkdir -p /tmp/workspace/diffusion-models/stable_diffusion_1_5/1
RUN cp /tmp/tutorials/Popular_Models_Guide/StableDiffusion/diffusion-models/stable_diffusion_1_5/config.pbtxt /tmp/workspace/diffusion-models/stable_diffusion_1_5/config.pbtxt

# Install TensorRT and copy the Diffusion backend to be used in Triton. Note the backend
# code consists of loading the Stable Diffusion model, compile into TensorRT, and store
# the TensorRT to Triton's model repository. If you're interested how that works,
# look at the backend diffusion directory.
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt==10.0.1
RUN git clone https://github.com/NVIDIA/TensorRT.git -b v10.0.1 --single-branch /tmp/TensorRT
RUN cp -rf /tmp/TensorRT/demo/Diffusion /opt/tritonserver/backends/diffusion/
#### End of Stable Diffusion modifications ####


#### Start of Anyscale modifications ####
# This section is to set up Anyscale related dependencies for it to run on the Anyscale
# platform. You can keep this section as is regardless which model you are working with.
RUN apt-get clean && apt-get update -y
RUN apt-get install -y sudo tzdata supervisor openssh-client openssh-server rsync zip unzip git nfs-common

# Delete the existing uid 1000 user so you can create the `ray` user.
RUN existing_user=$(id 1000 | sed "s/uid=1000(//;s/) gid=1000.*//") && userdel "$existing_user"
# The `ray` user needs to be at uid 1000 and gid 100.
RUN useradd -ms /bin/bash -d /home/ray ray --uid 1000 --gid 100

RUN sudo usermod -aG root ray
RUN echo 'ray ALL=NOPASSWD: ALL' >> /etc/sudoers
RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean

# Switch to the `ray` user.
USER ray
ENV HOME=/home/ray

RUN sudo apt-get update -y \
    && sudo apt-get install -y python3-venv \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo apt-get clean

RUN python3 -m venv --system-site-packages /home/ray/virtualenv
ENV PATH=/home/ray/virtualenv/bin:$PATH

# Install the Google Cloud SDK.
RUN echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl -f https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
RUN sudo apt-get update && sudo apt-get install google-cloud-cli -y

# Install Azure CLI
RUN curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

RUN mkdir -p /tmp/ray && mkdir -p /tmp/supervisord
RUN pip install --no-cache-dir anyscale jupyterlab==3.6.1 'urllib3<1.27' \
    Pillow==10.3.0 awscli==1.32.110 google-cloud-storage==2.16.0 \
    adlfs azure-identity

# Install Ray with support for the dashboard + cluster launcher (Python 3.12 wheel)
RUN pip install --no-cache-dir "ray[serve] @ https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp312-cp312-manylinux2014_x86_64.whl"

# Install missing dependencies for Ray nightly build AFTER Ray installation
# This ensures they're properly integrated with Ray's dependency management
RUN pip install --no-cache-dir attrs jsonschema packaging setuptools wheel && \
    /home/ray/virtualenv/bin/pip install --no-cache-dir --force-reinstall attrs jsonschema packaging setuptools wheel

# Set environment variables to ensure these packages are found in Ray's runtime environments
ENV PYTHONPATH="/home/ray/virtualenv/lib/python3.12/site-packages:${PYTHONPATH}"
ENV RAY_RUNTIME_ENV_ALLOW_WORKING_DIR_UPLOADS=1


# Give ray user permissions to write to /tmp/workspace for compile the TensorRT engine artifacts.
RUN sudo chmod a+rwx -R /tmp/workspace
