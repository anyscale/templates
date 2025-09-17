# ray-data-vhol-content

## container

```
FROM anyscale/ray:2.49.1-slim-py312-cu128

RUN pip install --no-cache-dir --upgrade "torch==2.6.0" "torchvision==0.21.0" "matplotlib==3.10.1" "diffusers==0.32.2" "transformers==4.50.0" "accelerate==1.5.2"

# Add your Debian packages here. Do not install unnecessary packages for a smaller image size to optimize build and cluster startup time.
RUN sudo apt-get update -y \
    && sudo apt-get install --no-install-recommends -y wget \
    && sudo rm -f /etc/apt/sources.list.d/*

```
    
## hardware

```
cloud: education-us-west-2
head_node:
  instance_type: m5.2xlarge
  flags: {}
worker_nodes:
  - instance_type: g5.2xlarge
    min_workers: 2
    max_workers: 2
enable_cross_zone_scaling: false
flags:
  allow-cross-zone-autoscaling: false
```
