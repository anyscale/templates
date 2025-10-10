# High-Performance and Robust Model Training With PyTorch and Ray Train: Optimizing for Cost and Performance

- Migrate model / data to Ray Train
- Metrics and checkpointing
- Fault Tolerance
- Integrate Ray Data and Ray Train
- Train Stable Diffusion

Bonus notebook: Observability (dashboards, profiling)

## About the instructor

##### Adam Breindel, Lead Technical Instructor, Anyscale

__Contact:__ `adamb@anyscale.com` - https://www.linkedin.com/in/adbreind

<img src="https://materials.s3.amazonaws.com/i/lpMDU9j.jpg" width=200 style="margin-right:2em;margin-top:1em"/>

* 25+ years building systems for startups and large enterprises including AI/ML, mobile, web, data eng
* 15+ years teaching front- and back-end technology

__Interesting projects...__
* My first full-time job in tech involved streaming neural net fraud scoring (debit cards)
* Realtime & offline analytics for banking
* Music synchronization and licensing for networked jukeboxes

__Industries/verticals__
* Finance / Insurance, Travel, Media / Entertainment, Government, Energy, Tech

## Useful links

Ray - https://www.ray.io/

* Ray docs - https://docs.ray.io/en/latest/

Anyscale - https://www.anyscale.com/

* Anyscale platform docs - https://docs.anyscale.com/
* Anyscale console (login) - https://console.anyscale.com/

## container

```
FROM anyscale/ray:2.49.2-slim-py312-cu128

RUN pip install --no-cache-dir --upgrade "torch==2.6.0" "torchvision==0.21.0" "matplotlib==3.10.1" "diffusers==0.32.2" "transformers==4.50.0" "accelerate==1.5.2" "xgboost==2.1.4" "pytorch-lightning==2.5.1" "pyarrow==19.0.1" "datasets==3.5.0" "evaluate==0.4.3" "scikit-learn==1.6.1" "torch-tb-profiler==0.4.3" "tensorboard==2.19.0"

# Add your Debian packages here. Do not install unnecessary packages for a smaller image size to optimize build and cluster startup time.
RUN sudo apt-get update -y \
    && sudo apt-get install --no-install-recommends -y wget \
    && sudo rm -f /etc/apt/sources.list.d/*
```

## hardware

```
cloud: education-us-west-2
head_node:
  instance_type: g4dn.2xlarge
  resources:
    CPU: 0
    GPU: 0
  flags: {}
worker_nodes:
  - instance_type: g5.2xlarge
    flags: {}
    min_nodes: 2
    max_nodes: 2
    market_type: ON_DEMAND
enable_cross_zone_scaling: true
auto_select_worker_config: false
idle_termination_minutes: 120
flags:
  allow-cross-zone-autoscaling: true
```
