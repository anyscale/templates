# Ray AI Libraries Intro

* Ray and AI Libraries Overview
* Intro to...
    * Ray Data
    * Ray Train
* Ray Data for Batch Inference with GenAI and LLMs 

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

* Ray - https://www.ray.io/
  * Ray docs - https://docs.ray.io/en/latest/
* Anyscale - https://www.anyscale.com/
  * Anyscale platform docs - https://docs.anyscale.com/
  * Anyscale console (login) - https://console.anyscale.com/

# Introduction - Ray - Anyscale

## What is Ray?

<img src='https://docs.ray.io/en/releases-2.38.0/_images/map-of-ray.svg' width=700 />

## Ray

* OSS framework for high-performance, resilient, scale-out computation on heterogeneous hardware
* Distributed scheduler supporting stateless functions ("tasks") as well as long-running stateful processes ("actors")
* Key features: 
  * Dependency tracking (task graphs)
  * Data movement and resource aware
  * Supports mix of resource requirements (e.g., GPUs), fractional, and custom resources
* Additional infra: object store, fault tolerance via GCS
* Ray AI Libraries are a set of high-level APIs for accomplishing common large-scale data + compute use cases (e.g., data transformation, model training)
* Easy, Python-based APIs and coding patterns

## Anyscale: Production-ready Ray from day one

* __Developer central__: multi-node backed IDE, advanced observability by Ray library
* __Optimized runtime__: faster performance and higher GPU utilization vs. OSS
* __Cluster controller__: proactive unhealthy node replacement, 0-100 node 60-sec cold starts
* __Expertise__: Training, 24/7 support, professional services


## container

```
FROM anyscale/ray:2.49.2-slim-py312-cu128

RUN uv pip install -r python_depset.lock --no-cache-dir --no-deps --system

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
