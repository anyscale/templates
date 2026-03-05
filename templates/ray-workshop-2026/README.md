# Ray Workshop: Build and Scale AI with Ray

Welcome to the **Ray Workshop**, a hands-on, instructor-led workshop designed to take you from Ray fundamentals to production-ready distributed AI workloads.

## Workshop Overview

This workshop is for developers and ML engineers who are new to Ray or looking to expand their Ray knowledge. Through four progressive modules, you'll gain practical experience building and scaling AI workloads, from parallel Python, to distributed data pipelines, to model training and serving.

**Audience:** New-to-Ray users or those seeking to expand their use of Ray libraries (200-level)

**Primary Goal:** Help participants gain practical knowledge to build and scale AI workloads using Ray's libraries

**Total Duration:** ~3-4 hours

## Modules

| Module | Title | Duration | Description |
|--------|-------|----------|-------------|
|[**Module 1**](Module1/) | Scaling Python for AI Workloads | 45 min | Learn Ray's core concepts: tasks, actors, and the object store. Understand how Ray scales Python and ML workloads, manage resources, and run distributed programs reliably. |
|[**Module 2**](Module2/) | Building Scalable Data Pipelines | 45 min | Build scalable data pipelines with Ray Data. Ingest, transform, and preprocess large multimodal datasets. Learn streaming execution, stateful transforms, and GPU-accelerated batch inference. |
|[**Module 3**](Module3/) | Distributed Training at Scale | 45 min | Scale model training using data and model parallelism with FSDP. Configure distributed training jobs, manage checkpoints, and integrate PyTorch with Ray Train for reliable training at scale. |
|[**Module 4**](Module4/) | Serving Models in Production | 30 min | Build and deploy scalable inference services with Ray Serve. Learn model composition, multi-deployment architectures, FastAPI integration, and fractional GPU resource management. |


Each module builds on concepts from the previous one, but the notebooks include enough context to be used independently as well.

## Repository Structure

```
ray-workshop/
├── README.md                                # This file
├── requirements.txt                         # All Python dependencies
│
├── Module1/                                 # Scaling Python for AI Workloads (45 min)
│   ├── 01_Ray_Core_Tasks.ipynb             
│   ├── 02_Ray_Core_Actors.ipynb            
│   ├── README.md                         
│   └── extra/                               # Self-study material
│
├── Module2/                                 # Building Scalable Data Pipelines (45 min)
│   ├── 00_Introduction_Ray_Data.ipynb  
│   ├── 01_Multimodal_Data_Processing.ipynb 
│   ├── README.md                          
│   └── extra/                               # Self-study material
│
├── Module3/                                 # Distributed Training at Scale (45 min)
│   ├── 01_Distributed_training_with_Ray.ipynb    
│   ├── 02_FSDP2_RayTrain_Tutorial_LIVE.ipynb     
│   ├── README.md                          
│   └── extra/                               # Self-study material
│
└── Module4/                                 # Serving Models in Production (30 min)
    ├── 01_Intro_Serve.ipynb               
    ├── README.md                         
    └── extra/                               # Self-study material
```

## Getting Started

### Prerequisites

- **Python 3.9+** installed
- Basic Python programming experience (functions, classes, decorators)
- Basic familiarity with machine learning concepts (training, embeddings, inference)
- Experience with PyTorch or similar ML frameworks (helpful but not required)

### Installation

```bash
# Install all dependencies at once
pip install -r requirements.txt
```

## Extra Learning Material

Each module includes an `extra/` folder with supplementary notebooks for self-study after the workshop. These cover architecture deep dives, advanced patterns, optimization techniques, and production best practices that go beyond the guided workshop time.

## Resources

- [Ray Documentation](https://docs.ray.io/)
- [Ray GitHub Repository](https://github.com/ray-project/ray)
- [Anyscale Platform](https://www.anyscale.com/)
- [Ray Community Slack](https://ray-project.slack.com/)
- [Ray Tutorials and Examples](https://docs.ray.io/en/latest/ray-overview/examples.html)