# Building distributed ML systems with Ray Core: Principles & Patterns - Ray Summit 2025

## Training Agenda

### Part 1: Ray Tasks

#### 1. Ray Tasks in Practice (`01_Ray_Tasks_in_Practice.ipynb`)
- Ray Core overview and fundamentals
- Creating and executing remote functions with `@ray.remote`
- Working with ObjectRefs and `ray.get()`
- Object store and passing data by reference
- Task chaining and dependencies
- Task retries and exception handling
- Runtime environments (dependencies, environment variables)
- Resource allocation (CPU, GPU, fractional resources)
- Nested tasks
- Pipeline data processing with `ray.wait()`
- Common anti-patterns and best practices

#### 2. Streaming Operator Execution (`02_Streaming_Operator_Execution.ipynb`)
- Building streaming pipelines with generators and iterators
- Backpressure and flow control
- Resource management and sharing

### Part 2: Ray Actors

#### 3. Ray Actors in Detail (`03_Ray_Actors_in_Detail.ipynb`)
- Introduction to stateful computation
- Creating and managing actors
- Actor handles and RPC semantics
- Actor lifecycle and resource management
- Actor patterns and best practices

#### 4. Example: Distributed Training (`04_Distributed_Training.ipynb`)
- Implementing distributed machine learning with actors
- Distributed worker process group pattern
- State management across workers
- Scaling training workloads

---
© 2025, Anyscale. All Rights Reserved.