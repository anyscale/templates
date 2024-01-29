# Optimize Models

We have provided various model configurations for different accelerator types and tensor parallelism (or tp). The supported accelerator types are:
T4, L4, A10G, A100-40G and A100-80G.

Tensor parallelism is a type of model parallelism in which specific model weights, gradients, and optimizer states are split across devices. This typically involves distributed computation of specific operations, modules, or layers of the model. 


## Configurations to optimize

These are some configurations you should consider changing when updating tensor parallelism or the accelerator type:
1. The `num_workers` configuration can be used to set the tensor parallelism for a model. A higher value for tensor parallelism will typically lead to lower latency at the cost of more GPUs per replica.
2. `resources_per_worker` in `scaling_config` and `resources` under `ray_actor_options` in the `deployment_config`. These determine the accelerator type used. It must follow the format of `"accelerator_type:T4":0.01`.
3. `engine_kwargs`: `max_num_batched_tokens` and `max_num_seqs`. These are the maximum number of batched tokens and sequences configured for each iteration in [vLLM](https://docs.vllm.ai/en/latest/models/engine_args.html). With increase in available GPU memory, you can increase these values. 
4. `autoscaling_config`: `max_concurrent_queries` - the maximum number of queries that will be handled concurrently by each replica of the model (should be set equal to `max_num_seqs`) and `target_num_ongoing_requests_per_replica` - the number of ongoing requests per replica that will trigger auto-scaling. Similar to the arguments above, these can be increased as the GPU memory changes.

You may consider optimizing for either latency or throughput. The per-request latency generally degrades as the number of concurrent requests increase. The provided configurations generally optimize for latency. We recommend starting with one of our configurations and running load tests if you would like to tune any of the above parameters. 

Note - You can only run one configuration for a model id at a time.
