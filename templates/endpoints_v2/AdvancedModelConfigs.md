# TODO
# RayLLM model registry

Each model is defined by a YAML configuration file in the `models` directory.

## Modify an existing model

To modify an existing model, simply edit the YAML file for that model.
Each config file consists of three sections: 

- `deployment_config`, 
- `engine_config`, 
- `scaling_config`.

It's best to check out examples of existing models to see how they are configured.

## Deployment config

The `deployment_config` section corresponds to
[Ray Serve configuration](https://docs.ray.io/en/latest/serve/production-guide/config.html)
and specifies how to [auto-scale the model](https://docs.ray.io/en/latest/serve/scaling-and-resource-allocation.html)
(via `autoscaling_config`) and what specific options you may need for your
Ray Actors during deployments (using `ray_actor_options`). We recommend using the values from our sample configuration files for `metrics_interval_s`, `look_back_period_s`, `smoothing_factor`, `downscale_delay_s` and `upscale_delay_s`. These are the configuration options you may want to modify:

* `min_replicas`, `initial_replicas`, `max_replicas` - Minimum, initial and maximum number of replicas of the model to deploy on your Ray cluster.
* `max_concurrent_queries` - Maximum number of queries that a Ray Serve replica can process at a time. Additional queries are queued at the proxy.
* `target_num_ongoing_requests_per_replica` - Guides the auto-scaling behavior. If the average number of ongoing requests across replicas is above this number, Ray Serve attempts to scale up the number of replicas, and vice-versa for downscaling. We typically set this to ~40% of the `max_concurrent_queries`.
* `ray_actor_options` - Similar to the `resources_per_worker` configuration in the `scaling_config`. Refer to the `scaling_config` section for more guidance.

## Engine config

Engine is the abstraction for interacting with a model. It is responsible for scheduling and running the model inside a Ray Actor worker group.

The `engine_config` section specifies the Hugging Face model ID (`model_id`), how to initialize it and what parameters to use when generating tokens with an LLM.

RayLLM supports continuous batching, meaning incoming requests are processed as soon as they arrive, and can be added to batches that are already being processed. This means that the model is not slowed down by certain sentences taking longer to generate than others. RayLLM also supports quantization, meaning compressed models can be deployed with cheaper hardware requirements. For more details on using quantized models in RayLLM, see the [quantization guide](quantization/README.md).

* `model_id` is the ID that refers to the model in the RayLLM or OpenAI API.
* `type` is the type of  inference engine. Only `VLLMEngine` is currently supported.
* `engine_kwargs` and `max_total_tokens` are configuration options for the inference engine (e.g. gpu memory utilization, quantization, max number of concurrent sequences). These options may vary depending on the hardware accelerator type and model size. We have tuned the parameters in the configuration files included in RayLLM for you to use as reference. 
* `generation` contains configurations related to default generation parameters such as `prompt_format` and `stopping_sequences`.
* `hf_model_id` is the Hugging Face model ID. This can also be a path to a local directory. If not specified, defaults to `model_id`.
* `runtime_env` is a dictionary that contains Ray runtime environment configuration. It allows you to set per-model pip packages and environment variables. See [Ray documentation on Runtime Environments](https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#runtime-environments) for more information.
* `s3_mirror_config` is a dictionary that contains configuration for loading the model from S3 instead of Hugging Face Hub. You can use this to speed up downloads.
* `gcs_mirror_config` is a dictionary that contains configuration for loading the model from Google Cloud Storage instead of Hugging Face Hub. You can use this to speed up downloads.

## Scaling config

Finally, the `scaling_config` section specifies what resources should be used to serve the model - this corresponds to Ray AIR [ScalingConfig](https://docs.ray.io/en/latest/train/api/doc/ray.train.ScalingConfig.html). Note that the `scaling_config` applies to each model replica, and not the entire model deployment (in other words, each replica will have `num_workers` workers).

* `num_workers` - Number of workers (i.e. Ray Actors) for each replica of the model. This controls the tensor parallelism for the model.
* `num_gpus_per_worker` - Number of GPUs to be allocated per worker. Typically, this should be 1. 
* `num_cpus_per_worker` - Number of CPUs to be allocated per worker. 
* `placement_strategy` - Ray supports different [placement strategies](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html#placement-strategy) for guiding the physical distribution of workers. To ensure all workers are on the same node, use "STRICT_PACK".
* `resources_per_worker` - we use `resources_per_worker` to set [Ray custom resources](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#id1) and place the models on specific node types. The node resources are set in node definitions. Here are some node setup examples while using [KubeRay](https://github.com/ray-project/ray-llm/tree/master/docs/kuberay) or [Ray Clusters](https://github.com/ray-project/ray-llm/blob/master/deploy/ray/rayllm-cluster.yaml#L35). If you're deploying locally, please refer to this [guide](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#specifying-node-resources). An example configuration of `resources_per_worker` involves setting `accelerator_type_a10`: 0.01 for a Llama-2-7b model to be deployed on an A10 GPU. Note the small fraction here (0.01). The `num_gpus_per_worker` configuration along with number of GPUs available on the node will help limit the actual number of workers that Ray schedules on the node. 

If you need to learn more about a specific configuration option, or need to add a new one, don't hesitate to reach out to the team.


## How do I deploy multiple models at once?

You can append another application configuration to the YAML in `serve.yaml` file. Alternatively, you can use the CLI linked above.

## How do I deploy a model to multiple nodes?

All our default model configurations enforce a model to be deployed on one node for high performance. However, you can easily change this if you want to deploy a model across nodes for lower cost or GPU availability. In order to do that, go to the YAML file in the model registry and change `placement_strategy` to `PACK` instead of `STRICT_PACK`.

## How can I configure the resources / instances being used or the scaling behavior of my service?

You can edit the Compute Configuration direclty on your Workspace.  [Compute configurations](https://docs.anyscale.com/configure/compute-configs/overview) define the shape of the cluster and what resources Anyscale will use to deploy models and serve traffic.  If you would like to edit the default compute configuration choose "Edit" on your workspace and update the configuration.  When moving to production and deploying as an Ansycale Service the new configuration will be used.

Note that certain models require special accelerators.  Be aware that updating the resources make cause issues with your application.  

## My deployment isn't starting/working correctly, how can I debug?

There can be several reasons for the deployment not starting or not working correctly. Here are some things to check:
1. You might have specified an invalid model id.
2. Your model may require resources that are not available on the cluster. A common issue is that the model requires Ray custom resources (eg. `accelerator_type_a10`) in order to be scheduled on the right node type, while your cluster is missing those custom resources. You can either modify the model configuration to remove those custom resources or better yet, add them to the node configuration of your Ray cluster. You can debug this issue by looking at Ray Autoscaler logs ([monitor.log](https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html#system-component-logs)).
3. Your model is a gated Hugging Face model (eg. meta-llama). In that case, you need to set the `HUGGING_FACE_HUB_TOKEN` environment variable cluster-wide. You can do that either in the Ray cluster configuration or by setting it before running `serve run`.
4. Your model may be running out of memory. You can usually spot this issue by looking for keywords related to "CUDA", "memory" and "NCCL" in the replica logs or `serve run` output. In that case, consider reducing the `max_batch_prefill_tokens` and `max_batch_total_tokens` (if applicable). See models/README.md for more information on those parameters.

In general, [Ray Dashboard](https://docs.ray.io/en/latest/serve/monitoring.html#ray-dashboard) is a useful debugging tool, letting you monitor your application and access Ray logs.
