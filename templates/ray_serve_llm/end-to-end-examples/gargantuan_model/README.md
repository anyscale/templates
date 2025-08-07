# Deploy a 670 billion parameter reasoning model, DeepSeek R1

**⏱️ Time to complete**: 30 min

Deploying a 670B parameter model like DeepSeek R1 presents significant technical challenges. The model is too large to fit in a GPU, or even a single node. This requires distributing the model across multiple GPUs and nodes using *tensor parallelism*, AKA intra-layer parallelism, and *pipeline parallelism*, AKA inter-layer parallelism. The Ray Serve LLM API automates this process.

Deploying the model also involves launching multiple nodes manually and configuring them to work together. Anyscale automates this process by autoscaling the cluster with the appropriate number of nodes and GPUs.

Beware: this is an expensive deployment. At the time of writing, the deployment cost is around $110 USD per hour in the `us-west-2` AWS region.We recommend *disabling cross-zone autoscaling* because this deployment has a high amount of inter-node traffic, and cross-zone traffic is expensive (around $0.02 per GB). This demo is pre-configured with cross-zone autoscaling disabled for your convenience.

## Prerequisites

This template only works on H100 GPUs in your self-hosted Anyscale cloud-- H100 GPUs are not available in Anyscale's public cloud.

To launch nodes with 1000 GB disk capacity (instead of the default 150 GB), modify the **`Instance config`** field under **`Manage Cluster`** → **`Advanced settings`**.

For more information about configuring the disk size of a Google Cloud Platform (GCP) cluster, see [Changing the default disk size for GCP clusters](https://docs.anyscale.com/configuration/compute/gcp/#changing-the-default-disk-size).
For more information about configuring the disk size of an Amazon Web Services (AWS) cluster, see [Changing the default disk size for AWS clusters](https://docs.anyscale.com/configuration/compute/aws/#changing-the-default-disk-size).

In the case of AWS, the corresponding settings are:

```json

    {
      "BlockDeviceMappings": [
        {
          "Ebs": {
            "VolumeSize": 1000,
            "VolumeType": "gp3",
            "DeleteOnTermination": true
          },
          "DeviceName": "/dev/sda1"
        }
      ]
    }
```

With this configuration, every launched node has 1000 GB disk capacity. This change may require restarting the cluster with new nodes, which Anyscale automatically handles.

![Configuring worker nodes](../../assets/select-2x-H100.png)

## Start the deployment

The following code deploys the DeepSeek R1 model using Ray Serve:


```python
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

llm_config = LLMConfig(
    model_loading_config={
        "model_id": "big_model",  # Model ID for Ray Serve
        "model_source": "deepseek-ai/DeepSeek-R1",  # Model ID on Hugging Face
    },
    deployment_config={
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 1,
        }
    },
    
    # Accelerator type. With autoscaling enabled, Anyscale automatically
    # launches the appropriate instance type.
    accelerator_type="H100",
    # Enable the vLLM V1 core engine.
    runtime_env={"env_vars": {"VLLM_USE_V1": "1"}},
    
    # Options passed through to the vLLM engine.
    engine_kwargs={
        # Automatic model parallelization across GPUs
        # Used by the auto-scaler to select the appropriate instance type.
        # In this case, it selects machines with 4x L40S GPUs.
        "tensor_parallel_size": 8,   # Splits model layers across 8 GPUs per node
        "pipeline_parallel_size": 2,  # Distributes across 2 nodes
        # Total: 8 GPUs × 2 nodes = 16 GPUs
        
        "gpu_memory_utilization": 0.92,  # Use 92% of GPU memory
        "dtype": "auto",
        
        # Performance tuning
        "max_num_seqs": 40,  # Max concurrent requests
        "max_model_len": 16384,  # Max tokens per pass
        "enable_chunked_prefill": True,
        "enable_prefix_caching": True
    },
)

# Serve the model.
llm_app = build_openai_app({"llm_configs": [llm_config]})
serve.run(llm_app)
```

When you run the preceding code, Anyscale automatically provisions a cluster with the appropriate instance type and number of nodes.
After running the preceding code, monitor the progress of the deployment in the Anyscale Console.
You might encounter warnings about insufficient capacity in your cloud region. If you experience a GPU shortage, Anyscale continues to poll the cloud provider until enough capacity is available and launches all the nodes at once when possible.
Beware that due to high demand, on-demand H100 GPUs are often out of capacity in most cloud providers.

Because the model is so large, it takes 15-25 minutes to download the model weights and split the model across the nodes.

### Verify deployment

The output should look like:
```
INFO 2025-03-02 17:17:14,162 serve 61769 -- Application 'default' is ready at http://127.0.0.1:8000/
INFO 2025-03-02 17:17:14,162 serve 61769 -- Deployed app 'default' successfully.
```

DeepSeek R1 is running across multiple nodes, ready to serve requests.


The deployment provides a standard OpenAI API interface:



```python
from openai import OpenAI

# Connect to your deployed model
client = OpenAI(
    base_url="http://localhost:8000/v1",  # Ray Serve endpoint
    api_key="not-needed"  # No API key required for local deployment
)

# Use the same model ID from your configuration
model_id = "big_model"

# Basic chat completion with streaming
response = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain quantum mechanics in simple terms."}
    ],
    stream=True
)

# Process streaming response
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Observability and monitoring

To monitor metrics, access the **`Ray Dashboard`** for the following:
- Real-time GPU utilization
- Request latency metrics
- Queries per second (QPS)
- Error rates and logs


## Cleanup

When finished, gracefully shut down the service:


```python
serve.shutdown()

# Or from command line
# serve shutdown --yes
```

Anyscale automatically detects idle nodes and scales down the cluster.

## Conclusion

Ray Serve LLM API simplifies deploying massive language models using a few lines of Python code. With production-ready scaling and fault tolerance features, you can focus on building applications rather than managing infrastructure.

To learn more, see the [Ray Serve LLM documentation](https://docs.ray.io/en/latest/serve/llm/serving-llms.html).
