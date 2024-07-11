# Quantization 

Quantization is a technique to reduce the computational and memory costs of running inference by representing the weights and/or activations with low-precision data types like 4-bit integer (int4) instead of the usual 16-bit floating point (float16).
Quantization allows users to deploy models with cheaper hardware requirements with potentially lower inference costs. 

Anyscale supports AWQ and SqueezeLLM weight-only quantization by integrating with [vLLM](https://github.com/vllm-project/vllm). Quantization can be enabled by specifying the `quantization` method in `engine_kwargs` and using a quantized model for `model_id` and `hf_model_id`. See the configs in this directory for quantization examples. Note that the AWQ and SqueezeLLM quantization methods in vLLM have not been fully optimized and can be slower than FP16 models for larger batch sizes. 