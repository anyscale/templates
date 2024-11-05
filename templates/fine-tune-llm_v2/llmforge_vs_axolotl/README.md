# Benchmarking Axolotl vs LLMForge MFUs
| Model | LLMForge | Axolotl |
|---|---|---|
| Llama-3.1-8B LoRA MFU | **55.9%** | 52.7% |
| Llama-3.1-8B Full-FT MFU | **56.6%** | 55.1% |

Results above are for fine-tuning on the Alpaca-12K dataset with a sequence length of 2048, and a per device batch size of 2 on a 4xA100-80G node with DeepSpeed ZeRO-2. All model optimizations in both libraries are applied (custom optimizers for Axolotl, Liger Kernel for both LLMForge and Axolotl). Benchmarked on Axolotl commit df539, and LLMForge release 0.5.7.

## Reproducing Results
```python
git clone https://github.com/axolotl-ai-cloud/axolotl.git
cd axolotl/
git checkout df359c8a6e14ecdd2e1eb0049bd8143c32421952
pip3 install packaging ninja
pip3 install -e '.[flash-attn,deepspeed]'

# return to top level directory
cd ..

# get dataset from huggingface
python get_dataset.py

# Axolotl
## for lora
accelerate launch -m axolotl.cli.train axolotl_files/llama-3-lora.yaml

## for full finetuning
accelerate launch -m axolotl.cli.train axolotl_files/llama-3-full.yaml

# LLMForge
## for lora
llmforge anyscale finetune llmforge_files/llama-3-lora.yaml

## for full finetuning
llmforge anyscale finetune  llmforge_files/llama-3-full.yaml
```

## Getting FLOPs
For full finetuning, the standard estimate for TFLOPs is:

`model_size_in_B * 4 * 2 * seqlen * global_batch_size / (time_in_sec_per_interation * total_gpus * 1e3)`

where the factor of 4 (instead of 3) accounts for the fact that gradient checkpointing is usually used, requiring extra recomputation of activations.

However, this estimate is not entirely accurate for models trained with LoRA, where the backward computation bypasses the full set of parameters. The `calculate_flops.py` script helps get a more accurate estimate of the FLOPs for a forward + backward pass. 
### Llama-3.1-8B-Instruct Example
| Model | Full (TFLOP)| LoRA (TFLOP) |
|---|---|---|
| Llama-3.1-8B-Instruct | 199.8 | 138.7 |
| Llama-3.1-8B-Instruct + GC | 261.4 | 200.6 |

We can see that the heuristic for scaling flops w.r.t gradient checkpointing (the factor of 3 -> 4) is not entirely accurate (assuming we trust the pytorch flop calculator), with less extra flops introduced than expected for full finetuning, and relatively more extra flops induced for LoRA.

Resources: [Stas Bekman Training Perf Guide](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md)

## Getting Step Times
Relying on the tqdm step time is not a perfect proxy, but if we inspect the [HF Trainer code](https://github.com/huggingface/transformers/blob/eb811449a2389e48930c45f84c88fd041735cf92/src/transformers/trainer.py#L2448), there isn't any significant overhead in the training loop in either our code or theirs - since we want to be able to update this benchmark for future versions without manually patching a step timer into the HF Trainer code, let's just use the tqdm timers as an estimate for now (there's also 0.3 ema smoothing applied, but this is consistent across both our code and the HF Trainer code).

We also turn off model checkpointing features in both LLMForge and Axolotl (gradient checkpointing we keep on for both). Below we show the average step time (at the end of training) for Axolotl and LLMForge for full finetuning and LoRA training.

| Framework | LLMForge |  Axolotl|
|---|---|---|
| LoRA (s/it) | 1.15  | 1.22 |
| Full (s/it) | 1.48 |  1.52 |


