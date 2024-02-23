# Launching LoRA fine-tuning

You can utilize [LoRA](https://arxiv.org/abs/2106.09685) to achieve more resource efficient fine-tuning results than full-parameter fine-tuning, but unlocking smaller instance types and more effecient model serving.
To launch a LoRA fine-tuning, you can use the following command or similar commands for other model sizes:

```
python train.py --size=7b --lora
```

Fine-tuning a model with LoRA results in a checkpoint containing only the fine-tuned weights.
As an example, the default Llama 2 LoRA configuration should yield a 42/64/202MB checkpoint for 7B/13B/70B models.
If we want to evaluate the model after training, we can merge the model weights with the original (non-fine-tuned) model.
We provide a script to merge the fine-tuned weights with the original weights to produce a full-parameter checkpoint.
The script has high CPU memory requirements because it requires us to load all parameters into memory at the same time,
13GB/24GB/152GB for 7B/13B/70B models. Downloading and loading the original weights should take ~1min/~2min/~10min each
on a p4de.24xlarge instance. You can run the script as follows:

```
python merge_lora_weights.py --model-name=7b --checkpoint=<path to your checkpoint> --output-path=<desired output path>
```

This leaves a self-contained LoRA fine-tuned model, config and tokenizer at the desired output path.
