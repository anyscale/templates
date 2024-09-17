# Intro to Fine-tuning Open-weight LLMs with Anyscale

**⏱️ Time to complete**: ~3 hours (includes the time for training the model)


This template comes with a installed library for training LLMs on Anyscale called [LLMForge](https://docs.anyscale.com/reference/llmforge-versions). It provides the fastest way to try out training LLMs with Ray on Anyscale. You can read more about this library and its features in the [docs](https://docs.anyscale.com/llms/finetuning/intro). For learning on how to serve the model online or offline for doing batch inference you can refer to the [serving template](https://console.anyscale.com/v2/template-preview/endpoints_v2) or the [offline batch inference template](https://console.anyscale.com/v2/template-preview/batch-llm), respecitvely.


## Getting Started

You can find some tested config files examples in the `training_configs` directory. LLMForge comes with a CLI that lets you pass in a config YAML file to start your training.

Then you can launch fine tuning by running the following command:
```bash
llmforge anyscale finetune training_configs/custom/meta-llama/Meta-Llama-3-8B/lora/4xA10-512.yaml
```

This code will run LoRA fine-tuning on the Meta-Llama-3-8B-Instruct model with 4xA10-512 configuration on a GSM-8k math dataset.

When the training is done, you will see a message like this:

```bash
Note: LoRA weights will also be stored in path <path>
````

This is the path where the adapted weights are stored, you can use them for inference. You can also see the list of your fine-tuned models in the `serving` tab in the Anyscale console.

LLMForge also supports logging metrics externally with WandB and MLflow. A user guide for setting up these logging integrations can be found in the [docs](https://docs.anyscale.com/llms/finetuning/guides/logging_integrations)

# What is Next?

* Make sure to checkout the [LLMForge documentation](https://docs.anyscale.com/llms/finetuning/intro) and [user guides](https://docs.anyscale.com/category/fine-tuning-beta) for more information on how to use the library.
* You can also check out more [end-to-end examples](#end-to-end-examples).
* You can follow the [serving template](https://console.anyscale.com/v2/template-preview/endpoints_v2) to learn how to serve the model online.
* You can follow the [offline batch inference template](https://console.anyscale.com/v2/template-preview/batch-llm) to learn how to do batch inference.



## End-to-end Examples

Here is a list of end-to-end examples that involve more steps such as data preprocessing, evaluation, etc but with a main focus on improving model quality via fine-tuning.

* [Fine-tuning for Function calling on custom data](./end-to-end-examples/fine-tune-function-calling/README.ipynb): This example demonstrates how to fine-tune a model on a custom dataset for function calling task.
* [Preference tuning for summarization](./end-to-end-examples/fine-tune-preference/README.ipynb): An example of Direct Preference Optimization (DPO) fine-tuning for summarization.
