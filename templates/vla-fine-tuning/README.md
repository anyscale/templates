# Distributed Vision-Language-Action (VLA) Model Fine-tuning with Ray

This template implements an end-to-end distributed fine-tuning pipeline for the [PI0.5](https://www.physicalintelligence.company/blog/pi05) Vision-Language-Action (VLA) model on the [Droid v1.0.1](https://droid-dataset.github.io/) robot manipulation dataset using [Ray](https://docs.ray.io/) on [Anyscale](https://anyscale.com).

It leverages Ray's ability to independently scale two distinct compute tiers — CPU nodes for streaming data preprocessing and GPU nodes for distributed training — so that neither side sits idle and GPU stalls waiting for data are eliminated.

![VLA Fine-tuning Pipeline](images/vla-fine-tuning-pipeline.png)

For the full walkthrough, open **[vla.ipynb](vla.ipynb)**.
