# Quickstart: Stable diffusion pre-training

<img src="https://anyscale-materials.s3.us-west-2.amazonaws.com/stable-diffusion/end_to_end_architecture_v6.jpeg" width=1000px />

Above is the reference architecture for Stable Diffusion pre-training
with Ray and Anyscale. You can view how to implement the architecture by
checking the provided python scripts. You can run interactive notebooks
to guide you through the major steps in the process.

Here’s what you can achieve with this reference implementation:

-   Pre-train the Stable Diffusion v2 model on a massive dataset of ~2
    billion images for less than \$40,000.
-   Eliminate preprocessing bottlenecks with Ray Data and improve
    training throughput by 30%.
-   Benefit from system and algorithm optimizations to reduce training
    costs by 3x compared to baseline methods.

To view a detailed benchmarking and explanation of our reference implementation, check out our [blog post here](https://www.anyscale.com/blog/stable-diffusion-pre-training).

## Interactive notebooks

Dive into these interactive notebooks to see how you can implement
Stable Diffusion v2 pre-training on Anyscale:

| Notebook                                                                             | Description                                                                                                                               | Input                                | Output                               | Time to complete  |
|---------------|---------------|---------------|---------------|---------------|
| 01_preprocess.ipynb                                         | Run a scalable data pipeline to process image and text data for Stable Diffusion pre-training.                                            | Image and caption data               | Image latents and caption embeddings | 🕙 5 minutes |
| 02_train.ipynb                                                   | Initiate a scalable training pipeline to efficiently produce Stable Diffusion v2 models.                                                  | Image latents and caption embeddings | Trained model                        | 🕙 5 minutes |
| 03_end_to_end.ipynb | Create and run a scalable online processing and training pipeline for more complex training scenarios running on heterogeneous resources. | Image and caption data               | Trained model                        | 🕙 5 minutes |


Note that the notebooks are designed to be quick demonstrations and as such run against a very small subset of the data, making use of smaller-sized models and smaller sized GPU instances (NIVIDIA A10 GPU).

## Want to pre-train with custom data? 📈

If you’re looking to scale your Stable Diffusion pre-training with
custom data, we’re here to help 🙌 !

👉 **[Check out this link](https://forms.gle/9aDkqAqobBctxxMa8) so we
can assist you**.

## Key benefits of using Anyscale

With Anyscale, you can significantly improve your pre-training setup,
given Anyscale’s unique features like:

-   **Heterogeneous Compute**: Easily scale your training across
    different machines, GPUs, and accelerators.
-   **Cost Efficient**: Use automatic on-demand to spot instances
    switching and fast nodes startup to lower compute costs.
-   **Smooth DevEx**: Develop in VSCode, monitor hardware usage in real
    time, and move from development to production seamlessly.

## Detailed cost analysis

For those who love the details, here’s a deep dive into our costs for
the full scale pre-training Stable Diffusion v2. We reduced training costs by 3x
compared to baseline methods.

| Parameter           | Value                                                                                                      |
|------------------------------------|------------------------------------|
| Instance Type       | p4de.24xlarge                                                                                              |
| Cloud Provider      | AWS (us-west-2)                                                                                            |
| GPU Type            | A100-80G                                                                                                   |
| Global Batch Size   | 4096                                                                                                       |
| Training Procedure  | Phase 1: 1,126,400,000 samples at resolution 256x256; Phase 2: 1,740,800,000 samples at resolution 512x512 |
| Total A100 Hours    | 13,165                                                                                                     |
| Total Training Cost | \$39,511 (1-yr reservation instances) \$67,405 (on-demand instances)                                       |
