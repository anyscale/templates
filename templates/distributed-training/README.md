# Fine-Tuning LLMs on Anyscale with DeepSpeed

In this application you will fine tune an LLM - GPTJ. GPT-J is a GPT-2-like causal language model trained on the Pile dataset. This particular model has 6 billion parameters. For more information on GPT-J, click [here](https://huggingface.co/docs/transformers/model_doc/gptj).

The application can be used by developers and datascientists alike.  Developers can leverage simple APIs to run fine tuning jobs on the cluster while data scientists and machine learning engineers can dive in deeper to view the underlying code using familiar tools like JupyterLab notebooks or Visual Studio Code. **These tools enable you to easily adapt this example to use other similar models or your own data**.

| App Details | Description |
| ---------------------- | ----------- |
| Summary | This app loads a pretrained GPTJ model from HuggingFace and fine tunes it on new text data.  |
| Time to Run | Around 20-40 minutes to fine tune on all of the data. |
| Minimum Compute Requirements | At least 1 GPU node. The default is 1 node (the head), and up to 15 worker nodes each with 1 NVIDIA T4 GPU. |
| Cluster Environment | This template uses a docker image built on top of the latest Anyscale-provided Ray image using Python 3.10: [`anyscale/ray:latest-py310-cu118`](https://docs.anyscale.com/reference/base-images/overview). See the appendix below for more details. |

## Using this application
You can use the application via the CLI.  Navigate to the "terminal" once started and run the following command:
```
anyscale job submit -- python gptj_deepspeed_fine_tuning.py
```
Once submitted, you can navigate to the Job page and view the training progress with the Ray Dashboard. 
![Ray Dashboard](https://github.com/anyscale/templates/releases/download/media/raydash.png)

Note: This application is based on an example.  If you wish to go step by step and learn more please visit the [Ray docs tutorial](https://docs.ray.io/en/latest/ray-air/examples/gptj_deepspeed_fine_tuning.html)  

### Next Steps

#### Training on your own data: Modifying the Script 
Once your application is ready and launched you may view the script with VSCode or Jupyter and modify to use your own data!  Read more about loading data with Ray [from your file store or database here](https://docs.ray.io/en/latest/data/loading-data.html).  Make sure the data you use has a similar structure to the [Shakespeare dataset we use.](https://huggingface.co/datasets/tiny_shakespeare)

Modify the code under the 'loading data' section of the script to load your own fine-tuning dataset.

Once the code is updated, run the same command as before:
```
anyscale job submit -- python gptj_deepspeed_fine_tuning.py
```

## Saving your model
The fine tuning job automatically saves checkpoints during training in your [default mounted user storage](https://docs.anyscale.com/develop/workspaces/storage#user-storage).  You can view the model by navigating to "Files" viewer and selecting "User Storage".
![Files](https://github.com/anyscale/templates/releases/download/media/files.png)

Within 2 minutes you will be fine-tuning GPT-J on a corpus of Shakspeare data!  Let's dive in and explore the power of Anyscale and Ray together.


## Appendix

### Advanced - Workspaces and Configurations
This application makes use of [Anyscale Workspaces](https://docs.anyscale.com/develop/workspaces/get-started) and Ray AIR (with the 🤗 Transformers integration) to fine-tune an LLM. Workspace is a fully managed development environment focused on developer productivity. With workspaces, ML practitioners and ML platform developers can quickly build distributed Ray applications and advance from research to development to production easily, all within single environment.

To run this example, we've set up your Anyscale Workspace to have access to a head node with one GPU with 16 or more GBs of memory and 15 g4dn.4xlarge instances for the worker node group. This is done by defining a "compute configuration".  Learn more about [Compute Configs here](https://docs.anyscale.com/configure/compute-configs/overview).  It is easy to change your Compute Config once you launch by clicking "Workspace" and Editing the selection.  
![Config](https://github.com/anyscale/templates/releases/download/media/edit.png)


When you run the fine tuning job we execute a python script thats distributed with Ray as an [Anyscale Job](https://docs.anyscale.com/productionize/jobs/get-started).   

### Advanced: Build off of this template's cluster environment
#### Option 1: Build a new cluster environment on Anyscale
You'll find a cluster_env.yaml file in the working directory of the template. Feel free to modify this to include more requirements, then follow [this](https://docs.anyscale.com/configure/dependency-management/cluster-environments#creating-a-cluster-environment) guide to use the Anyscale CLI to create a new cluster environment.

Finally, update your workspace's cluster environment to this new one after it's done building.

#### Option 2: Build a new docker image with your own infrastructure
Use the following docker pull command if you want to manually build a new Docker image based off of this one.

```bash
docker pull us-docker.pkg.dev/anyscale-workspace-templates/workspace-templates/fine-tune-gptj:latest
```