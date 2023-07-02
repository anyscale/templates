# Fine-Tuning LLMs on Anyscale with DeepSpeed

In this application you will fine tune an LLM - GPTJ. GPT-J is a GPT-2-like causal language model trained on the Pile dataset. This particular model has 6 billion parameters. For more information on GPT-J, click [here](https://huggingface.co/docs/transformers/model_doc/gptj).

The application can be used by developers and datascientists alike.  Developers can leverage simple APIs to run fine tuning jobs on the cluster while data scientists and machine learning engineers can dive in deeper to view the underlying code using familiar tools like JupyterLab notebooks or Visual Studio Code. **These tools enable you to easily adapt this example to use other similar models or your own data**.

## Using this application
You can use the application in one of two ways:

1. Via the CLI.  Navigate to the "terminal" once started and run the following command:
```
anyscale job submit -- python gptj_deepspeed_fine_tuning.py
```
Once submitted, you can navigate to the Job page and view the training progress with the Ray Dashboard. 
![Ray Dashboard](https://github.com/anyscale/templates/releases/download/media/raydash.png)

2. By using the interactive notebook which will go step by step through the process of loading data, tokenizing and splitting the data, loading the model, and performing fine tuning.  You can follow along in VSCode or Jupyter!   Note: You can also use the CLI command directly from the interactive notebook.  Take a look under the "Submitting an Anyscale Job" section of the notebook.
![IDES](https://github.com/anyscale/templates/releases/download/media/ides.png)

### Next Steps

#### Training on your own data: Modifying the Script 
Once your application is ready and launched you may view the script with VSCode or Jupyter and modify to use your own data!  Read more about loading data with Ray [from your file store or database here](https://docs.ray.io/en/latest/data/loading-data.html).  Make sure the data you use has a similar structure to the [Shakespeare dataset we use.](https://huggingface.co/datasets/tiny_shakespeare)

The lines of code to update look like the following:
![Code](https://github.com/anyscale/templates/releases/download/media/code.png)

Once the code is updated, run the same command as before:
```
anyscale job submit -- python gptj_deepspeed_fine_tuning.py
```

## Serving your model
The fine tuning job saves checkpoints during training in your [default mounted user storage](https://docs.anyscale.com/develop/workspaces/storage#user-storage).  See the "Serving" Application to see how you can now deploy and serve this model for production traffic with [Anyscale Production Services](https://docs.anyscale.com/productionize/services/get-started).  

Within 2 minutes you will be fine-tuning GPT-J on a corpus of Shakspeare data!  Let's dive in and explore the power of Anyscale and Ray together.


## Appendix

### Advanced - Workspaces and Configurations
This application makes use of [Anyscale Workspaces](https://docs.anyscale.com/develop/workspaces/get-started) and Ray AIR (with the 🤗 Transformers integration) to fine-tune an LLM. Workspace is a fully managed development environment focused on developer productivity. With workspaces, ML practitioners and ML platform developers can quickly build distributed Ray applications and advance from research to development to production easily, all within single environment.

To run this example, we've set up your Anyscale Workspace to have access to a head node with one GPU with 16 or more GBs of memory and 15 g4dn.4xlarge instances for the worker node group. This is done by defining a "compute configuration".  Learn more about [Compute Configs here] (https://docs.anyscale.com/configure/compute-configs/overview).  It is easy to change your Compute Config once you launch by clicking "Workspace" and Editing the selection.  
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