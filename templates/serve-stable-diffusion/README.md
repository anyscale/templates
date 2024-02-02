# Serving a Stable Diffusion Model with Ray Serve
This template shows you how to develop and test the model locally and deploy it into production.

## Step 1: Install python dependencies
```
pip install diffusers==0.25.0 transformers==4.36.2 accelerate==0.25.0
```

## Step 2: Deploy the model locally
- Open a new terminal (ctl+shift+`) in VS Code. 
- Run the command below to deploy your model at http://localhost:8000.  

```bash
serve run main:stable_diffusion_app
```

## Step 3: Send test requests to the running model
- Open a new terminal. Run the command below to send a request to your model. 
- An image should be generated in the current directory
```bash
python query.py
```

--------
##  Note: the following steps are still under implementation. We'll send a notice after they are ready for test.
--------

## Step 4: Deploy the model as an Anyscale Service
Deploy it as an Anyscale Service for staging or production traffic with `--publish` flag

```bash
serve run main:stable_diffusion_app --publish
```


## Step 5: Query your Anyscale Service
Navigate to Service UI with the URL generated from the previous step, click **Query** button to get detailed query instructions and integrate it into your own app.

![deploy-pop-up](https://github.com/anyscale/templates/blob/main/templates/serve-stable-diffusion/assets/query_instructions.png?raw=true)

The benefits of using Anyscale Services for staging/production traffic:
- Zero-downtime upgrade
- Better fault tolerence (auto-recover from node failures, etc.)


## (Optional) Step 6: Iterate in the Workspace and update the Service without downtime
You can make code changes in the same Workspace and deploy an upgrade to your Service without downtime using the same command.

```bash
serve run main:stable_diffusion_app --publish
```

