# Serving a Stable Diffusion Model with Ray Serve
This template shows you how to develop and test the model locally and deploy it into production.

## Step 1: Deploy the model locally
- Open a terminal (ctl+shift+`) in VS Code. 
- Run the command below to deploy your model at http://localhost:8000
```bash
serve run app:entrypoint
```

## Step 2: Send test requests to the running model
- Open a new terminal. Run the command below to send a request to your model. 
- An image should be generated in the current directory (it takes 1-2 mins to load the model and generate the first image. ~10s for to generate following images)
```bash
python query.py
```

(Note for internal testers: the following features are not available yet)

## Step 3: Deploy the model as an Anyscale Service
Deploy it as an Anyscale Service for staging or production traffic with `--publish` flag

```bash
serve run app:entrypoint --publish
```


## Step 4: Query your Anyscale Service
Navigate to Service UI with the URL generated from the previous step, click **Query** butoton to get detailed query instructions and intregate it into your own app.

![deploy-pop-up](./assets/query_instructions.png)

The benefits of using Anyscale Services for staging/production traffic:
- Zero-downtime upgrade: iterate in the same workspace and perform rolling upgrade (deploy it with the same command) to your Service.
- better fault tolerence (auto-recover from node failures, etc.).
