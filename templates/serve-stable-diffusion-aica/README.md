# Serving a Stable Diffusion Model with Ray Serve
This template shows you how to develop and test the model locally and deploy it into production.

## Step 1: Deploy the model locally
- Open a new terminal (ctl+shift+`) in VS Code. 
- Run the command below to deploy your model at http://localhost:8000.  

```bash
serve run main:stable_diffusion_app
```

## Step 2: Send test requests to the running model
- Open a new terminal. Run the command below to send a request to your model. 
- An image should be generated in the current directory
```bash
python query.py
```

## Step 3: Deploy the model as an Anyscale Service
Deploy it behind a stable endpoint as an Anyscale Service for staging or production traffic with the `anyscale` CLI.

```bash
anyscale service rollout -f service.yaml --name {ENTER_NAME_FOR_SERVICE}
```

## Step 4: Query your Anyscale Service
Navigate to Service UI with the URL generated from the previous step, click **Query** button to get detailed query instructions. For querying, update the `HOST` and `TOKEN` in `query.py` based on the values found in the **Query** panel.

![deploy-pop-up](https://github.com/anyscale/templates/blob/main/templates/serve-stable-diffusion-aica/assets/query_instructions.png?raw=true)

You can also use the FastAPI UI through the API docs. 
![deploy-pop-up](https://github.com/anyscale/templates/blob/main/templates/serve-stable-diffusion-aica/assets/fastapi_docs.png?raw=true)


The benefits of using Anyscale Services for staging/production traffic:
- Zero-downtime upgrade
- Better fault tolerence (auto-recover from node failures, etc.)
- Observability features


## (Optional) Step 5: Iterate in the Workspace and update the Service without downtime
You can make code changes in the same Workspace and deploy an upgrade to your Service without downtime using the same command.

```bash
anyscale service rollout -f service.yaml --name {ENTER_NAME_FOR_SERVICE}
```

