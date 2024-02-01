# Serving a Stable Diffusion Model with Ray Serve
This template shows you how to develop and test the model locally and deploy it into production.

## Step 1: Install python dependencies
```
pip install --user diffusers==0.25.0 transformers==4.36.2 accelerate==0.25.0
```

## Step 2: Deploy the model locally
- Open a new terminal. 
- Run the command below to deploy your model at http://localhost:8000.  

This template uses an A10G GPU by default. You can update the `accelerator_type` config in `main.py` to use the GPU desired. Note that A10G is not available on GCP and you would need to switch to L4.

```bash
serve run main:stable_diffusion_app
```

## Step 3: Send test requests to the running model
- Open a new terminal. Run the command below to send a request to your model. 
- An image should be generated in the current directory
```bash
python query.py
```

## Step 4: Deploy the model as an Anyscale Service
Deploy the model to production using the `anyscale service rollout` command.

This creates a long-running [service](https://docs.anyscale.com/services/get-started) with a stable endpoint to query the application.

Note that we installed some pip packages in the workspace that had to be added in to the runtime environment. For faster setup of your deployments, you can build a new [cluster environment](https://docs.anyscale.com/configure/dependency-management/cluster-environments) with these packages.

```bash
anyscale service rollout -f service.yaml --name {ENTER_NAME_FOR_SERVICE}
```

## Step 5: Query your Anyscale Service
Query the service using the same `query.py` script as when testing it locally, with two changes:
1. Update the `HOST` to the service endpoint.
2. Update the `TOKEN` field to add the authorization token as a header in the HTTP request.

Both of these values are printed when you run `anyscale service rollout`. You can also find them on the service page. For example, if the output looks like:
```bash
(anyscale +4.0s) You can query the service endpoint using the curl request below:
(anyscale +4.0s) curl -H 'Authorization: Bearer 26hTWi2kZwEz0Tdi1_CKRep4NLXbuuaSTDb3WMXK9DM' https://stable_diffusion_app-4rq8m.cld-ltw6mi8dxaebc3yf.s.anyscaleuserdata-staging.com
```

Then:
- The authorization token is `26hTWi2kZwEz0Tdi1_CKRep4NLXbuuaSTDb3WMXK9DM`.
- The service endpoint is `https://stable_diffusion_app-4rq8m.cld-ltw6mi8dxaebc3yf.s.anyscaleuserdata-staging.com`.

In the services UI, click the **Query** button on the top-right side to get these two fields

![deploy-pop-up](https://github.com/anyscale/templates/blob/main/templates/serve-stable-diffusion-aica/assets/query_instructions.png?raw=true)

After updating these fields, run:
```bash
python query.py
```

Another way to query is through the FastAPI UI linked through the API docs on the Service page: 
![deploy-pop-up](https://github.com/anyscale/templates/blob/main/templates/serve-stable-diffusion-aica/assets/fastapi_docs.png?raw=true)


## (Optional) Step 6: Iterate in the Workspace and update the Service without downtime
You can make code changes in the same Workspace and deploy an upgrade to your Service without downtime using the same command.

```bash
anyscale service rollout -f service.yaml --name {ENTER_NAME_FOR_SERVICE}
```

