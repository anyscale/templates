# Introduction to Services
This tutorial shows you how to:
1. Develop a simple Ray Serve application locally.
2. Deploy the application to production as an Anyscale service.
3. Monitor the production application.
4. Configure service scaling.

**Note**: This tutorial is run within a workspace. Please overview the `Introduction to Workspaces` template first before this tutorial.

## Develop a Serve app locally

 The fastest way to develop a Ray Serve app is locally within the workspace. A Serve app running within a workspace behaves identically to a Serve app running as a production service, only it does not have a stable DNS name or fault tolerance.

 To get started, create a file called `main.py` and fill it with the following skeleton code:

```python
import requests
from fastapi import FastAPI
from ray import serve

fastapi = FastAPI()

@serve.deployment
@serve.ingress(fastapi)
class FastAPIDeployment:
    # FastAPI will automatically parse the HTTP request for us.
    # Check out https://docs.ray.io/en/latest/serve/http-guide.html
    @fastapi.get("/hello")
    def say_hello(self, name: str) -> str:
        return f"Hello {name}!"

my_app = FastAPIDeployment.bind()

### Run the app locally
Run the command below to run the serve app locally on `localhost:8000`.

If you want to deploy again, just run the command again to update the deployment.

**Tip**: to more easily view Serve backend logs, you may find it convenient to use `serve run main:my_app --blocking` in a new VSCode terminal. This will block and print out application logs (exceptions, etc.) in the terminal.


```python
!serve run main:my_app --non-blocking
```

### Send a test request
Run the following cell to query the local serve app.


```python
import requests

print(requests.get("http://localhost:8000/hello", params={"name": "Theodore"}).json())
```

## Deploy to production as a service

In order to enable fault tolerance and expose your app to the public internet, you must "Deploy" the application, which will create an Anyscale Service backed by a public load balancer. This service will run in a separate Ray cluster from the workspace, and will be monitored by the Anyscale control plane to recover on node failures. You will also be able to deploy rolling updates to the service without incurring downtime.

Use the following command to deploy your app as `my_service`.


```python
!serve deploy main:my_app --name=my_service
```

**Tip**: if your app has PyPI dependencies added from the workspace, `serve deploy` will automatically compile these dependencies into a Docker image prior to deploying to optimize startup time.

### Service UI Overview

Navigate to your newly created service in the Anyscale UI (`Home > Services > my_service`). It should be in "Starting" state. Click into it and wait for the service to enter "Active" state.

You should see the service state, key metrics, and system event logs on the overview page.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-services/assets/service-overview.png" height=400px>

### Query from the public Internet

Once the service is up, you can query the service from the public Internet using the same logic as when testing it locally, with two changes:
1. Update the `HOST` to the service endpoint.
2. Add the authorization token as a header in the HTTP request.

Both of these values are printed when you run `serve deploy`. You can also find them on the service page. For example, if the output looks like:
```bash
(anyscale +4.0s) You can query the service endpoint using the curl request below:
(anyscale +4.0s) curl -H 'Authorization: Bearer 26hTWi2kZwEz0Tdi1_CKRep4NLXbuuaSTDb3WMXK9DM' https://stable_diffusion_app-4rq8m.cld-ltw6mi8dxaebc3yf.s.anyscaleuserdata-staging.com
```

Then:
- The authorization token is `26hTWi2kZwEz0Tdi1_CKRep4NLXbuuaSTDb3WMXK9DM`.
- The service endpoint is `https://stable_diffusion_app-4rq8m.cld-ltw6mi8dxaebc3yf.s.anyscaleuserdata-staging.com`.


```python
import requests

HOST = "TODO_INSERT_YOUR_SERVICE_HOST"
TOKEN = "TODO_INSERT_YOUR_SERVICE_TOKEN"

def send_request(name: str) -> str:
    response: requests.Response = requests.get(
        f"{HOST}/hello",
        params={"name": name},
        headers={
            "Authorization": f"Bearer {TOKEN}",
        },
    )
    response.raise_for_status()
    return response.content
```


```python
print(send_request("Theodore"))
```

## Monitoring production services

Along with the monitoring tools that come with workspaces, in services you also get a number of built-in metrics out of the box in the `Metrics` tab. This tab includes aggregated metrics across all rollouts for the service (possibly from multiple Ray clusters).

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-services/assets/service-metrics.png" height=500px>

## Configure Service Scaling

By default, the service you created has a single replica. To change this, set the `num_replicas` argument in the [serve.deployment decorator](https://docs.ray.io/en/latest/serve/configure-serve-deployment.html) as follows in `main.py`. For more advanced scaling options, refer to [Serve Autoscaling](https://docs.ray.io/en/latest/serve/autoscaling-guide.html#serve-autoscaling).

```python
@serve.deployment(num_replicas=4)
@serve.ingress(fastapi)
class FastAPIDeployment:
    ...
```

Redeploy locally using `serve run`.


```python
!serve run main:my_app --non-blocking
```

You can check in the Ray Dashboard of the workspace that the number of replicas has been increased:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-services/assets/serve-replicas.png" height=400px/>

We can also deploy the update to our production service. Make sure to include the `--name` option to specify which service to deploy to. This will trigger a staged rollout of the service:


```python
!serve deploy main:my_app --name=my_service
```

Monitor the status of the rollout in the service overview page. Once the new Ray cluster with the updated app config is running, the previous cluster will be shut down:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-services/assets/service-rollout.png" height=300px/>

### Understanding Ray Serve vs Ray cluster config

When scaling your service, it is important to understand the interaction of the Serve scaling config (i.e., contents of `@serve.deployment`), vs the Ray cluster config (i.e., number of Ray worker nodes). In general, you can think of the Ray cluster config as an upper bound on service scaling, since Ray Serve runs inside the Ray cluster.

For example, suppose the Ray cluster was configured to have at most 100 CPUs, then Serve would only be able to launch up to 100 replicas, no matter the deployment config.

For this reason, we generally recommend using the "Auto-select machines" cluster config for services (this is the default).

This concludes the services intro tutorial. To learn more, check out the model serving templates available in the template gallery, as well as the Ray Serve [documentation](https://docs.ray.io/en/latest/serve/index.html).

## Summary

This notebook:
- Developed and ran a simple serve app in the local workspace.
- Deployed the application to production as a service.
- Overviewed production monitoring.
- Scaled the service and covered Ray Serve vs Ray cluster config.
