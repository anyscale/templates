# Introduction to Anyscale Services

Deploy your machine learning apps into production with Anyscale Services for scalability, fault tolerance, high availability, and zero downtime upgrades.

**⏱️ Time to complete**: 10 min

After implementing and testing your machine learning workloads, it’s time to move them into production. An Anyscale Service packages your application code, dependencies, and compute configurations, deploying them behind a REST endpoint for easy integration and scalability.

This example takes you through a common development to production workflow with services:
1. Development
    a. Develop a service in a workspace.
    b. Run the app in a workspace.
    c. Send a test request.
2. Production
    a. Deploy as an Anyscale Service.
    b. Query the service.
    c. Monitor the service.
    d. Configure scaling.
    e. Update the service.
    f. Terminate the service.

## Development

Start by writing your machine learning service using [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), an open source distributed serving library for building online inference APIs.

### Develop a service in a workspace

 This example begins in an [Anyscale Workspace](https://docs.endpoints.anyscale.com/preview/platform/workspaces/), which is a fully managed development environment connected to a Ray cluster. Look at the following simple Ray Serve app created in `main.py`:

```python
import requests
from fastapi import FastAPI
from ray import serve

fastapi = FastAPI()

@serve.deployment
@serve.ingress(fastapi)
class FastAPIDeployment:
    # FastAPI automatically parses the HTTP request.
    # See https://docs.ray.io/en/latest/serve/http-guide.html
    @fastapi.get("/hello")
    def say_hello(self, name: str) -> str:
        return f"Hello {name}!"

my_app = FastAPIDeployment.bind()
```

Here’s a breakdown of this code that integrates [Ray Serve with Fast API](https://docs.ray.io/en/latest/serve/http-guide.html):

- It defines a FastAPI app named **`fastapi`**.
- **`@serve.deployment`** decorates the class **`FastAPIDeployment`**, indicating it's a [Ray Serve deployment](https://docs.ray.io/en/latest/serve/key-concepts.html#deployment).
- **`@serve.ingress(fastapi)`** marks **`fastapi`** as the [entry point for incoming requests](https://docs.ray.io/en/latest/serve/key-concepts.html#ingress-deployment-http-handling) to this deployment.
- **`say_hello`** method is a GET endpoint **`/hello`** in FastAPI, taking a **`name`** parameter and returning a greeting.
- **`FastAPIDeployment.bind()`** binds the deployment to Ray Serve, making it ready to handle requests.

### Run the app in a workspace

Execute the command below to run the Ray Serve app on the workspace cluster. This command takes in an import path to the deployment formatted as `module:application`.

**Tip**: To stream logs to the console and debug the service, remove the `--non-blocking` flag. You can terminate the process with either the stop button in a notebook of Ctrl-C in the terminal.


```python
!serve run main:my_app --non-blocking
```

### Send a test request
Run the following cell to query the Ray Serve app running in the workspace.


```python
import requests

print(requests.get("http://localhost:8000/hello", params={"name": "Theodore"}).json())
```

## Deploy to production as a service

To enable fault tolerance and expose your app to the public internet, you must "deploy" it, which creates an Anyscale Service backed by a public load balancer. Anyscale deploys the app in a new cluster, separate from the workspace cluster. The Anyscale control plane monitors the service to recover on node failures. You can also deploy rolling updates to the service without incurring downtime.

Use the following command to deploy your app as `my_service`.


```python
!anyscale service deploy main:my_app --name=my_service
```

### Service Overview page in the console

Navigate to your newly created service in the Anyscale console at **Home > Services > my_service**. It should be in the "Starting" state. Click the service name and wait for the service to enter the "Running" state.

You should see the service state, key metrics, and system event logs on the Overview page.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-services/assets/service-overview.png" height=400px>

### Query from the public internet

Once the service is running, query the service from the public internet using similar logic from the test query in the development workspace. Make two changes:
1. Update the `HOST` to the service endpoint.
2. Add the authorization token as a header in the HTTP request.

Find the `HOST` and authorization token values in:
- The output of `anyscale service deploy`
- On the service page by clicking on the **Query**

For example, look for the following in the `anyscale service deploy` output: 

```bash
(anyscale +4.0s) You can query the service endpoint using the curl request below:
(anyscale +4.0s) curl -H 'Authorization: Bearer 26hTWi2kZwEz0Tdi1_CKRep4NLXbuuaSTDb3WMXK9DM' https://stable_diffusion_app-4rq8m.cld-ltw6mi8dxaebc3yf.s.anyscaleuserdata-staging.com
```

In the previous output:
- The service endpoint value is: `https://stable_diffusion_app-4rq8m.cld-ltw6mi8dxaebc3yf.s.anyscaleuserdata-staging.com`.
- The authorization token value is: `26hTWi2kZwEz0Tdi1_CKRep4NLXbuuaSTDb3WMXK9DM`.

Replace the placeholder values in the following cell before running it:


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

## Monitor production services

Along with the monitoring tools that come with workspaces, services provide additional built-in metrics that you can find in the `Metrics` tab. This tab includes aggregated metrics across all rollouts for a service, possibly from multiple clusters.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-services/assets/service-metrics.png" height=500px>

## Configure service scaling

By default, a service has a single replica. To change this configuration, set the `num_replicas` argument in the [serve.deployment decorator](https://docs.ray.io/en/latest/serve/configure-serve-deployment.html) as follows in `main.py`:

```python
@serve.deployment(num_replicas=4)
@serve.ingress(fastapi)
class FastAPIDeployment:
    ...
```

 For more advanced scaling options, see [Ray Serve Autoscaling](https://docs.ray.io/en/latest/serve/autoscaling-guide.html#serve-autoscaling).
 
Rerun the service in the development workspace using `serve run`.


```python
!serve run main:my_app --non-blocking
```

Verify the increase in the number of replicas in the Ray Dashboard of the development workspace:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-services/assets/serve-replicas.png" height=400px/>

On the production service, deploy the update, making sure to include the `--name` option to specify which service to deploy to. This command triggers a staged rollout of the service:


```python
!anyscale service deploy main:my_app --name=my_service
```

Monitor the status of the rollout in the service Overview page. Once the new cluster with the updated app config is running, Ray Serve shuts down the previous cluster:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-services/assets/service-rollout.png" height=300px/>

### Ray Serve autoscaling config vs compute config

 During service scaling, the Ray Serve autoscaling config interacts with the compute config of the cluster. The `@serve.deployment` decorater contains the autoscaling config, such as `num_replicas`, and the compute config sets the number of worker nodes. Generally, the compute config is an upper bound on service scaling because Ray Serve runs inside the cluster. For example, if you configure the cluster to have at most 100 CPUs, then Ray Serve can only launch up to 100 replicas, regardless of the autoscaling config.

For this reason, enable the "Auto-select machines" compute config for services. This setting is on by default.

#### Edit a compute config

When Anyscale first creates a service, it copies the compute config from the workspace. After that, Anyscale decouples the service cluster's compute config from the workspace and you can edit it independently.

To learn more, try other model serving templates available in the template gallery, and the Ray Serve [documentation](https://docs.ray.io/en/latest/serve/index.html).

## Summary

In this notebook you:
- Developed and ran a simple Ray Serve app in a development workspace.
- Deployed the app to production as a service.
- Monitored the service.
- Scaled the service that uses both the Ray Serve autoscaling config and compute config together.
