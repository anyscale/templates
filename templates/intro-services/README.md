# Introduction to Anyscale Services

Deploy your machine learning apps into production with [Anyscale Services](https://docs.anyscale.com/platform/services/) for scalability, fault tolerance, high availability, and zero downtime upgrades.

---

**⏱️ Time to complete**: 10 min

**Prerequisite**: [Intro to Workspaces](https://console.anyscale.com/template-preview/workspace-intro)

After implementing and testing your machine learning workloads, it’s time to move them into production. An Anyscale Service packages your application code, dependencies, and compute configurations, deploying them behind a REST endpoint for easy integration and scalability.

This interactive example takes you through a common development to production workflow with services:

- Development
  - Develop a service in a workspace.
  - Run the app in a workspace.
  - Send a test request.
- Production
  - Deploy as an Anyscale Service.
  - Check the status of the service.
  - Query the service.
  - Monitor the service.
  - Configure scaling.
  - Update the service.
  - Terminate the service.


## Development

Start by writing your machine learning service using [Ray Serve](https://docs.ray.io/en/latest/serve/index.html), an open source distributed serving library for building online inference APIs.

### Develop a service in a workspace

This example begins in an [Anyscale Workspace](https://docs.anyscale.com/platform/workspaces/), which is a fully managed development environment connected to a Ray cluster. Look at the following simple Ray Serve app created in `main.py`:



```python
import logging
import requests
from fastapi import FastAPI
from ray import serve

fastapi = FastAPI()
logger = logging.getLogger("ray.serve")

@serve.deployment
@serve.ingress(fastapi)
class FastAPIDeployment:
    # FastAPI automatically parses the HTTP request.
    @fastapi.get("/hello")
    def say_hello(self, name: str) -> str:
        logger.info("Handling request!")
        return f"Hello {name}!"

my_app = FastAPIDeployment.bind()
```

The following is a breakdown of this code that integrates [Ray Serve with Fast API](https://docs.ray.io/en/latest/serve/http-guide.html):

- It defines a FastAPI app named `fastapi` and a logger named `ray.serve`.
- `@serve.deployment` decorates the class `FastAPIDeployment`, indicating it's a [Ray Serve deployment](https://docs.ray.io/en/latest/serve/key-concepts.html#deployment).
- `@serve.ingress(fastapi)` marks `fastapi` as the [entry point for incoming requests](https://docs.ray.io/en/latest/serve/key-concepts.html#ingress-deployment-http-handling) to this deployment.
- `say_hello` handles GET requests to the `/hello` endpoint in FastAPI, taking a `name` parameter, printing a log message, and returning a greeting.
- `FastAPIDeployment.bind()` binds the deployment to Ray Serve, making it ready to handle requests.


### Run the app in a workspace

Execute the commands below to start Ray Serve and run the Serve app on the workspace cluster. This command takes in an import path to the deployment formatted as `module:application`.



```python
!serve start --http-port 8005
!serve run main:my_app --non-blocking
```

**Note**: This command blocks and streams logs to the console for easier debugging in development. To terminate this service and continue with this example, either click the stop button in the notebook or `Ctrl-C` in the terminal.


### Send a test request

Your app is accessible through `localhost:8005`. Run the following to send a GET request to the `/hello` endpoint with query parameter `name` set to “Theodore.”



```python
import requests

print(requests.get("http://localhost:8005/hello", params={"name": "Theodore"}).json())
```

## Production

To move into production, use Anyscale Services to deploy your Ray Serve app to a new separate cluster without any code modifications. Anyscale handles the scaling, fault tolerance, and load balancing of your services to ensure uninterrupted service across node failures, heavy traffic, and rolling updates.

### Deploy as an Anyscale Service

Use the following to deploy `my_service` in a single command:



```python
# Define the service name - change this to customize your service name
SERVICE_NAME = "my_service"

print(f"Service name: {SERVICE_NAME}")

!anyscale service deploy main:my_app --name={SERVICE_NAME}
```

**Note**: This Anyscale Service pulls the associated dependencies, compute config, and service config from the workspace. To define these explicitly, you can deploy from a `config.yaml` file using the `-f` flag. See [ServiceConfig reference](https://docs.anyscale.com/reference/service-api#serviceconfig) for details.


### Check the status of the service

To get the status of `my_service`, run the following:



```python
!anyscale service status --name={SERVICE_NAME}
```

### Query the service

When you deploy, you expose the service to a publicly accessible IP address which you can send requests to.

In the preceding cell’s output, copy your `API_KEY` and `BASE_URL`. As an example, the values look like the following:

- `API_KEY`: `NMv1Dq3f2pDxWjj-efKKqMUk9UO-xfU3Lo5OhpjAHiI`
- `BASE_URL`: `https://my-service-jrvwy.cld-w3gg9qpy7ft3ayan.s.anyscaleuserdata.com/`

Fill in the following placeholder values for the `BASE_URL` and `API_KEY` in the following Python requests object:



```python
import subprocess
import re

# Extract service information from the status command
def extract_service_info(service_name):
    """Extract the API token and base URL from anyscale service status command."""
    try:
        # Run the service status command
        result = subprocess.run(
            ["anyscale", "service", "status", f"--name={service_name}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract query_auth_token
        token_match = re.search(r'query_auth_token:\s*(\S+)', result.stdout)
        api_key = token_match.group(1) if token_match else None
        
        # Extract query_url (handle potential line breaks and color codes in output)
        # The URL might be split across multiple lines with ANSI color codes
        url_match = re.search(r'query_url:\s*\n?\s*(?:\[[0-9;]+m)?(https://[^\s\[\n]+)(?:\[[0-9;]+m)?(?:\n\s*(?:\[[0-9;]+m)?([^\s\[\n]+))?', result.stdout)
        if url_match:
            base_url = url_match.group(1)
            # If there's a second part (like "om" after line break), append it
            if url_match.group(2):
                base_url += url_match.group(2)
        else:
            base_url = None
        
        return api_key, base_url
    except subprocess.CalledProcessError as e:
        print(f"Error running service status command: {e}")
        return None, None

# Extract the service info for the deployed service
API_KEY, BASE_URL = extract_service_info(SERVICE_NAME)

print(f"Extracted API_KEY: {API_KEY}")
print(f"Extracted BASE_URL: {BASE_URL}")

# Verify we got valid values
if not API_KEY or not BASE_URL:
    print("Warning: Could not extract service information. Please check the service status manually.")
else:
    print("✅ Successfully extracted service credentials for testing!")

```


```python
import requests

def send_request(name: str) -> str:
    response: requests.Response = requests.get(
        f"{BASE_URL}/hello",
        params={"name": name},
        headers={
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    response.raise_for_status()
    return response.content
```


```python
# Note: make sure you have network connectivity to the public IP address
print(send_request("Theodore"))
```

### Monitor the service

To view the service, navigate to 🏠 **> Services > `my_service`**. On this page, inspect key metrics, events, and logs. With Anyscale’s monitoring dashboards, you can track performance and adjust configurations as needed without deep diving into infrastructure management. See [Monitor a service](https://docs.anyscale.com/platform/services/monitoring).

By clicking on the **Running** service, you can view the status of deployments and how many replicas each contains. For example, your `FastAPIDeployment` has `1` replica.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-services/assets/service-overview.png" height=400px />

In the Logs, you can search for the message “Handling request!” to view each request for easier debugging.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-services/assets/service-logs.png" height=400px />


### Configure scaling

Each Ray Serve deployment has one replica by default. There is one worker process running the model and serving requests.

As a quick example to scale out from one to autoscaling replicas, modify the original service script `main.py`. Add the `num_replicas` argument to the `@serve.deployment` decorator as follows:

```diff
import requests
from fastapi import FastAPI
from ray import serve

fastapi = FastAPI()

- @serve.deployment
+ @serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class FastAPIDeployment:
    # FastAPI automatically parses the HTTP request.
    @fastapi.get("/hello")
    def say_hello(self, name: str) -> str:
        return f"Hello {name}!"

my_app = FastAPIDeployment.bind()
```


<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-services/assets/service-replicas.png" height=400px />

**Note**: This approach is a way to quickly modify scale for this example. As a best practice in production, define [autoscaling behavior](https://docs.anyscale.com/platform/services/scale-a-service#autoscaling) in the [ServiceConfig](https://docs.anyscale.com/reference/service-api#serviceconfig) contained in a `config.yaml` file. The number of worker nodes that Anyscale launches dynamically scales up and down in response to traffic and is scoped by the overall cluster compute config you define.


### Update the service

To deploy the update, execute the following command to trigger a staged rollout of the new service with zero downtime:



```python
!anyscale service deploy main:my_app --name=my_service
```

In the service overview page, you can monitor the status of the update and see Ray Serve shut down the previous cluster.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/intro-services/assets/service-rollout.png" height=400px />

**Note**: Using this command triggers an automatic rollout which gradually shifts traffic from the previous cluster, or primary version, to the incoming cluster, or canary version. To learn more about configuring rollout behavior, see [Update a service](https://docs.anyscale.com/platform/services/update-a-service).


### Terminate the service

To tear down the service cluster, run the following command:



```python
!anyscale service terminate --name=my_service
```

## Summary

In this example, you learned the basics of Anyscale Services:

- Develop a service in a workspace.
  - Run the app in a workspace.
  - Send a test request.
- Deploy as an Anyscale Service.
  - Query the service.
  - Check the status of the service.
  - Monitor the service.
  - Configure scaling.
  - Update the service.
  - Terminate the service.

