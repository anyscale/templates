## Serving a Stable Diffusion Model with Ray Serve

**⏱️ Time to complete**: 5 min (15 on GCP)

This template shows you how to:
1. Develop and run a Ray Serve application running a stable diffusion model.
2. Send test requests to the application running locally.
3. Deploy the application to production as a service.
4. Send requests to the application running in production as a service.

### Step 1: Install python dependencies

The application requires a few extra Python dependencies. Install them using `pip` and they'll be saved in the workspace and picked up when deploying to production.


```python
!pip install --user -q diffusers==0.25.0 transformers==4.36.2 accelerate==0.25.0 && echo 'Install complete!'
```

### Step 2: Run the model locally
- Run the command below in a VSCode terminal (Ctrl-`).
- The model will be available at http://localhost:8000.
- The command will block and print logs for the application.

```bash
# Run the following in a VSCode terminal because it's a blocking command.
$ serve run main:stable_diffusion_app
```

### Step 3: Send a test request to the model running locally

The `generate_image` function sends an HTTP request to the model and saves the response as a local image.

As the request to generate the image runs, logs will be printed to the terminal that is running `serve run`.


```python
import requests

HOST = "http://localhost:8000"

def generate_image(prompt: str, image_size: int) -> bytes:
    response: requests.Response = requests.get(
        f"{HOST}/imagine",
        params={"prompt": prompt, "img_size": image_size},
    )
    response.raise_for_status()
    return response.content
```


```python
image = generate_image("twin peaks sf in basquiat painting style", 640)

filename = "image.png"
with open(filename, "wb") as f:
    f.write(image)

from IPython.display import Image
Image(filename=filename)
```

### Step 4: Deploy the model to production as a service

Deploy the model to production using the `anyscale service rollout` command.

This creates a long-running [service](https://docs.anyscale.com/platform/services/) with a stable endpoint to query the application.

Note that we installed some pip packages in the workspace that had to be added in to the runtime environment. For faster setup of your deployments, you can build a new [cluster environment](https://docs.anyscale.com/configure/dependency-management/cluster-environments) with these packages.


```python
!anyscale service rollout -f service.yaml --name {ENTER_NAME_FOR_SERVICE}
```

### Step 5: Send a test request to the model running in the service

Query the service using the same logic as when testing it locally, with two changes:
1. Update the `HOST` to the service endpoint.
2. Add the authorization token as a header in the HTTP request.

Both of these values are printed when you run `anyscale service rollout`. You can also find them on the service page. For example, if the output looks like:
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

def generate_image(prompt: str, image_size: int) -> bytes:
    response: requests.Response = requests.get(
        f"{HOST}/imagine",
        params={"prompt": prompt, "img_size": image_size},
        headers={
            "Authorization": f"Bearer {TOKEN}",
        },
    )
    response.raise_for_status()
    return response.content
```


```python
image = generate_image("twin peaks sf in basquiat painting style", 640)

filename = "image.png"
with open(filename, "wb") as f:
    f.write(image)

from IPython.display import Image
Image(filename=filename)
```

## Summary

This notebook:
- Developed and ran a model serving application locally.
- Sent a test request to the application locally.
- Deployed the application to production as a service.
- Sent another test request to the application running as a service.


