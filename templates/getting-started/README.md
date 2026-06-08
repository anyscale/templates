# Welcome to Anyscale

<div align="left">
  <a target="_blank" href="https://console.anyscale.com/template-preview/getting-started"><img src="https://img.shields.io/badge/🚀 Run_on-Anyscale-9hf"></a>&nbsp;
  <a href="https://github.com/anyscale/templates/tree/main/templates/getting-started" role="button"><img src="https://img.shields.io/static/v1?label=&message=View%20On%20GitHub&color=586069&logo=github&labelColor=2f363d"></a>&nbsp;
</div>

Welcome to your first Workspace - a powerful dev environment designed for making developing distributed AI applications a breeze. This is not just any dev box, it can seamlessly transition to a dev cluster without any configuration hassles.

Let's dive in with a simple "Hello World" example.

## Get the code

```bash
git clone https://github.com/anyscale/templates && cd templates/templates/getting-started
```


```python
import ray
import time

@ray.remote(num_cpus=1)
def hello_world(sleep=0):
    time.sleep(sleep)
    return "Hello World!"

print(ray.get(hello_world.remote()))

```

## Instant Scaling

Now, let's witness the power of Anyscale's instant scaling. With just a few lines of code, you can scale your applications horizontally, leveraging the distributed computing of [Ray](https://www.ray.io/).

Let's scale up!


```python
futures = [hello_world.remote(sleep=5) for x in range(30)]
results = ray.get(futures)
print("Success!")
```
