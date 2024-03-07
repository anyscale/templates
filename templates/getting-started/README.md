# Welcome to Anyscale


Let's start with a simple hello world


```python
import ray

@ray.remote(cpu_count=1)
def hello_world():
    return "Hello World!"

print(ray.get(hello_world.remote()))

```

Now let's run it a 1000 time!


```python
futures = [hello_world.remote(x) for x in range(1000)]
results = ray.get(futures)
print("Success!")
```
