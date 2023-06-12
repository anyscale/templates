import ray

@ray.remote
def square(x):
    import time
    time.sleep(30)
    return x * x

# run 30 tasks
remotes = [square.remote(x) for x in range(30)]
print(ray.get(remotes))
