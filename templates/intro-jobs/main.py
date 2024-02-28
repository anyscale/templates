import ray
import emoji
import time

@ray.remote
def process(x):
   print(emoji.emojize("Processing :thumbs_up:"), x)
   time.sleep(1)
   return x * 2

result = ray.get([process.remote(x) for x in range(10)])
print("The job result is", result)
