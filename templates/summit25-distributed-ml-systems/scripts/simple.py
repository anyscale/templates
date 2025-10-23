"""This module submits a simple Ray task, gets the result, and prints it."""
import ray
import time

# Define a simple Ray task
@ray.remote
def simple_task():
    time.sleep(60 * 2)
    return "Hello, Ray!"

if __name__ == "__main__":
    # The raylet process logs detailed information about events like task execution
    # and object transfers between nodes. Set the logging level at runtime
    # to get more information.
    ray.init(runtime_env={"env_vars": {"RAY_BACKEND_LOG_LEVEL": "debug"}})

    # Invoke the task
    result = simple_task.remote()

    # Get the result
    print(ray.get(result))

# Inspect the logs under python-core-driver-{job_id}
