from locust import HttpUser, constant, task, LoadTestShape


class ImageClassificationUser(HttpUser):
    wait_time = constant(0)

    @task
    def classify_image(self):
        payload = {
            "uri": "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/refs/heads/master/n01491361_tiger_shark.JPEG"
        }
        self.client.post("/", json=payload)


class AutoscalingTrafficPattern(LoadTestShape):
    """
    Traffic pattern to demonstrate autoscaling behavior.

    Stages simulate realistic traffic with gradual ramp up, sustained load,
    spike, and ramp down to observe scale-up and scale-down behavior.
    """
    stages = [
        {"cumulative_duration": 60, "users": 1, "spawn_rate": 1},
        {"cumulative_duration": 180, "users": 20, "spawn_rate": 1 / 30}, # 1 user per 30 seconds
        {"cumulative_duration": 360, "users": 40, "spawn_rate": 1 / 30},
    ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["cumulative_duration"]:
                return (stage["users"], stage["spawn_rate"])

        return None
