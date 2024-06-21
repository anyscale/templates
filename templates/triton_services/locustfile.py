from locust import HttpUser, task

PROMPT = "dogs%20in%20new%20york,%20realistic,%204k,%20photograph"


class RayServeUser(HttpUser):
    @task
    def ray_serve_task(self):
        path = f"/generate?prompt={PROMPT}"
        self.client.get(path)
