import requests
import os
import time
from fastapi import UploadFile

request_url = "http://localhost:8000/"
start = time.time()

for i in range(100):
    with open("img0.jpg", "rb") as image:
        resp = requests.post(request_url, files={"file": image})

duration = time.time() - start
print(resp.text)
print("Response took " + str(duration) + " seconds")
