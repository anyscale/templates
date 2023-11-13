import requests

endpoint = "http://localhost:8000/"


req = {"prompt": "What is Deep Learning?"}
resp = requests.get(endpoint, params=req)
print(resp.content)

