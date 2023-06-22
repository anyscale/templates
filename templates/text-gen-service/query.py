import sys
import requests

prompt = sys.argv[1]
response = requests.post(
    "http://127.0.0.1:8000/query", params={"prompt": prompt}
)
print(response.content.decode())
