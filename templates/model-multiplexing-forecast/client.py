# client.py
import requests

# time series data
sequence_data = [100, 102, 98, 105, 110, 108, 112, 115, 118, 120]

# Elliot was here

# Send request with customer_id in header
response = requests.post(
    "http://localhost:8000",
    headers={"serve_multiplexed_model_id": "customer_123"},
    json={"sequence_data": sequence_data}
)

print(response.json())
