"""Locust load test for the kMoL ensemble Serve endpoint (REC 4).

Drive this OPEN-LOOP from a separate node/process at high concurrency and watch GPU
utilization at the same time — if util is low at peak, you're bottlenecked on the
client or on featurization, not on Ray.

Run:
    locust -f scripts/locustfile.py --host http://<serve-host>:8000 \
        --users 200 --spawn-rate 50 --run-time 2m --headless \
        --csv results/kmol_locust
"""

import os
import random

from locust import HttpUser, between, task

# Realistic drug-like SMILES; each request sends one molecule (the common online case).
SMILES = [
    "CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1CCC[C@H]1c1cccnc1",
    "Clc1ccccc1C(=O)Nc1ccccc1", "COc1ccc2nc(sc2c1)N", "CC(=O)Nc1ccc(O)cc1",
    "CCN(CC)CCNC(=O)c1ccc(N)cc1", "O=C(O)c1ccccc1O", "CC(=O)Nc1ccc(OCC(O)CNC(C)C)cc1",
]

# Optional bearer token for an Anyscale Service endpoint.
_TOKEN = os.environ.get("KMOL_SERVICE_TOKEN")


class KmolUser(HttpUser):
    # Near-zero think time → approximates open-loop when users are high.
    wait_time = between(0.0, 0.01)

    def on_start(self):
        if _TOKEN:
            self.client.headers.update({"Authorization": f"Bearer {_TOKEN}"})

    @task
    def predict(self):
        with self.client.post("/", json={"smiles": random.choice(SMILES)},
                              catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
            elif "logits" not in resp.text:
                resp.failure("no logits in response")
