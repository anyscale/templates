"""
Demo: _by_reference=True vs False object store usage.

Usage:
    python by_reference_demo.py              # gRPC mode (no object store)
    python by_reference_demo.py --by-ref     # Ray Core mode (uses object store)
"""
import argparse
import subprocess
import threading
import time
import re
import ray
import requests
from ray import serve
from ray.serve.handle import DeploymentHandle
from concurrent.futures import ThreadPoolExecutor


@serve.deployment
class Model:
    def generate(self, data: dict) -> dict:
        time.sleep(1.0)  # Slow down to keep objects in store longer
        return {"output": f"Processed: {data.get('text', '')[:20].upper()}..."}


def create_gateway(by_ref: bool):
    @serve.deployment
    class Gateway:
        def __init__(self, model: DeploymentHandle):
            self._model = model.options(_by_reference=by_ref)
        
        async def __call__(self, req):
            return await self._model.generate.remote(await req.json())
    return Gateway


def get_memory_stats():
    out = subprocess.run(["ray", "memory", "--stats-only"], capture_output=True, text=True).stdout
    match = re.search(r'Plasma memory usage (\d+) MiB, (\d+) objects', out)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def monitor(stop):
    while not stop.is_set():
        mib, obj = get_memory_stats()
        print(f"  [monitor] {mib} MiB, {obj} objects")
        time.sleep(0.3)


def send_request(payload):
    resp = requests.post("http://localhost:8000/", json=payload)
    return resp.ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--by-ref", action="store_true", help="Use _by_reference=True")
    parser.add_argument("-n", type=int, default=20, help="Number of requests")
    parser.add_argument("--kb", type=int, default=200, help="Payload size KB")
    args = parser.parse_args()

    print(f"\nMode: {'Ray Core' if args.by_ref else 'gRPC'} | {args.n} requests | {args.kb}KB\n")
    
    if not ray.is_initialized():
        ray.init()
    
    Gateway = create_gateway(args.by_ref)
    serve.run(Gateway.bind(Model.bind()))
    
    stop = threading.Event()
    threading.Thread(target=monitor, args=(stop,), daemon=True).start()
    
    # Send requests CONCURRENTLY so multiple are in-flight at once
    payload = {"text": "x" * (args.kb * 1024)}
    print(f"Sending {args.n} concurrent requests...")
    with ThreadPoolExecutor(max_workers=args.n) as executor:
        futures = [executor.submit(send_request, payload) for _ in range(args.n)]
        for i, f in enumerate(futures):
            ok = f.result()
            status = "succeeded" if ok else "failed"
            print(f"Request {i+1}/{args.n} {status}")
    
    stop.set()
    print(f"\nExpected: {'non-zero' if args.by_ref else 'zero'} object store usage\n")


if __name__ == "__main__":
    main()
