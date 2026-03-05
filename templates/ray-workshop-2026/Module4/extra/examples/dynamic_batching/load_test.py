import asyncio
import time
import aiohttp
import numpy as np


async def send_request(session, url, data):
    async with session.post(url, json=data) as response:
        return await response.json()


async def send_concurrent_requests(num_requests, url="http://localhost:8000/"):
    print(f"Sending {num_requests} concurrent requests to {url}...")
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(session, url, {"image": np.random.rand(28, 28).tolist()})
            for _ in range(num_requests)
        ]
        responses = await asyncio.gather(*tasks)
    
    return responses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-requests", type=int, default=100)
    parser.add_argument("--url", type=str, default="http://localhost:8000/")
    args = parser.parse_args()
    
    start = time.time()
    responses = asyncio.run(send_concurrent_requests(args.num_requests, args.url))
    elapsed = time.time() - start
    
    print(f"Completed {len(responses)} requests in {elapsed:.2f}s")
    print(f"Throughput: {len(responses)/elapsed:.2f} req/s")

