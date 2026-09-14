"""
TRL's vLLM server as a Ray actor.

TRL's GRPO "server" mode (`use_vllm: true`, `vllm_mode: server`) generates rollouts on a
separate `trl vllm-serve` process and pushes the trainer's weights to it after every
optimizer step (NCCL, LoRA merged on the fly). This module runs that server as a Ray
actor holding its own GPU(s), so Ray schedules the generator and the trainer as two
resource bundles on the same cluster. `train_grpo_trl.py --ray` starts it automatically
when the config asks for server mode and no `--vllm_server_base_url` was given.

    from vllm_server_actor import start_vllm_server
    url, actor = start_vllm_server(model="Qwen/Qwen3-VL-2B-Instruct", revision="main", tensor_parallel_size=1)
    ...  # pass url as --vllm_server_base_url
    ray.kill(actor)

Requires the `vllm` extra:  uv run --frozen --extra vllm ...
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import ray


@ray.remote
class VLLMServer:
    def __init__(
        self,
        model: str,
        revision: str = "main",
        tensor_parallel_size: int = 1,
        port: int = 8000,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = 4096,
    ):
        self.port = port
        self.ip = ray.util.get_node_ip_address()
        cmd = [
            sys.executable, "-m", "trl.scripts.vllm_serve",
            "--model", model,
            "--revision", revision,
            "--tensor_parallel_size", str(tensor_parallel_size),
            "--host", "0.0.0.0",
            "--port", str(port),
            "--gpu_memory_utilization", str(gpu_memory_utilization),
        ]
        if max_model_len:
            cmd += ["--max_model_len", str(max_model_len)]
        # vLLM inherits this actor's CUDA_VISIBLE_DEVICES, i.e. exactly the GPUs Ray gave it.
        self.proc = subprocess.Popen(cmd, env=os.environ.copy(), stdout=sys.stdout, stderr=sys.stderr)

    def url(self) -> str:
        return f"http://{self.ip}:{self.port}"

    def wait_ready(self, timeout_s: float = 900.0) -> str:
        import requests

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"trl vllm-serve exited with code {self.proc.returncode}")
            try:
                if requests.get(f"{self.url()}/health/", timeout=2).status_code == 200:
                    return self.url()
            except requests.RequestException:
                pass
            time.sleep(3)
        raise TimeoutError(f"vLLM server not healthy after {timeout_s:.0f}s")

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def start_vllm_server(model: str, revision: str = "main", tensor_parallel_size: int = 1, **kwargs):
    """Start the server actor on `tensor_parallel_size` GPUs, block until healthy, return (url, actor)."""
    actor = VLLMServer.options(num_gpus=tensor_parallel_size, name="trl-vllm-server", lifetime="detached").remote(
        model=model, revision=revision, tensor_parallel_size=tensor_parallel_size, **kwargs
    )
    url = ray.get(actor.wait_ready.remote())
    print(f"[vllm] server ready at {url} ({tensor_parallel_size} GPU)", flush=True)
    return url, actor
