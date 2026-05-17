"""
Demo orchestrator: deploy GR00T (or Placeholder) via Ray Serve HTTP, then spawn
Isaac Lab sim workers as SUBPROCESSES on Ray tasks (NOT as Ray actors — Isaac
Sim's asyncio init breaks inside actor threads; subprocesses get clean
interpreters + event loops).

Architecture:

    head node
    │
    ├── Ray Serve cluster (starts on some GPU worker)
    │     ├── GR00T policy replica(s)   [POST /predict] [GET /stats]
    │     └── HTTP endpoint at http://HEAD:8000
    │
    └── N sim worker tasks (one per GPU worker)
          each task.remote() runs on a worker and shell-execs:
            python sim_worker.py --policy-url http://HEAD:8000 ...

Usage:
    # Placeholder test (no GR00T, proves architecture):
    python run_demo.py --placeholder --num-workers 1 --episodes 1 --max-steps 50

    # Full GR00T demo:
    python run_demo.py --num-workers 2 --episodes 1 --max-steps 200
"""
import argparse
import json
import os
import socket
import time
import subprocess

import ray
from ray import serve


def _get_head_ip() -> str:
    """Find an IP that sim worker subprocesses can use to reach Ray Serve."""
    # Try the ray dashboard's bind IP first
    try:
        ctx = ray.get_runtime_context()
        gcs_addr = ctx.gcs_address  # e.g. '10.0.18.189:6379'
        ip = gcs_addr.split(":")[0]
        if ip and ip != "0.0.0.0" and ip != "127.0.0.1":
            return ip
    except Exception:
        pass
    # Fallback: best-effort
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--placeholder", action="store_true",
                        help="Use random-action placeholder instead of loading GR00T")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel Isaac Lab sim workers")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--action-horizon", type=int, default=8,
                        help="Execute N actions from each chunk before re-querying")
    parser.add_argument("--model-path", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument("--embodiment-tag", default="REAL_G1")
    parser.add_argument("--task", default="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0")
    parser.add_argument("--instruction",
                        default="pick up the apple and place it on the plate")
    parser.add_argument("--output-dir",
                        default="/home/ray/groot_demo/demo_output")
    parser.add_argument("--num-policy-replicas", type=int, default=1)
    parser.add_argument("--serve-port", type=int, default=8000)
    parser.add_argument("--worker-dir", default="/home/ray/groot_demo",
                        help="Where on the worker the sim_worker.py + g1_env.py live")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # HF_TOKEN propagation (Cosmos-Reason2-2B is gated)
    # ----------------------------------------------------------------
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        tok_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(tok_path):
            with open(tok_path) as f:
                hf_token = f.read().strip()
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        print(f"[Orchestrator] HF_TOKEN loaded (...{hf_token[-8:]})")
    else:
        print("[Orchestrator] WARNING: no HF_TOKEN — gated downloads will fail")

    # ----------------------------------------------------------------
    # Ray init + Serve start
    # ----------------------------------------------------------------
    runtime_env = {"env_vars": {"HF_TOKEN": hf_token}} if hf_token else {}
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)

    serve.start(
        detached=False,
        http_options={"host": "0.0.0.0", "port": args.serve_port},
    )

    head_ip = _get_head_ip()
    policy_url = f"http://{head_ip}:{args.serve_port}"
    print(f"[Orchestrator] Ray Serve will expose policy at {policy_url}")

    # ----------------------------------------------------------------
    # Deploy policy
    # ----------------------------------------------------------------
    import sys
    sys.path.insert(0, args.worker_dir)  # so policy_server imports
    from policy_server import GR00TPolicyServer, PlaceholderPolicyServer

    if args.placeholder:
        print("[Orchestrator] Deploying PlaceholderPolicyServer (random actions)")
        deployment = PlaceholderPolicyServer.options(
            num_replicas=args.num_policy_replicas,
        ).bind()
    else:
        print(f"[Orchestrator] Deploying GR00TPolicyServer  model={args.model_path}")
        deployment = GR00TPolicyServer.options(
            num_replicas=args.num_policy_replicas,
        ).bind(
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
        )

    serve.run(deployment, name="gr00t-policy")
    print(f"[Orchestrator] Policy deployed. Sanity-checking HTTP endpoint...")

    # Sanity HTTP check (also warms model for GR00T case)
    import requests, pickle
    import numpy as np
    identity_pose = np.array([0.3, 0.0, 0.0, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    dummy_obs = {
        "video": {"ego_view": np.zeros((1, 2, 256, 256, 3), dtype=np.uint8)},
        "state": {
            "left_wrist_eef_9d":  identity_pose[None, None, :].copy(),
            "right_wrist_eef_9d": identity_pose[None, None, :].copy(),
            "left_hand":   np.zeros((1, 1, 7), dtype=np.float32),
            "right_hand":  np.zeros((1, 1, 7), dtype=np.float32),
            "left_arm":    np.zeros((1, 1, 7), dtype=np.float32),
            "right_arm":   np.zeros((1, 1, 7), dtype=np.float32),
            "waist":       np.zeros((1, 1, 3), dtype=np.float32),
        },
        "language": {"annotation.human.task_description": [[args.instruction]]},
    }
    t0 = time.time()
    try:
        r = requests.post(f"{policy_url}/predict",
                          data=pickle.dumps(dummy_obs),
                          timeout=180 if not args.placeholder else 30)
        r.raise_for_status()
        resp = pickle.loads(r.content)
        print(f"[Orchestrator] Sanity ping OK in {time.time()-t0:.1f}s; "
              f"action keys: {list(resp['action'].keys())}")
    except Exception as e:
        print(f"[Orchestrator] Sanity ping FAILED: {e}")
        return 1

    # ----------------------------------------------------------------
    # Launch sim workers (each on a Ray task that shell-execs sim_worker.py)
    # ----------------------------------------------------------------
    @ray.remote(num_gpus=1, runtime_env=runtime_env)
    def run_sim_subprocess(worker_id, policy_url, worker_dir, cli_args):
        """Launch sim_worker.py as a subprocess on the assigned GPU node."""
        import os, subprocess
        results_file = f"/tmp/worker_{worker_id}_results.json"
        cmd = (
            f"cd {worker_dir} && "
            f"timeout 900 python -u sim_worker.py "
            f"--worker-id {worker_id} "
            f"--policy-url {policy_url} "
            f"--results-file {results_file} "
            f"{cli_args}"
        )
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, executable="/bin/bash",
        )
        # Read results if the subprocess wrote them
        try:
            with open(results_file) as f:
                results = json.load(f)
        except Exception:
            results = []
        hostname = os.uname().nodename
        return {
            "host": hostname,
            "worker_id": worker_id,
            "exit_code": r.returncode,
            "stdout_tail": "\n".join(r.stdout.splitlines()[-30:]),
            "stderr_tail": "\n".join(r.stderr.splitlines()[-15:]),
            "results": results,
        }

    cli_args_common = (
        f"--task '{args.task}' "
        f"--instruction '{args.instruction}' "
        f"--episodes {args.episodes} "
        f"--max-steps {args.max_steps} "
        f"--action-horizon {args.action_horizon} "
        f"--output-dir '{args.output_dir}' "
    )

    print(f"[Orchestrator] Launching {args.num_workers} sim subprocess(es)...")
    t_start = time.time()
    futures = [
        run_sim_subprocess.remote(
            wi,
            policy_url,
            args.worker_dir,
            cli_args_common + f"--seed {42 + wi * 1000}",
        )
        for wi in range(args.num_workers)
    ]

    sim_results = ray.get(futures)
    total_time = time.time() - t_start

    # ----------------------------------------------------------------
    # Print results
    # ----------------------------------------------------------------
    print("=" * 70)
    print(f"Demo complete in {total_time:.1f}s")
    print("=" * 70)
    for r in sim_results:
        print(f"\n[host={r['host']} worker={r['worker_id']} exit={r['exit_code']}]")
        print("--- stdout tail ---")
        print(r["stdout_tail"])
        if r["exit_code"] != 0:
            print("--- stderr tail ---")
            print(r["stderr_tail"])
        for ep in r["results"]:
            print(f"  ep{ep['episode']}: {ep['steps']} steps, "
                  f"{ep['policy_calls']} calls, "
                  f"{ep['avg_policy_latency_ms']:.1f}ms avg, "
                  f"gif={ep['gif_path']}")

    # Server stats
    try:
        r = requests.get(f"{policy_url}/stats", timeout=10)
        print("\nPolicy server stats:")
        print(json.dumps(r.json(), indent=2))
    except Exception as e:
        print(f"(couldn't fetch stats: {e})")


if __name__ == "__main__":
    main()
