"""
Isaac Lab G1 sim worker - runs as a STANDALONE SUBPROCESS, not a Ray actor.

WHY subprocess and not @ray.remote actor?
Isaac Sim uses asyncio internally via omni.kit.async_engine. Inside Ray actor
threads, MainEventLoopWrapper.g_main_event_loop is None, so scene loading
crashes with 'NoneType' object has no attribute 'create_task'.

Spawning a subprocess gives Isaac Sim a clean Python interpreter + event loop.
The cost is: communicating with the Ray Serve policy via HTTP instead of Ray
DeploymentHandle. Ray Serve exposes an HTTP endpoint at http://HEAD:8000 by
default; we POST obs dicts and get actions back.

Usage (invoked by run_demo.py):
    python sim_worker.py \
        --worker-id 0 \
        --policy-url http://10.0.18.189:8000 \
        --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0 \
        --episodes 1 \
        --max-steps 50 \
        --action-horizon 8 \
        --output-dir /home/ray/groot_demo/demo_output \
        --seed 42
"""
import argparse
import json
import os
import pickle
import sys
import time
from typing import List

import numpy as np


def _serialize_obs(obs: dict) -> bytes:
    """Serialize an obs dict with numpy arrays for HTTP transport."""
    return pickle.dumps(obs)


def _deserialize_response(data: bytes) -> dict:
    return pickle.loads(data)


def query_policy(policy_url: str, obs: dict, timeout: float = 60.0) -> dict:
    """POST obs dict to Ray Serve policy; get action chunk back."""
    import requests
    body = _serialize_obs(obs)
    r = requests.post(
        policy_url.rstrip("/") + "/predict",
        data=body,
        headers={"Content-Type": "application/octet-stream"},
        timeout=timeout,
    )
    r.raise_for_status()
    return _deserialize_response(r.content)


def save_gif(frames: List[np.ndarray], path: str):
    import imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    imageio.mimsave(path, frames, fps=15, loop=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--policy-url", default="http://127.0.0.1:8000",
                        help="Ray Serve HTTP endpoint for the policy")
    parser.add_argument("--task", default="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0")
    parser.add_argument("--instruction", default="pick up the apple and place it on the plate")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--output-dir", default="demo_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument("--save-frames-every", type=int, default=2)
    parser.add_argument("--results-file", default=None,
                        help="If set, write run results as JSON here")
    args = parser.parse_args()

    print(f"[Worker-{args.worker_id}] Starting sim subprocess")
    print(f"[Worker-{args.worker_id}] Policy URL: {args.policy_url}")
    print(f"[Worker-{args.worker_id}] Task: {args.task}", flush=True)

    # ------------------------------------------------------------
    # Import env (this boots Isaac Sim - slow, ~90s first time).
    # ------------------------------------------------------------
    from g1_env import G1LocomanipulationEnv

    env = G1LocomanipulationEnv(
        task_name=args.task,
        language_instruction=args.instruction,
        headless=bool(args.headless),
        seed=args.seed,
    )
    print(f"[Worker-{args.worker_id}] Env ready", flush=True)

    # GR00T returns 40-step chunks; don't exceed that
    GR00T_HORIZON = 40
    action_horizon = min(args.action_horizon, GR00T_HORIZON)

    all_results = []
    for ep_idx in range(args.episodes):
        obs = env.reset()
        frames = []
        latencies = []
        policy_calls = 0

        action_chunk = None
        chunk_idx = action_horizon  # force first query

        t_start = time.time()
        total_reward = 0.0
        step = 0

        try:
            for step in range(args.max_steps):
                # Query policy when chunk exhausted
                if chunk_idx >= action_horizon:
                    t0 = time.time()
                    response = query_policy(args.policy_url, obs, timeout=120.0)
                    action_chunk = response["action"]
                    latencies.append(response.get("latency_ms", (time.time() - t0) * 1000))
                    policy_calls += 1
                    chunk_idx = 0

                obs, reward, done, info = env.step(action_chunk, step_idx=chunk_idx)
                chunk_idx += 1
                total_reward += float(reward)

                if step % args.save_frames_every == 0:
                    frames.append(env.render_frame())

                if done:
                    print(f"[Worker-{args.worker_id}] Episode {ep_idx} done at step {step} "
                          f"(reward={total_reward:.3f})", flush=True)
                    break
        except Exception as e:
            print(f"[Worker-{args.worker_id}] Error during episode: {type(e).__name__}: {e}",
                  flush=True)
            import traceback
            traceback.print_exc()
            # Don't re-raise - save whatever frames we got

        episode_time = time.time() - t_start

        gif_path = os.path.join(args.output_dir,
                                 f"worker{args.worker_id}_ep{ep_idx}.gif")
        if frames:
            save_gif(frames, gif_path)
            print(f"[Worker-{args.worker_id}] Saved GIF: {gif_path} ({len(frames)} frames)",
                  flush=True)
        else:
            gif_path = None

        result = {
            "worker_id": args.worker_id,
            "episode": ep_idx,
            "steps": step + 1,
            "policy_calls": policy_calls,
            "episode_time_s": episode_time,
            "avg_policy_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "total_reward": total_reward,
            "task": args.task,
            "instruction": args.instruction,
            "gif_path": gif_path,
        }
        all_results.append(result)

        print(f"[Worker-{args.worker_id}] Episode {ep_idx}: "
              f"{result['steps']} steps, {result['policy_calls']} policy calls, "
              f"{result['avg_policy_latency_ms']:.1f}ms avg latency", flush=True)

    env.close()

    # Write results file for the orchestrator to pick up.
    if args.results_file:
        os.makedirs(os.path.dirname(args.results_file) or ".", exist_ok=True)
        with open(args.results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"[Worker-{args.worker_id}] Wrote results to {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
