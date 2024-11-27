# ws-template-imports-start
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import random

from typing import Optional
# ws-template-imports-end

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

# ws-template-code-start
class SimpleCorridor(gym.Env):
    """Example of a custom env in which the agent has to walk down a corridor.

    ------------
    |S........G|
    ------------
    , where S is the starting position, G is the goal position, and fields with '.'
    mark free spaces, over which the agent may step. The length of the above example
    corridor is 10.
    Allowed actions are left (0) and right (1).
    The reward function is -0.01 per step taken and a uniform random value between
    0.5 and 1.5 when reaching the goal state.

    You can configure the length of the corridor via the env's config. Thus, in your
    AlgorithmConfig, you can do:
    `config.environment(env_config={"corridor_length": ..})`.
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.end_pos = config.get("corridor_length", 7)
        self.cur_pos = 0
        self.action_space = Discrete(2)
        self.observation_space = Box(0.0, self.end_pos, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        self.cur_pos = 0
        # Return obs and (empty) info dict.
        return np.array([self.cur_pos], np.float32), {"env_state": "reset"}

    def step(self, action):
        assert action in [0, 1], action
        # Move left.
        if action == 0 and self.cur_pos > 0:
            self.cur_pos -= 1
        # Move right.
        elif action == 1:
            self.cur_pos += 1

        # The environment only ever terminates when we reach the goal state.
        terminated = self.cur_pos >= self.end_pos
        truncated = False
        # Produce a random reward from [0.5, 1.5] when we reach the goal.
        reward = random.uniform(0.5, 1.5) if terminated else -0.01
        infos = {}
        return (
            np.array([self.cur_pos], np.float32),
            reward,
            terminated,
            truncated,
            infos,
        )
# ws-template-code-end


if __name__ == "__main__":
    config = (
        PPOConfig()
        .environment(
            SimpleCorridor,
            env_config={
                "corridor_length": 7,
            },
            clip_rewards=True,
        )
        .env_runners(num_env_runners=63)
        )
    config.learners(num_learners=4)
    algorithm = config.build()
    mean_reward = 0
    import time
    t0 = time.time()
    t_delta = 0
    while  mean_reward < 0.9 and t_delta < 2 * 60:
        result = algorithm.step()
        mean_reward = result["env_runners"]["episode_return_mean"]
        t_delta = time.time() - t0
        print(f"Mean reward after {t_delta} seconds: {mean_reward}")
    
    algorithm.stop()
    print(f"Final reward: {mean_reward}")