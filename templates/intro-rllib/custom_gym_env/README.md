# Custom gym environments

**⏱️ Time to complete**: 5 min

In this notebook, we introduce our own environment to RLlib.

## Prerequisite
- This template requires a concenptual understanding of RL.
- Install the required libraries with the following command
```shell
pip install "ray[rllib]" torch "gymnasium[atari,accept-rom-license,mujoco]" python-opencv-headless
```

## Introducing a custom environment

The expectation is that the cluster scales to 1 GPU and 32 ??????? CPUs for around 5 minutes to learn Pong.
You can run the training script with the following command:

```shell
python custom_gym_env.py
```

While the training is running, have a look at the `custom_gym_env.py` file to see how we define our own environment and apply an algorithm to it. 

Here is what you should learn from looking at `custom_gym_env.py`:
- Environments for RLlib implement the (gym.Env API)[https://gymnasium.farama.org/api/env/]
- You set an environment with `config.environment()`
