# Reinforcement Learning with RLlib

**⏱️ Time to complete**: 5 min

RLlib is Ray's library for reinforcement learning (RL). Built on Ray, it is highly scalable and fault-tolerant.
This template walks you through running a quick RL training. Specifically, we use RLlib's main APIs for defining a training workload, kick it off and check it's metrics. In addition, you can explore common use-cases like introducing your own environment. 

## Prerequisite
- This template requires a concenptual understanding of RL.
- Install the required libraries with the following command
```shell
pip install "ray[rllib]" torch "gymnasium[atari,accept-rom-license,mujoco]" python-opencv-headless
```

## Learning Pong from images

We start by learning the classic [Pong](https://en.wikipedia.org/wiki/Pong) video game.
The expectation is that the cluster scales to 4 GPUs and 96 CPUs for around 5 minutes to learn Pong.
You can run the training script with the following command:

```shell
python atari_ppo.py
```

While the training is running, have a look at the `atari_ppo.py` file to see how we build a `config` and then call `config.build()` to build an algorithm object and interact with it. You don't have to get every aspect of it just now.

Here is what you should learn from looking at `atari_ppo.py`:
- RLlib uses a builder pattern to declare all aspects of your RL training in a config first. 
    - We start by creating a config object of our algorithm of choice - in this case `PPOConfig()`. 
    - With APIs like `config.training()`, we adjust it.
- Creating an algorithm object from a config solidifies the configuration.
    - Environments are created and resources requested.
    - You can iterate on calling `algorithm.step()`, which will tell RLlib to step through the environment and learn on a set number of experiences before returning.
    - Finally, you can evaluate the trained algorithm with `algorithm.evaluate()`.

**NOTE:** You have already run distributed training now across 4 GPUs and 96 CPUs.
The 4 GPUs are used to calculate updates to the Algorithm's ANNs while 95 CPUs are used to run the environments and collect experiences in parallel.