# ws-template-imports-start
import gymnasium as gym

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module.frame_stacking import FrameStackingEnvToModule
from ray.rllib.connectors.learner.frame_stacking import FrameStackingLearner
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack
# ws-template-imports-end

NUM_LEARNERS = 4
ENV = "ale_py:ALE/Pong-v5"

# ws-template-code-start

def _make_env_to_module_connector(env):
    return FrameStackingEnvToModule(num_frames=4)


def _make_learner_connector(input_observation_space, input_action_space):
    return FrameStackingLearner(num_frames=4)


# Create a custom Atari setup (w/o the usual RLlib-hard-coded framestacking in it).
# We would like our frame stacking connector to do this job.
def _env_creator(cfg):
    return wrap_atari_for_new_api_stack(
        gym.make(ENV, **cfg, render_mode="rgb_array"),
        # Perform frame-stacking through ConnectorV2 API.
        framestack=None,
    )


tune.register_env("env", _env_creator)

config = (
    PPOConfig()
    .environment(
        "env",
        env_config={
            # Make analogous to old v4 + NoFrameskip.
            "frameskip": 1,
            "full_action_space": False,
            "repeat_action_probability": 0.0,
        },
        clip_rewards=True,
    )
    .env_runners(
        env_to_module_connector=_make_env_to_module_connector,
    )
    .training(
        learner_connector=_make_learner_connector,
        train_batch_size_per_learner=4000,
        minibatch_size=128,
        lambda_=0.95,
        kl_coeff=0.5,
        clip_param=0.1,
        vf_clip_param=10.0,
        entropy_coeff=0.01,
        num_epochs=10,
        lr=0.00015 * NUM_LEARNERS,
        grad_clip=100.0,
        grad_clip_by="global_norm",
    )
    .rl_module(
        model_config=DefaultModelConfig(
            conv_filters=[[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]],
            conv_activation="relu",
            head_fcnet_hiddens=[256],
            vf_share_layers=True,
        ),
    )
)
# ws-template-code-end

if __name__ == "__main__":
    config.env_runners(num_env_runners=95)
    config.learners(
        num_learners=4,
        num_gpus_per_learner=1,
    )
    algorithm = config.build()
    mean_reward = 0
    import time
    t0 = time.time()
    t_delta = 0
    while  mean_reward < 20 and t_delta < 10 * 60:
        result = algorithm.train()
        mean_reward = result["env_runners"].get("episode_return_mean", 0)
        t_delta = time.time() - t0
        print(f"Mean reward after {t_delta} seconds: {mean_reward}")

    algorithm.stop()
    print(f"Final reward: {mean_reward}")