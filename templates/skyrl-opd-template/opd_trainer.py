"""On-Policy Distillation (OPD) trainer for SkyRL — the whole method in one file.

This is a small subclass of SkyRL's PPO trainer. It mirrors SkyRL's
`examples/train/on_policy_distillation/main_on_policy_distill.py`, kept here next
to the notebook so you can read the actual OPD logic without digging through the
SkyRL repo. `run_opd_gsm8k.sh` invokes this file with a GSM8K + 4xL4 config.

How OPD works, in three moves:
  1. The TEACHER is loaded into SkyRL's *reference-model* slot
     (`trainer.ref.model.path`). Normally that slot holds a frozen copy of the
     policy; here it holds the (larger) teacher instead.
  2. For every token the student samples, we then have both the teacher's
     log-prob of that token (`base_action_log_probs`) and the student's own
     (`action_log_probs`). The per-token reward is `teacher_lp - student_lp`:
     positive when the teacher likes the token more than the student does, so the
     update pulls the student toward the teacher on the student's own rollouts.
     That is reverse-KL distillation (the "k1" estimator).
  3. The advantage estimator is a no-op: the per-token reward IS the advantage
     (no GRPO group baseline, no GAE) — the KL signal is used directly.

Run it from the SkyRL repo root (it imports SkyRL internals):
    uv run --isolated --extra fsdp opd_trainer.py <config overrides>
"""
import sys

import ray
import torch

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.backends.skyrl_train.utils.ppo_utils import register_advantage_estimator
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch


class OnPolicyDistillationTrainer(RayPPOTrainer):
    """A PPO trainer whose reward is per-token agreement with the teacher."""

    def apply_reward_kl_penalty(self, data: TrainingInputBatch) -> TrainingInputBatch:
        loss_mask = data["loss_mask"]
        teacher_action_log_probs = data["base_action_log_probs"]  # teacher scores the student's tokens (ref slot)
        action_log_probs = data["action_log_probs"]               # the student's own log-probs
        # reward_t = teacher_logprob_t - student_logprob_t  (reverse-KL, per token)
        data["rewards"] = -(action_log_probs - teacher_action_log_probs) * loss_mask
        return data


@register_advantage_estimator("no_op")
def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    # advantage == reward: use the per-token KL signal directly.
    return token_level_rewards, token_level_rewards


class OnPolicyDistillationExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return OnPolicyDistillationTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    OnPolicyDistillationExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
