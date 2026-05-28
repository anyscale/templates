#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill==2.7.0

# Minimal-scale run of all 7 tutorial notebooks: the env knobs below shrink epochs /
# dataset size / boost rounds (defaults in each notebook are the real customer config).
# Order matters: the intro carries the one-time torch install, and 04a populates the
# Food-101 HF cache that 04d1 reuses.
export EPOCHS=1 FOOD101_PCT=1 NUM_BOOST_ROUND=10 PENDULUM_STEPS=1000
for nb in \
  "getting-started/01_02_03_intro_to_ray_train" \
  "workload-patterns/04a_vision_pattern" \
  "workload-patterns/04b_tabular_workload_pattern" \
  "workload-patterns/04c_time_series_workload_pattern" \
  "workload-patterns/04d1_generative_cv_pattern" \
  "workload-patterns/04d2_policy_learning_pattern" \
  "workload-patterns/04e_rec_sys_workload_pattern"
do
  papermill "${nb}.ipynb" "/tmp/$(basename "${nb}").out.ipynb" --log-output --kernel python3 --cwd .
done
