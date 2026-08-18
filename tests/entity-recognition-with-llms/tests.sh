#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill

# The shipped config trains 5 epochs = 625 steps, ~25 min on one L4. Drop to one
# epoch. Cut only the epochs, not max_samples: 1000 samples at an effective batch of
# 8 is exactly 125 steps, which is also save_steps, so the run still writes the
# checkpoint-125 that the serving cells load by name. The notebook copies this file
# to shared storage at runtime, so patching it here leaves the template alone.
python3 - <<'PY'
import yaml

path = "lora_sft_ray.yaml"
config = yaml.safe_load(open(path))
config.update(num_train_epochs=1.0)
yaml.safe_dump(config, open(path, "w"), sort_keys=False)
PY

papermill README.ipynb /tmp/entity-recognition-with-llms.out.ipynb --log-output --kernel python3 --cwd .

# Reached only on success; the test pipeline fails a "pass" that lacks it.
echo "RAYAPP_TESTS_COMPLETE"
