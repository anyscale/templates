#!/usr/bin/env bash
set -euxo pipefail

pip install -q papermill

# The shipped config is 1000 samples x 5 epochs = 625 steps, ~25 min on one L4. CI
# needs the pipeline to work, not the adapter to be good. The notebook copies this
# file to shared storage at runtime, so patching it here leaves the template alone.
python3 - <<'PY'
import yaml

path = "lora_sft_ray.yaml"
config = yaml.safe_load(open(path))
config.update(num_train_epochs=1.0, max_samples=200, save_steps=25, eval_steps=25)
yaml.safe_dump(config, open(path, "w"), sort_keys=False)
PY

papermill README.ipynb /tmp/entity-recognition-with-llms.out.ipynb --log-output --kernel python3 --cwd .
