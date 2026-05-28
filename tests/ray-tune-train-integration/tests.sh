#!/usr/bin/env bash
set -euo pipefail

# Shrink the 12-trial / 5-epoch full search down to 3 trials / 2 epochs so the
# notebook finishes in <30 min on 2× A10G/L4. Each knob falls back to Max's
# original value (preserved as the os.environ.get default in every patched
# cell), so users running the notebook unmodified get the full search.
export NUM_SAMPLES="${NUM_SAMPLES:-3}"
export NUM_EPOCHS="${NUM_EPOCHS:-2}"
export GRACE_PERIOD="${GRACE_PERIOD:-1}"
export MAX_T="${MAX_T:-2}"
export RANDOM_SEARCH_STEPS="${RANDOM_SEARCH_STEPS:-2}"

# bayesian-optimization is commented out in requirements.txt (optional for
# users); install it here so CI exercises the BayesOpt branch in Cell 24
# instead of falling through its try/except ImportError.
pip install -r requirements.txt papermill bayesian-optimization==1.4.3
papermill README.ipynb output.ipynb -k python3 --log-output
