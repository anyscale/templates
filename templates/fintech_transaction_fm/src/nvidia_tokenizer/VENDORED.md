# Vendored from NVIDIA transaction-foundation-model

`src/nvidia_tokenizer/` (the tokenizer package) and `src/decoder_inference.py` are vendored
verbatim from https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model
(`src/tokenizer/`, `src/decoder_inference.py`), Apache-2.0. Per-file SPDX headers are preserved.

Only local change: `numerical.py` imports `KBinsDiscretizer` with a sklearn fallback so the package
works without RAPIDS `cuml` installed (the default `amount_strategy="fixed"` never calls it).
`cudf`/`cupy` are still required (see requirements.txt).
