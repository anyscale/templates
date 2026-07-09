# Vendored from NVIDIA transaction-foundation-model

`src/nvidia_tokenizer/` (the tokenizer package) and `src/decoder_inference.py` are vendored
verbatim from https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model
(`src/tokenizer/`, `src/decoder_inference.py`), Apache-2.0. Per-file SPDX headers are preserved.

Local changes (both guard-only; GPU behavior unchanged):
1. `numerical.py` imports `KBinsDiscretizer` with a sklearn fallback so the package works
   without RAPIDS `cuml` installed (the default `amount_strategy="fixed"` never calls it).
2. `src/decoder_inference.py` guards `pin_memory()` / `torch.cuda.synchronize()` /
   `torch.cuda.empty_cache()` behind a device check so `extract_embeddings_batched` runs
   on CPU-only workers (mini/CI); the CUDA path is identical to upstream.

`cudf`/`cupy` are still required for the GPU reference path (see requirements.txt); the
notebooks' distributed data stages use `src/nvtokenize_cpu.py`, a pandas mirror verified
byte-identical (PLAN_RAY_DATA.md Stage 0).
