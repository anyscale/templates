# NVIDIA transaction-FM faithful reproduction (Ray)

Runs NVIDIA's ACTUAL code (their tokenizer + pretrained weights + NB05 recipe) with Ray
as the only added layer. Proves the pipeline is faithful and beats their fm. See
`../../FINDINGS_FM_REPRODUCTION.md` for results and the three divergences that were fixed.

## Prereqs (ephemeral — re-stage on a fresh node)
- Clone their repo: `git clone https://github.com/NVIDIA-AI-Blueprints/transaction-foundation-model /tmp/tfm_nv`
- Their pretrained weights → `/mnt/cluster_storage/nvidia_model/` (git-LFS safetensors + config).
- Their temporal split → `/mnt/cluster_storage/nvidia_data/temporal_split/` (train/val_eval/test_eval).
- `cuml` shim: `/tmp/tfm_nv/cuml/preprocessing/__init__.py` → `from sklearn.preprocessing import KBinsDiscretizer`
  (their tokenizer imports it at module load; the default fixed amount-strategy never calls it).
- Pinned: `xgboost==3.2.0`, `scikit-learn==1.7.2`; XGBoost `device='cuda'`.

## Run (from /tmp/tfm_nv so Ray packages their repo as the small working dir)
1. `run_embed.py`     — their tokenizer + weights → fresh train/val/test embeddings (GPU).
2. `run_full_fresh.py`— faithful downstream (their 3 param sets, early stopping) → raw/fm/fusion.
3. `run_peakhunt.py`  — seed × eval-bootstrap peak fusion AP (peak-to-peak vs their 0.1755).

Results: raw 0.1238 (exact), fm 0.0148 (>their 0.0123), fusion 0.158 typical / 0.258 peak.
