# FinTech Batch Fraud Risk Scoring Pipeline

**⏱️ Time to complete**: 30 min
### Anyscale Technical Demo — Ray Data on Anyscale Jobs

---

## Business Context

Fintech companies run nightly or hourly batch jobs to score every transaction for fraud risk. These pipelines combine **CPU-intensive feature engineering** (aggregation windows, velocity checks, device fingerprinting) with **ML model inference** (XGBoost). Current Spark-based solutions are slow, expensive, and operationally complex.

**Today:** A Spark + Hive pipeline that takes **6+ hours** for 10M transactions, requires a dedicated Spark cluster, and if it fails at hour 5 — you start over.

**With Ray Data on Anyscale:** The same job runs in **minutes**, with streaming execution, automatic fault tolerance, and full observability.

---

## Architecture

```
Anyscale Job / Workspace
│
├─ [Ray Data] read_parquet ─────────── Distributed parallel read
├─ [map_batches / CPU] join ────────── Broadcast join with pre-computed aggregates
├─ [map_batches / CPU] features ────── 21 engineered features
├─ [map_batches / CPU] FraudScorer ─── XGBoost inference + risk classification
└─ [write_parquet] ─────────────────── Scored transactions + alerts
```

Data streams between stages — no intermediate disk writes, no idle workers.

---

**Demo timing:** This notebook is designed for a **10-15 minute** live walkthrough.
- **Min 0-2:** Problem framing (this cell)
- **Min 2-4:** Inspect data + model metrics (pre-computed)
- **Min 4-8:** Run the scoring pipeline (live execution)
- **Min 8-11:** Results & validation
- **Min 11-13:** Fault tolerance & observability
- **Min 13-15:** Path to production, Spark comparison

## Step 1: Connect to the Ray Cluster

In an Anyscale Workspace, Ray is **pre-initialized** — no cluster setup, no YARN config, no Spark context.

The workspace launched with `workspace.yaml` which specified:
- **Head node:** m5.4xlarge (16 vCPU, 64 GB)
- **Workers:** 2-20 x m5.4xlarge (autoscale on demand)
- **No GPUs needed** — XGBoost is CPU-optimized


```python
!pip install -q -r requirements.txt
```


```python
import sys, os

# Point to the fraud-risk root so src.* and scripts.* are importable
DEMO_ROOT = os.path.abspath(os.getcwd())
if DEMO_ROOT not in sys.path:
    sys.path.insert(0, DEMO_ROOT)

import ray

# In Anyscale Workspace, Ray is pre-initialized.
ray.init(
    ignore_reinit_error=True,
    runtime_env={"working_dir": DEMO_ROOT},
)

resources = ray.cluster_resources()
print("Ray cluster resources:")
for resource, count in sorted(resources.items()):
    if not resource.startswith('node:'):
        print(f"  {resource:<20} {count}")

nodes = ray.nodes()
print(f"\nCluster nodes: {len(nodes)}")
for n in nodes:
    res = ', '.join(f"{k}={v}" for k, v in n['Resources'].items() if not k.startswith('node:'))
    print(f"  {n['NodeManagerAddress']:<20} alive={n['Alive']}  {res}")
```

## Step 2: Pre-Run Setup (Skip in Live Demo)

The cell below generates synthetic transaction data and trains the XGBoost model. These are **pre-run setup steps** that take several minutes.

> **For live demo:** Artifacts should already exist under the demo base directory (`/mnt/shared_storage/fintech-demo` in a typical workspace, or `./demo_data` when cluster storage is not used). Run the cell — it will detect existing artifacts and skip. If artifacts are missing, it will generate them (takes ~2-3 min for medium scale).


```python
# Pre-run setup: generate data + train model (skip if artifacts exist)
from src.data_generator import save_dataset, SCALE_MAP
from src.paths import get_demo_base_dir

BASE_DIR = get_demo_base_dir()
SCALE = "medium"  # 1M transactions

RAW_DIR = f"{BASE_DIR}/raw/{SCALE}"
MODEL_PATH = f"{BASE_DIR}/model/fraud_model.json"
METRICS_PATH = f"{BASE_DIR}/model/fraud_model_metrics.json"
FEATURES_PATH = f"{BASE_DIR}/features/{SCALE}/train_features.parquet"
OUTPUT_PATH = f"{BASE_DIR}/scored/{SCALE}/"

TXN_PATH = os.path.join(RAW_DIR, "transactions.parquet")
USER_AGG_PATH = os.path.join(RAW_DIR, "user_aggregates.parquet")
MERCHANT_AGG_PATH = os.path.join(RAW_DIR, "merchant_aggregates.parquet")

# --- Step 1: Generate data ---
if os.path.exists(TXN_PATH):
    print(f"Artifacts already exist — skipping data generation.")
    print(f"  Transactions: {TXN_PATH}")
else:
    print("Generating synthetic transaction data...")
    save_dataset(RAW_DIR, num_transactions=SCALE_MAP[SCALE])

# --- Step 2: Train model ---
if os.path.exists(MODEL_PATH):
    print(f"\nTrained model already exists — skipping training.")
    print(f"  Model: {MODEL_PATH}")
else:
    print("\nTraining XGBoost fraud detection model...")
    from scripts import __init__  # noqa — ensure scripts is a package
    # Import the training helper from 02_train_model
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_script",
        os.path.join(DEMO_ROOT, "scripts", "02_train_model.py"),
    )
    train_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_script)

    # Feature-engineer training data
    train_script.prepare_training_features(RAW_DIR, FEATURES_PATH)
    # Train model
    from src.model_training import train_fraud_model
    train_fraud_model(FEATURES_PATH, MODEL_PATH)

print("\nSetup complete. Ready for live demo.")
```

## Step 3: Inspect the Transaction Data

Let's look at what we're scoring: **1M synthetic financial transactions** with realistic fraud patterns.

- **50K unique users**, each with a home country and spending profile
- **10K merchants** across 7 categories
- **2% fraud rate** with 5 distinct fraud patterns (velocity, amount anomaly, geographic, off-hours, card testing)


```python
import pandas as pd
import numpy as np

# Load a sample to inspect (fast read)
df = pd.read_parquet(TXN_PATH)

print(f"Dataset shape: {df.shape[0]:,} transactions x {df.shape[1]} columns")
print(f"\nSchema:")
for col in df.columns:
    print(f"  {col:<22} {df[col].dtype}")

print(f"\nDate range: {df['timestamp'].min()} → {df['timestamp'].max()}")
print(f"\nFraud rate: {df['is_fraud'].mean()*100:.2f}% ({df['is_fraud'].sum():,} fraudulent transactions)")

print(f"\nAmount statistics:")
print(f"  Mean:   ${df['amount'].mean():,.2f}")
print(f"  Median: ${df['amount'].median():,.2f}")
print(f"  Max:    ${df['amount'].max():,.2f}")

print(f"\nMerchant category distribution:")
for cat, count in df['merchant_category'].value_counts().items():
    pct = count / len(df) * 100
    print(f"  {cat:<14} {count:>8,}  ({pct:.1f}%)")

print(f"\nSample transactions:")
df.head(5)
```

## Step 4: Review the Trained Model

The XGBoost model was trained on **21 engineered features** spanning:
- **Transaction-level:** log amount, round amount flag, time-of-day, day-of-week, off-hours flag
- **User aggregates:** average/std spending, z-score, velocity (1h/24h counts), merchant/country diversity
- **Merchant-level:** historical fraud rate, average transaction amount
- **Categorical:** encoded merchant category, card type, device type, country risk score
- **Interaction:** amount x velocity, z-score x international flag


```python
import json

# Load pre-computed model metrics
with open(METRICS_PATH, "r") as f:
    model_metrics = json.load(f)

print(f"{'=' * 50}")
print(f"  TRAINED MODEL METRICS")
print(f"{'=' * 50}")
print(f"  AUC-ROC:           {model_metrics['auc_roc']:.4f}")
print(f"  Precision:         {model_metrics['precision']:.4f}")
print(f"  Recall:            {model_metrics['recall']:.4f}")
print(f"  F1 Score:          {model_metrics['f1_score']:.4f}")
print(f"  Training samples:  {model_metrics['train_samples']:,}")
print(f"  Validation samples:{model_metrics['val_samples']:,}")
print(f"  Val fraud count:   {model_metrics['val_fraud_count']:,}")
print(f"  Best iteration:    {model_metrics['best_iteration']}")
print(f"{'=' * 50}")
print(f"\nModel path: {MODEL_PATH}")
```

## Step 4b (Optional): Tune XGBoost Hyperparameters with Ray Tune + ASHA

Use this optional step to run parallel hyperparameter search with **Ray Tune** and prune weak trials early with **ASHA**, based on the Ray docs pattern:
https://docs.ray.io/en/latest/tune/examples/tune-xgboost.html

- **Why now:** Fraud behavior drifts fast, so model quality decays unless we retrain and retune regularly.
- **Move faster:** Parallel trials shrink iteration cycles from days to hours.
- **Spend smarter:** ASHA stops underperforming trials early, focusing compute on promising configs.
- **Operationally simple:** Same code scales across the Ray cluster, then we keep the best model for Step 5+.



```python
# Optional: hyperparameter tuning with Ray Tune + XGBoost + ASHA.
# Skip this cell for the standard fast demo path.
# If imports fail, run: %pip install "ray[tune]>=2.9"
import os
import uuid

from ray import tune
from ray.tune.schedulers import ASHAScheduler

from src.model_training import train_fraud_model

TUNE_NUM_SAMPLES = 20
TUNE_MAX_BOOST_ROUNDS = 40
TRIAL_CPUS = 2

def tune_fraud_model(config):
    xgb_overrides = {
        "max_depth": int(config["max_depth"]),
        "learning_rate": float(config["learning_rate"]),
        "min_child_weight": int(config["min_child_weight"]),
        "subsample": float(config["subsample"]),
        "colsample_bytree": float(config["colsample_bytree"]),
        "gamma": float(config["gamma"]),
        "alpha": float(config["reg_alpha"]),
        "lambda": float(config["reg_lambda"]),
    }

    trial_model_path = os.path.join(
        BASE_DIR,
        "model",
        "tune_trials",
        uuid.uuid4().hex,
        "fraud_model.json",
    )

    metrics = train_fraud_model(
        train_features_path=FEATURES_PATH,
        model_output_path=trial_model_path,
        xgb_params=xgb_overrides,
        num_boost_round=int(config["num_boost_round"]),
        early_stopping_rounds=30,
        tune_report=True,
        tune_report_frequency=5,
    )

    # Final summary metrics for the trial.
    tune.report(
        auc=metrics["auc_roc"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1_score"],
    )

search_space = {
    "max_depth": tune.randint(3, 10),
    "learning_rate": tune.loguniform(1e-2, 2e-1),
    "min_child_weight": tune.randint(1, 8),
    "subsample": tune.uniform(0.6, 1.0),
    "colsample_bytree": tune.uniform(0.6, 1.0),
    "gamma": tune.uniform(0.0, 5.0),
    "reg_alpha": tune.loguniform(1e-4, 1.0),
    "reg_lambda": tune.loguniform(1e-3, 10.0),
    "num_boost_round": tune.choice([150, 200, 300, 400]),
}

scheduler = ASHAScheduler(
    metric="auc",
    mode="max",
    max_t=TUNE_MAX_BOOST_ROUNDS,
    grace_period=30,
    reduction_factor=2,
)

tuner = tune.Tuner(
    tune.with_resources(tune_fraud_model, resources={"cpu": TRIAL_CPUS}),
    param_space=search_space,
    tune_config=tune.TuneConfig(
        num_samples=TUNE_NUM_SAMPLES,
        scheduler=scheduler,
    ),
    run_config=tune.RunConfig(name="fraud_risk_xgboost_tuning", storage_path="/mnt/shared_storage/fintech-demo/tune"),
)

results = tuner.fit()
best_result = results.get_best_result(metric="auc", mode="max")
best_config = best_result.config

print("Best trial metrics:")
print(f"  AUC:       {best_result.metrics.get('auc', float('nan')):.4f}")
print(f"  Precision: {best_result.metrics.get('precision', float('nan')):.4f}")
print(f"  Recall:    {best_result.metrics.get('recall', float('nan')):.4f}")
print(f"  F1:        {best_result.metrics.get('f1', float('nan')):.4f}")

print("\nBest hyperparameters:")
for k, v in best_config.items():
    print(f"  {k}: {v}")

best_xgb_params = {
    "max_depth": int(best_config["max_depth"]),
    "learning_rate": float(best_config["learning_rate"]),
    "min_child_weight": int(best_config["min_child_weight"]),
    "subsample": float(best_config["subsample"]),
    "colsample_bytree": float(best_config["colsample_bytree"]),
    "gamma": float(best_config["gamma"]),
    "alpha": float(best_config["reg_alpha"]),
    "lambda": float(best_config["reg_lambda"]),
}

TUNED_MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model_tuned.json")
best_metrics = train_fraud_model(
    train_features_path=FEATURES_PATH,
    model_output_path=TUNED_MODEL_PATH,
    xgb_params=best_xgb_params,
    num_boost_round=int(best_config["num_boost_round"]),
    early_stopping_rounds=30,
)

# Use tuned model for downstream scoring steps.
MODEL_PATH = TUNED_MODEL_PATH
METRICS_PATH = TUNED_MODEL_PATH.replace(".json", "_metrics.json")

print(f"\nTuned model saved -> {MODEL_PATH}")
print(f"Tuned metrics saved -> {METRICS_PATH}")
print(f"Tuned model AUC-ROC -> {best_metrics['auc_roc']:.4f}")

```

## Step 5: Run the Fraud Scoring Pipeline

This is the **main event**. We'll score all 1M transactions through the full Ray Data pipeline:

1. **Read** — distributed parallel Parquet read
2. **Join** — broadcast join with pre-computed user + merchant aggregates
3. **Features** — compute 21 engineered features per transaction
4. **Score** — XGBoost inference with risk tier classification
5. **Write** — scored transactions to Parquet output

> **TIP:** Open the **Ray Dashboard** (link in Anyscale Workspace header) and watch:
> - **Jobs tab:** pipeline stages executing
> - **Cluster tab:** CPU workers activating across nodes
> - **Metrics tab:** throughput and backpressure


```python
from src.pipeline import run_fraud_scoring_pipeline

NUM_WORKERS = 10

print("Starting fraud scoring pipeline...")
print(f"  Input:       {TXN_PATH}")
print(f"  Output:      {OUTPUT_PATH}")
print(f"  Model:       {MODEL_PATH}")
print(f"  Workers:     {NUM_WORKERS}")
print(f"\nTIP: Open Ray Dashboard -> Jobs to watch data streaming through stages.\n")

metrics = run_fraud_scoring_pipeline(
    input_path=TXN_PATH,
    output_path=OUTPUT_PATH,
    model_path=MODEL_PATH,
    user_features_path=USER_AGG_PATH,
    merchant_features_path=MERCHANT_AGG_PATH,
    num_workers=NUM_WORKERS,
)
```

## Step 6: Inspect Results — Risk Distribution & Model Accuracy

Let's load the scored output and validate:
- **Risk tier distribution:** How many transactions fall into low / medium / high / critical?
- **Model accuracy:** AUC-ROC and precision/recall at various thresholds (since we have ground-truth labels in this synthetic dataset)


```python
from scripts.validate_results import load_scored_data, print_risk_distribution, print_model_accuracy

df_scored = load_scored_data(OUTPUT_PATH)
print(f"Loaded {len(df_scored):,} scored transactions")

print_risk_distribution(df_scored)
print_model_accuracy(df_scored)
```

## Step 7: Fault Tolerance & Observability

### Fault Tolerance (built into Ray Data)
- **Automatic retry:** If a worker node dies mid-batch, Ray Data re-schedules that batch on another node — no full pipeline restart
- **Backpressure:** If scoring is slower than feature engineering, Ray Data automatically throttles upstream stages to prevent OOM
- **Checkpointing:** Output is written incrementally — a partial failure doesn't lose already-written results

### Observability (built into Anyscale)
- **Ray Dashboard:** Real-time view of data flowing through pipeline stages, per-worker throughput, memory usage
- **Anyscale Metrics:** Historical job metrics, cost tracking, cluster utilization
- **Logs:** Structured logs from every worker, searchable in Anyscale console

### Job Scheduling
- Schedule this pipeline as a **recurring Anyscale Job** (hourly, nightly)
- Cluster **auto-provisions** for the job and **scales to zero** when done
- Built-in **retry policy** (max_retries: 2 in job_config.yaml) handles transient failures

## Step 8: Path to Production

The same code you just ran interactively can be submitted as a **production Anyscale Job** with one command:

```bash
# Submit from CLI:
anyscale job submit --config-file job_config.yaml

# Scale to 10M transactions:
anyscale job submit --config-file job_config.yaml \
  --override-entrypoint 'python scripts/03_run_scoring.py --scale large --num-workers 20'
```

### Ray Data vs. Spark: Side-by-Side

| Dimension | Spark + Hive | Ray Data on Anyscale |
|-----------|-------------|---------------------|
| **10M txn scoring** | ~6 hours | ~8 minutes |
| **Execution model** | Stage-by-stage, shuffle-heavy | Streaming, no intermediate writes |
| **Fault tolerance** | Full stage restart | Per-batch retry |
| **Autoscaling** | Static cluster or slow resize | Scales in seconds, to zero when idle |
| **Cluster management** | YARN/EMR config, JVM tuning | Zero-ops (Anyscale manages it) |
| **Observability** | Spark UI + external tools | Ray Dashboard + Anyscale Metrics |
| **Code complexity** | PySpark UDFs, Hive metastore | Pure Python, same code dev → prod |


```python
from src.utils import print_metrics_table, estimate_single_node_time, estimate_spark_time

# Final metrics summary
print_metrics_table(metrics)

print("Path to production as an Anyscale Job:")
print("""
  # Submit from CLI (run from fraud-risk/ directory):
  anyscale job submit --config-file job_config.yaml

  # Scale to 10M transactions:
  anyscale job submit --config-file job_config.yaml \\
    --override-entrypoint 'python scripts/03_run_scoring.py --scale large --num-workers 20'
""")

print("Key differentiators vs. Spark:")
print("  Streaming execution  -- data flows through stages, no intermediate disk writes")
print("  Per-batch fault tolerance -- failed batches retry, no full stage restart")
print("  Autoscaling          -- workers scale to demand, then scale to zero")
print("  Zero-ops cluster     -- Anyscale manages provisioning, no YARN/EMR config")
print("  One codebase         -- same code runs in Workspace and as a scheduled Job")
```
