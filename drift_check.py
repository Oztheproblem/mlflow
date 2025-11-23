import time

import mlflow
import numpy as np
import pandas as pd

EXPERIMENT_NAME = "live_predictions_log"

# 1. Find the experiment
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    print(f"❌ Experiment '{EXPERIMENT_NAME}' not found.")
    raise SystemExit()

experiment_id = experiment.experiment_id

# 2. Time window: last 24 hours (in seconds)
now_ts = int(time.time())
lookback_seconds = 24 * 60 * 60
cutoff_ts = now_ts - lookback_seconds

# 3. Load runs (NO filter_string – we filter with pandas)
runs_df = mlflow.search_runs(
    [experiment_id],
    max_results=1000,
)

if runs_df.empty:
    print("⚠️ No runs found in experiment.")
    raise SystemExit()

# start_time is datetime64[ns, UTC] in your setup
# so we make a matching datetime for the cutoff
cutoff_dt = pd.to_datetime(cutoff_ts, unit="s", utc=True)

recent_runs = runs_df[runs_df["start_time"] >= cutoff_dt]

if recent_runs.empty:
    print("⚠️ No predictions found in the last 24 hours.")
    raise SystemExit()

# 4. Extract prediction metric
if "metrics.prediction" not in recent_runs.columns:
    print("⚠️ No 'prediction' metric found in recent runs.")
    print(
        "   Check that predict_live.py is logging mlflow.log_metric('prediction', value)."
    )
    raise SystemExit()

preds = recent_runs["metrics.prediction"].astype(float)

mean_val = float(np.mean(preds))
median_val = float(np.median(preds))
std_val = float(np.std(preds))
min_val = float(np.min(preds))
max_val = float(np.max(preds))

print("📊 Drift stats over recent predictions:")
print(f"  Mean:   {mean_val:.4f}")
print(f"  Median: {median_val:.4f}")
print(f"  Std:    {std_val:.4f}")
print(f"  Min:    {min_val:.4f}")
print(f"  Max:    {max_val:.4f}")

# 5. Log drift metrics into a dedicated experiment
mlflow.set_experiment("drift_monitoring")

with mlflow.start_run(run_name=f"drift_check_{now_ts}"):
    mlflow.log_metric("mean_prediction", mean_val)
    mlflow.log_metric("median_prediction", median_val)
    mlflow.log_metric("std_prediction", std_val)
    mlflow.log_metric("min_prediction", min_val)
    mlflow.log_metric("max_prediction", max_val)

print("✅ Drift metrics logged to 'drift_monitoring' experiment.")
