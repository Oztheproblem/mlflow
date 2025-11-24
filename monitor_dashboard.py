from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

# ----------------------------
# 1. Load live predictions
# ----------------------------
live_df = mlflow.search_runs(experiment_names=["live_predictions_log"])

if live_df.empty:
    print("⚠️ No live predictions found.")
    exit()

# Extract fields
live_df["timestamp"] = pd.to_datetime(live_df["start_time"], unit="ms", utc=True)
live_df["prediction"] = live_df["metrics.prediction"]

# Last hour window
cutoff = datetime.now(tz=live_df["timestamp"].dt.tz) - timedelta(hours=1)
recent = live_df[live_df["timestamp"] >= cutoff]

# Summary stats
latest_pred = recent["prediction"].iloc[-1]
mean_pred = recent["prediction"].mean()
std_pred = recent["prediction"].std()
min_pred = recent["prediction"].min()
max_pred = recent["prediction"].max()


# ----------------------------
# 2. Load drift results
# ----------------------------
drift_df = mlflow.search_runs(experiment_names=["drift_monitoring"])

if drift_df.empty:
    print("⚠️ No drift data found.")
    drift_detected = False
else:
    drift_detected = drift_df["params.drift_detected"].iloc[-1] == "True"


# ----------------------------
# 3. Basic anomaly detection
# ----------------------------

# Rule: if latest prediction deviates more than 3 std from mean
anomaly = abs(latest_pred - mean_pred) > (3 * std_pred if std_pred is not None else 0)


# ----------------------------
# 4. Create dashboard chart
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(recent["timestamp"], recent["prediction"], marker="o")
plt.title("Model Performance Dashboard")
plt.xlabel("Time")
plt.ylabel("Prediction")
plt.grid()

chart_path = "monitor_dashboard.png"
plt.savefig(chart_path)
plt.close()


# ----------------------------
# 5. Log dashboard to MLflow
# ----------------------------
mlflow.set_experiment("monitoring_dashboard")

with mlflow.start_run():
    mlflow.log_metric("latest_prediction", latest_pred)
    mlflow.log_metric("mean_prediction", mean_pred)
    mlflow.log_metric("std_prediction", std_pred if std_pred else 0)
    mlflow.log_metric("min_prediction", min_pred)
    mlflow.log_metric("max_prediction", max_pred)
    mlflow.log_metric("drift_detected", int(drift_detected))
    mlflow.log_metric("anomaly_detected", int(anomaly))

    mlflow.log_artifact(chart_path)

print("✅ Monitoring dashboard updated & logged to MLflow!")
