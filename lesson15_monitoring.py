import os
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

# -----------------------------
# Config
# -----------------------------
LIVE_EXPERIMENT = "live_predictions_log"
DRIFT_EXPERIMENT = "drift_monitoring"

# How much data to use
WINDOW_HOURS = 1  # last 1 hour for drift stats
HISTORY_HOURS = 24  # last 24 hours for dashboard plot

# Drift rule: |latest - mean| > N * std
DRIFT_STD_THRESHOLD = 3.0


def load_live_predictions():
    """Load prediction runs from MLflow and return as a cleaned DataFrame."""
    runs_df = mlflow.search_runs(experiment_names=[LIVE_EXPERIMENT])

    if runs_df.empty:
        print("⚠️ No runs found in live_predictions_log yet.")
        return None

    # Convert MLflow ms timestamps -> timezone-aware datetimes
    runs_df["start_time"] = pd.to_datetime(runs_df["start_time"], unit="ms", utc=True)

    # Extract prediction metric (logged as 'prediction' in predict_live.py)
    if "metrics.prediction" not in runs_df.columns:
        print("⚠️ Column 'metrics.prediction' not found in runs.")
        return None

    runs_df["prediction"] = runs_df["metrics.prediction"].astype(float)

    # Sort by time just to be safe
    runs_df = runs_df.sort_values("start_time").reset_index(drop=True)
    return runs_df


def compute_drift_stats(runs_df):
    """Compute drift statistics for the last WINDOW_HOURS of data."""
    now = datetime.now(timezone.utc)
    cutoff_recent = now - timedelta(hours=WINDOW_HOURS)

    recent = runs_df[runs_df["start_time"] >= cutoff_recent]

    if recent.empty:
        print(f"⚠️ No predictions in the last {WINDOW_HOURS} hour(s).")
        return None

    preds_recent = recent["prediction"]

    latest_prediction = preds_recent.iloc[-1]
    mean_prediction = preds_recent.mean()
    std_prediction = preds_recent.std(ddof=0)  # population std

    if std_prediction > 0:
        z_score = abs(latest_prediction - mean_prediction) / std_prediction
    else:
        z_score = 0.0

    drift_detected = z_score > DRIFT_STD_THRESHOLD

    stats = {
        "latest_prediction": float(latest_prediction),
        "mean_prediction": float(mean_prediction),
        "std_prediction": float(std_prediction),
        "z_score_latest": float(z_score),
        "drift_detected": int(drift_detected),
    }

    print("📊 Drift stats (last hour):")
    for k, v in stats.items():
        print(f"  - {k}: {v}")

    return stats


def build_dashboard(runs_df, stats):
    """Create a PNG dashboard summarising recent predictions & drift."""

    now = datetime.now(timezone.utc)
    cutoff_history = now - timedelta(hours=HISTORY_HOURS)
    history = runs_df[runs_df["start_time"] >= cutoff_history]

    if history.empty:
        print(f"⚠️ No history found in the last {HISTORY_HOURS} hours.")
        return None

    times = history["start_time"]
    preds = history["prediction"]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Prediction trend
    ax.plot(times, preds, marker="o", linestyle="-")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Prediction value")

    # Mean & std bands based on last WINDOW_HOURS
    mean_val = stats["mean_prediction"]
    std_val = stats["std_prediction"]

    ax.axhline(mean_val, linestyle="--", label="Mean (last hour)")
    if std_val > 0:
        upper = mean_val + DRIFT_STD_THRESHOLD * std_val
        lower = mean_val - DRIFT_STD_THRESHOLD * std_val
        ax.axhline(upper, linestyle=":", label=f"+{DRIFT_STD_THRESHOLD}σ")
        ax.axhline(lower, linestyle=":", label=f"-{DRIFT_STD_THRESHOLD}σ")

    # Highlight latest point
    latest_time = times.iloc[-1]
    latest_pred = stats["latest_prediction"]
    drift_flag = bool(stats["drift_detected"])

    ax.scatter(
        [latest_time],
        [latest_pred],
        s=80,
        marker="x",
        label="Latest prediction (drift!)" if drift_flag else "Latest prediction",
    )

    title_status = "DRIFT DETECTED 🚨" if drift_flag else "Stable ✅"
    ax.set_title(f"Lesson 15 Monitoring Dashboard — {title_status}")

    ax.legend()
    fig.autofmt_xdate()

    # Save locally so MLflow can log it
    os.makedirs("artifacts", exist_ok=True)
    dashboard_path = os.path.join("artifacts", "monitoring_dashboard.png")
    fig.savefig(dashboard_path, bbox_inches="tight")
    plt.close(fig)

    print(f"🖼 Dashboard saved to: {dashboard_path}")
    return dashboard_path


def create_alert_file(stats):
    """Create a small text file when drift is detected."""
    os.makedirs("artifacts", exist_ok=True)
    alert_path = os.path.join("artifacts", "drift_alert.txt")

    with open(alert_path, "w", encoding="utf-8") as f:
        f.write("DRIFT ALERT 🚨\n\n")
        f.write("A drift event was detected by Lesson 15 monitoring.\n\n")
        f.write("Stats (last hour):\n")
        for k, v in stats.items():
            f.write(f"- {k}: {v}\n")

    print(f"📄 Alert file created at: {alert_path}")
    return alert_path


def main():
    # Use / create the drift_monitoring experiment
    mlflow.set_experiment(DRIFT_EXPERIMENT)

    # 1. Load live predictions
    runs_df = load_live_predictions()
    if runs_df is None:
        return

    # 2. Compute drift stats for recent window
    stats = compute_drift_stats(runs_df)
    if stats is None:
        return

    # 3. Build dashboard over longer history window
    dashboard_path = build_dashboard(runs_df, stats)

    # 4. Log everything into MLflow
    run_name = f"lesson15_drift_check_{int(datetime.utcnow().timestamp())}"

    with mlflow.start_run(run_name=run_name):
        # Log metrics
        mlflow.log_metric("latest_prediction", stats["latest_prediction"])
        mlflow.log_metric("mean_prediction_last_hour", stats["mean_prediction"])
        mlflow.log_metric("std_prediction_last_hour", stats["std_prediction"])
        mlflow.log_metric("z_score_latest", stats["z_score_latest"])
        mlflow.log_metric("drift_detected", stats["drift_detected"])

        # Log dashboard artifact
        if dashboard_path is not None:
            mlflow.log_artifact(dashboard_path, artifact_path="dashboard")

        # 5. If drift detected → log alert artifact
        if stats["drift_detected"] == 1:
            alert_path = create_alert_file(stats)
            mlflow.log_artifact(alert_path, artifact_path="alerts")
            mlflow.set_tag("alert", "drift_detected")
            print("🚨 Drift detected — alert logged to MLflow.")
        else:
            print("✅ No drift detected. Monitoring run logged.")

    print("✅ Lesson 15 monitoring run completed.")


if __name__ == "__main__":
    main()
