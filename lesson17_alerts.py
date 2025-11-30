from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd

EXPERIMENT_MONITORING = "drift_monitoring"
EXPERIMENT_ALERTS = "drift_alerts"

# How many latest runs to use as "history" for the dynamic threshold
BASELINE_WINDOW = 20
STD_MULTIPLIER = 3.0

ALERT_LOG_FILE = "alerts.log"


def load_monitoring_runs(experiment_name: str = EXPERIMENT_MONITORING) -> pd.DataFrame:
    """
    Load all runs from the monitoring experiment and return a sorted DataFrame.
    """
    print(f"🔍 Loading runs from experiment: {experiment_name!r}")
    runs_df = mlflow.search_runs(experiment_names=[experiment_name])

    if runs_df.empty:
        raise SystemExit(
            "⚠️ No runs found in 'drift_monitoring'. "
            "Run lesson15_monitoring.py a few times first."
        )

    runs_df["start_time"] = pd.to_datetime(runs_df["start_time"], unit="ms", utc=True)

    cols = [
        "run_id",
        "start_time",
        "metrics.mean_prediction",
    ]
    missing = [c for c in cols if c not in runs_df.columns]
    if missing:
        raise SystemExit(
            f"Some expected metrics are missing from runs: {missing}. "
            "Make sure lesson15_monitoring.py is logging 'mean_prediction'."
        )

    runs_df = runs_df[cols].sort_values("start_time").reset_index(drop=True)
    return runs_df


def compute_dynamic_threshold(df: pd.DataFrame) -> tuple[float, float, float]:
    """
    Use the last N runs to compute a dynamic threshold:
    threshold = baseline_mean + STD_MULTIPLIER * baseline_std
    """
    history = df.tail(BASELINE_WINDOW)
    mean_series = history["metrics.mean_prediction"]

    baseline_mean = float(mean_series.mean())
    baseline_std = float(mean_series.std(ddof=0))  # population std
    threshold = baseline_mean + STD_MULTIPLIER * baseline_std

    return baseline_mean, baseline_std, threshold


def log_alert_to_file(message: str, log_path: Path) -> None:
    """
    Append an alert message to alerts.log with a timestamp.
    """
    timestamp = datetime.utcnow().isoformat()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp} UTC] {message}\n")
    print(f"📝 Logged alert to {log_path}")


def log_alert_to_mlflow(
    latest_mean: float,
    baseline_mean: float,
    baseline_std: float,
    threshold: float,
    alert_triggered: bool,
) -> None:
    """
    Log alert check as an MLflow run in the 'drift_alerts' experiment.
    """
    mlflow.set_experiment(EXPERIMENT_ALERTS)
    with mlflow.start_run(run_name="lesson17_drift_alert"):
        mlflow.log_metric("latest_mean_prediction", latest_mean)
        mlflow.log_metric("baseline_mean", baseline_mean)
        mlflow.log_metric("baseline_std", baseline_std)
        mlflow.log_metric("dynamic_threshold", threshold)
        mlflow.log_metric("alert_triggered", int(alert_triggered))


def main():
    project_root = Path(__file__).resolve().parent
    log_path = project_root / ALERT_LOG_FILE

    # 1) Load monitoring runs
    df = load_monitoring_runs()

    # 2) Latest monitoring run
    latest = df.iloc[-1]
    latest_mean = float(latest["metrics.mean_prediction"])
    latest_time = latest["start_time"]

    # 3) Compute dynamic threshold from recent history
    baseline_mean, baseline_std, threshold = compute_dynamic_threshold(df)

    # 4) Decide if alert should trigger
    alert_triggered = latest_mean > threshold

    status = "TRIGGERED" if alert_triggered else "OK"
    message = (
        f"Alert status: {status} | "
        f"latest_mean={latest_mean:.4f}, "
        f"baseline_mean={baseline_mean:.4f}, "
        f"baseline_std={baseline_std:.4f}, "
        f"threshold={threshold:.4f}, "
        f"time={latest_time}"
    )

    print("➕ Drift check result:")
    print(message)

    # 5) Log to file
    log_alert_to_file(message, log_path)

    # 6) Log to MLflow
    log_alert_to_mlflow(
        latest_mean=latest_mean,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        threshold=threshold,
        alert_triggered=alert_triggered,
    )

    print("✅ Lesson 17 complete: alert check recorded in alerts.log and MLflow.")


if __name__ == "__main__":
    main()
