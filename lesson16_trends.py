from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd


def load_monitoring_runs(experiment_name: str = "drift_monitoring") -> pd.DataFrame:
    """
    Load all runs from the drift_monitoring experiment as a DataFrame.
    """
    print(f"🔍 Loading runs from experiment: {experiment_name!r}")
    runs_df = mlflow.search_runs(experiment_names=[experiment_name])

    if runs_df.empty:
        raise SystemExit(
            "⚠️ No runs found in 'drift_monitoring'. "
            "Run lesson15_monitoring.py a few times first."
        )

    # Convert timestamp
    runs_df["start_time"] = pd.to_datetime(
        runs_df["start_time"], unit="ms", utc=True
    ).dt.tz_convert("Europe/London")  # adjust to your timezone if needed

    # Keep only the columns we care about
    cols = [
        "run_id",
        "start_time",
        "metrics.mean_prediction",
        "metrics.median_prediction",
        "metrics.std_prediction",
        "metrics.min_prediction",
        "metrics.max_prediction",
    ]
    missing = [c for c in cols if c not in runs_df.columns]
    if missing:
        raise SystemExit(
            f"Some expected metrics are missing from runs: {missing}. "
            "Make sure lesson15_monitoring.py is logging them."
        )

    return runs_df[cols].sort_values("start_time").reset_index(drop=True)


def save_history_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save monitoring history to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"💾 Saved monitoring history to: {path}")


def plot_mean_prediction(df: pd.DataFrame, png_path: Path) -> None:
    """
    Plot mean prediction over time and save as PNG.
    """
    plt.figure()
    plt.plot(df["start_time"], df["metrics.mean_prediction"], marker="o")
    plt.xlabel("Time")
    plt.ylabel("Mean prediction")
    plt.title("Mean prediction over time (drift_monitoring)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path)
    plt.close()
    print(f"📈 Saved trend plot to: {png_path}")


def main():
    project_root = Path(__file__).resolve().parent

    # 1) Load monitoring runs
    df = load_monitoring_runs(experiment_name="drift_monitoring")

    # 2) Save CSV
    csv_path = project_root / "monitoring_history.csv"
    save_history_csv(df, csv_path)

    # 3) Save trend plot
    png_path = project_root / "monitoring_mean_prediction.png"
    plot_mean_prediction(df, png_path)

    print("✅ Lesson 16 complete: history CSV + trend plot created.")


if __name__ == "__main__":
    main()
