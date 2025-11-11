import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

# --------------------------------------------------
# 🧭 1. Set experiment (avoid old metadata issues)
# --------------------------------------------------
mlflow.set_experiment("lesson13_visualization_log")

# --------------------------------------------------
# 🧩 2. Load or simulate prediction data
# --------------------------------------------------
# If you have a CSV of logged predictions, load it here instead:
# df = pd.read_csv("live_predictions.csv")

# For this lesson, we’ll simulate sample predictions
timestamps = pd.date_range(start="2025-11-01", periods=10, freq="D")
predictions = np.random.uniform(120, 190, size=10)  # simulated predictions

df = pd.DataFrame({"timestamp": timestamps, "prediction": predictions})

# --------------------------------------------------
# 🧠 3. Visualize predictions over time
# --------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(df["timestamp"], df["prediction"], marker="o", linestyle="-", color="tab:blue")
plt.title("📈 Live Model Predictions Over Time")
plt.xlabel("Timestamp")
plt.ylabel("Prediction Value")
plt.grid(True)
plt.tight_layout()

# --------------------------------------------------
# 🧾 4. Log the visualization as an MLflow artifact
# --------------------------------------------------
with mlflow.start_run(run_name="lesson13_visualization") as run:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "prediction_trend.png")
        plt.savefig(output_path)
        mlflow.log_artifact(output_path, artifact_path="visualizations")

        # Optionally, log summary metrics
        mlflow.log_metric("mean_prediction", df["prediction"].mean())
        mlflow.log_metric("max_prediction", df["prediction"].max())
        mlflow.log_metric("min_prediction", df["prediction"].min())

        print(f"✅ Plot saved and logged to MLflow at: {output_path}")

# --------------------------------------------------
# 🧩 5. (Optional) Preview the chart locally
# --------------------------------------------------
plt.show()

print("\n🎯 Lesson 13 completed successfully — open MLflow UI with:")
print("   mlflow ui")
print(
    "Then visit http://127.0.0.1:5000 → 'lesson13_visualization_log' → Artifacts → visualizations → prediction_trend.png"
)
