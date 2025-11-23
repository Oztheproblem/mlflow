import time

import mlflow
import requests

# === 1. MLflow model server URL ===
URL = "http://127.0.0.1:5001/invocations"

# === 2. Single sample with 10 features (diabetes dataset style) ===
# You can tweak these numbers later if you like.
features = [0.05, 0.03, 0.02, 0.04, 0.01, 0.02, 0.05, 0.04, 0.03, 0.02]

payload = {
    "inputs": [features]  # MLflow 2.x scoring format: list of rows
}

# === 3. Call the served model ===
response = requests.post(URL, json=payload)

# If there’s any error, show the full response so we can debug
try:
    response.raise_for_status()
except requests.exceptions.HTTPError:
    print("❌ Error from model server:")
    print("Status code:", response.status_code)
    print("Response body:", response.text)
    raise

result = response.json()
print("✅ Raw server response:", result)

# The sklearn pyfunc wrapper returns {"predictions": [value]}
prediction_value = float(result["predictions"][0])
print(f"✅ Parsed prediction value: {prediction_value}")

# === 4. Log this live prediction to MLflow ===
mlflow.set_experiment("live_predictions_log")

with mlflow.start_run(run_name=f"live_prediction_{int(time.time())}"):
    # Log the 10 input features
    for i, val in enumerate(features, start=1):
        mlflow.log_param(f"feature_{i}", val)

    # Log the prediction as a metric
    mlflow.log_metric("prediction", prediction_value)

    # Log a simple timestamp param
    mlflow.log_param("timestamp", int(time.time()))

print("📈 Live prediction logged to 'live_predictions_log' experiment.")
