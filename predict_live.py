import time
from datetime import datetime

import mlflow
import requests

# ✅ Step 1: Define model endpoint
url = "http://127.0.0.1:5001/invocations"

# ✅ Step 2: Example input data (10 numerical features)
inputs = [0.05, 0.03, 0.02, 0.04, 0.01, 0.02, 0.05, 0.04, 0.03, 0.02]
data = {"inputs": [inputs]}

# ✅ Step 3: Send request to model
response = requests.post(url, json=data)

# ✅ Step 4: Log prediction in MLflow
mlflow.set_experiment("live_predictions_log")

with mlflow.start_run(run_name=f"live_prediction_{int(time.time())}"):
    # Log input features and timestamp
    for i, val in enumerate(inputs):
        mlflow.log_param(f"feature_{i + 1}", val)

    mlflow.log_param("timestamp", datetime.now().isoformat())

    if response.status_code == 200:
        prediction = response.json()["predictions"][0]
        print(f"✅ Model prediction: {prediction}")

        # Log prediction output
        mlflow.log_metric("prediction_value", prediction)

    else:
        print(f"❌ Error {response.status_code}: {response.text}")
        mlflow.log_param("error", response.text)
