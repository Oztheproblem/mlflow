import mlflow
from mlflow.tracking import MlflowClient

# Connect to the MLflow tracking server
client = MlflowClient()

# Define your experiment name
experiment_name = "lesson_8_model_logging"
experiment = client.get_experiment_by_name(experiment_name)

# Get all runs for this experiment
runs = client.search_runs(experiment.experiment_id, order_by=["metrics.mse ASC"])

# Pick the best model (lowest mean squared error)
best_run = runs[0]
best_run_id = best_run.info.run_id
best_mse = best_run.data.metrics["mse"]

print(f"Best model found: Run ID = {best_run_id}, MSE = {best_mse}")

# Register the model (if not already done)
model_uri = f"runs:/{best_run_id}/model"
model_name = "Lesson11_AutoPromoteModel"

# Ensure the model name exists in the registry
try:
    client.create_registered_model(model_name)
    print(f"Created new registered model: {model_name}")
except mlflow.exceptions.RestException:
    print(f"Model {model_name} already exists in the registry.")

try:
    mv = client.create_model_version(
        name=model_name, source=model_uri, run_id=best_run_id
    )
    print(f"Model registered as version {mv.version}")
except mlflow.exceptions.RestException:
    print("Model already exists, retrieving existing version...")

# Promote the model to Production (this simulates CI/CD promotion)
latest_versions = client.get_latest_versions(model_name, stages=["None", "Staging"])
if latest_versions:
    for v in latest_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=v.version,
            stage="Production",
            archive_existing_versions=True,
        )
    print(f"Model {model_name} promoted to Production ✅")
else:
    print("No model versions found to promote.")
