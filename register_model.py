import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define and train model
alpha = 0.5
model = Ridge(alpha=alpha)
model.fit(X_train, y_train)

# Set the experiment
mlflow.set_experiment("lesson_10_model_registry")

# Start MLflow run
with mlflow.start_run(run_name=f"ridge_alpha_{alpha}"):
    mlflow.log_param("alpha", alpha)
    mlflow.sklearn.log_model(model, "model")
    print("✅ Model logged successfully!")

    # Register the model to the MLflow Model Registry
    result = mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
        name="ridge_regression_model",
    )

    print(f"📦 Registered model: {result.name}")
    print(f"🧾 Version: {result.version}")
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="ridge_regression_model", version=1, stage="Staging"
)
print("🚀 Model moved to Staging!")
