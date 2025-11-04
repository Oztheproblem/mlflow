import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load dataset
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Define hyperparameter
alpha = 0.5  # Ridge regression regularization parameter

# Set up experiment
mlflow.set_experiment("lesson_8_model_logging")

with mlflow.start_run(run_name="ridge_regression"):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Log parameters, metrics, and model
    mlflow.log_param("alpha", alpha)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Model trained with alpha={alpha}, MSE={mse:.2f}, R2={r2:.2f}")
