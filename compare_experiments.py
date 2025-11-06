import mlflow
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define a list of alpha values to test
alphas = [0.1, 0.5, 1.0, 2.0]

mlflow.set_experiment("lesson_9_model_comparison")

for alpha in alphas:
    with mlflow.start_run(run_name=f"ridge_alpha_{alpha}"):
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        # Log params and metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_metric("mse", mse)

        print(f"Alpha={alpha} | MSE={mse:.2f}")

print("✅ All experiments logged successfully!")
