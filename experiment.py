import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

# 1️⃣ Create some fake experiment data
epochs = list(range(1, 11))
accuracy = np.random.uniform(0.7, 0.95, 10)
loss = np.random.uniform(0.1, 0.3, 10)

# 2️⃣ Prepare a dataframe
results = pd.DataFrame({"epoch": epochs, "accuracy": accuracy, "loss": loss})

# 3️⃣ Start the MLflow run
mlflow.set_experiment("lesson_6_artifacts")
with mlflow.start_run():
    mlflow.log_param("model_type", "demo_model")
    mlflow.log_param("learning_rate", 0.01)

    for epoch, acc, l in zip(epochs, accuracy, loss):
        mlflow.log_metric("accuracy", acc, step=epoch)
        mlflow.log_metric("loss", l, step=epoch)

    # Save and log artifacts
    results.to_csv("results.csv", index=False)
    mlflow.log_artifact("results.csv")

    plt.plot(epochs, accuracy, marker="o")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("accuracy_plot.png")
    mlflow.log_artifact("accuracy_plot.png")

print("✅ Experiment complete — metrics and artifacts logged successfully!")
