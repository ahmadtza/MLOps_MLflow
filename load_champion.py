import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris


# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
from mlflow import MlflowClient

client = MlflowClient()

model_version = client.get_model_version_by_alias(
    "Iris Random Forest",
    "champion"
)

print("Champion version:", model_version.version)

# Load champion model
model_uri = "models:/Iris Random Forest@champion"

model = mlflow.sklearn.load_model(model_uri)


# Load Iris dataset
iris = load_iris()


# Select one sample
sample = iris.data[[0]]


# Prediction
prediction = model.predict(sample)


print("Model URI:", model_uri)
print("Input:", sample)
print("Prediction:", prediction)
print(
    "Predicted class:",
    iris.target_names[prediction[0]]
)