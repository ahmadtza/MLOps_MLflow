import mlflow
from mlflow import MlflowClient


# Connect to MLflow Tracking Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Create MLflow client
client = MlflowClient()


model_name = "Iris Random Forest"


# Version 1 = current trusted model
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version="1"
)


# Version 2 = new candidate model
client.set_registered_model_alias(
    name=model_name,
    alias="candidate",
    version="2"
)


print("Aliases created successfully!")