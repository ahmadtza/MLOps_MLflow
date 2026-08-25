import mlflow
from mlflow import MlflowClient


mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

model_name = "Iris Random Forest"


client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version="1"
)


print("Rollback completed.")
print("Champion is now Version 1.")