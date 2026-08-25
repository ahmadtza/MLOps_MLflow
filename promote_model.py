import mlflow
from mlflow import MlflowClient


mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

model_name = "Iris Random Forest"


# Get model versions behind aliases
champion_version = client.get_model_version_by_alias(
    model_name,
    "champion"
)

candidate_version = client.get_model_version_by_alias(
    model_name,
    "candidate"
)


print("Champion version:", champion_version.version)
print("Candidate version:", candidate_version.version)


# Get associated runs
champion_run = client.get_run(
    champion_version.run_id
)

candidate_run = client.get_run(
    candidate_version.run_id
)


# Read F1 metrics
champion_f1 = champion_run.data.metrics["f1_score"]
candidate_f1 = candidate_run.data.metrics["f1_score"]


print("Champion F1:", champion_f1)
print("Candidate F1:", candidate_f1)


# Promotion rule
if candidate_f1 >= champion_f1:

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=candidate_version.version
    )

    print(
        f"Candidate version {candidate_version.version} "
        "promoted to champion!"
    )

else:

    print("Candidate was not promoted.")