import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("My First MLOps Experiment")

with mlflow.start_run():

    mlflow.log_param("model", "Random Forest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("accuracy", 0.96)

print("Run completed successfully!")