import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris


# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Run ID from our successful run
run_id = "cde3d796a89246049fbdd8da36825eaa"


# Model URI
model_uri = f"runs:/{run_id}/random_forest_model"


# Load trained model
loaded_model = mlflow.sklearn.load_model(model_uri)


# Load Iris dataset
iris = load_iris()


# Take one sample
sample = iris.data[[0]]

print("Input sample:")
print(sample)


# Prediction
prediction = loaded_model.predict(sample)

print("Prediction:", prediction)

print("Predicted class:", iris.target_names[prediction[0]])
