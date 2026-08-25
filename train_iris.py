import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Connect to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Iris Classification")


# Load dataset
iris = load_iris()

X = iris.data
y = iris.target


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Model parameters
n_estimators = 200
max_depth = 8


with mlflow.start_run():

    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    # Log parameters
    mlflow.log_params({
        "n_estimators": n_estimators,
        "max_depth": max_depth
    })

    # Log metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # Log trained model
    mlflow.sklearn.log_model(
    sk_model=model,
    name="random_forest_model",
    registered_model_name="Iris Random Forest"
)

    # Print results
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", f1)


print("Run completed successfully!")