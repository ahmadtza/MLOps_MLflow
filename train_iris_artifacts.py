import mlflow
import mlflow.sklearn

import pandas as pd
import matplotlib.pyplot as plt

from mlflow.models import infer_signature

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. Connect to MLflow
# ============================================================

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Iris Artifacts Experiment")


# ============================================================
# 2. Enable Autologging
# ============================================================

# We disable automatic model logging because
# we want to log the model ourselves with:
# signature + input_example
mlflow.sklearn.autolog(
    log_models=False
)


# ============================================================
# 3. Load Iris dataset
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target


# ============================================================
# 4. Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 5. Convert train/test data to Pandas DataFrames
# ============================================================

X_train_df = pd.DataFrame(
    X_train,
    columns=iris.feature_names
)

X_test_df = pd.DataFrame(
    X_test,
    columns=iris.feature_names
)


train_df = X_train_df.copy()
train_df["target"] = y_train


test_df = X_test_df.copy()
test_df["target"] = y_test


# ============================================================
# 6. Create MLflow Dataset objects
# ============================================================

train_dataset = mlflow.data.from_pandas(
    train_df,
    name="iris_train"
)

test_dataset = mlflow.data.from_pandas(
    test_df,
    name="iris_test"
)


# ============================================================
# 7. Model parameters
# ============================================================

n_estimators = 150
max_depth = 6


# ============================================================
# 8. Start MLflow Run
# ============================================================

with mlflow.start_run():

    # --------------------------------------------------------
    # Log dataset inputs
    # --------------------------------------------------------

    mlflow.log_input(
        train_dataset,
        context="training"
    )

    mlflow.log_input(
        test_dataset,
        context="testing"
    )


    # --------------------------------------------------------
    # Log project parameters
    # --------------------------------------------------------

    mlflow.log_params({
        "train_rows": X_train.shape[0],
        "test_rows": X_test.shape[0],
        "num_features": X_train.shape[1],
        "test_size": 0.2,
        "split_random_state": 42
    })


    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )


    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    model.fit(
        X_train_df,
        y_train
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        X_test_df
    )


    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("Accuracy:", accuracy)


    # --------------------------------------------------------
    # Manual metric logging
    # --------------------------------------------------------

    mlflow.log_metric(
        "test_accuracy_manual",
        accuracy
    )


    # ========================================================
    # 9. Classification Report
    # ========================================================

    report = classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    )

    report_path = "classification_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)


    mlflow.log_artifact(
        report_path,
        artifact_path="reports"
    )

    print(
        "Classification report logged successfully!"
    )


    # ========================================================
    # 10. Feature Importance
    # ========================================================

    importances = model.feature_importances_

    feature_names = iris.feature_names


    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        feature_names,
        importances
    )

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.xlabel(
        "Features"
    )

    plt.ylabel(
        "Importance"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()


    feature_plot_path = (
        "feature_importance.png"
    )

    plt.savefig(
        feature_plot_path
    )

    plt.close()


    mlflow.log_artifact(
        feature_plot_path,
        artifact_path="plots"
    )

    print(
        "Feature importance logged successfully!"
    )


    # ========================================================
    # 11. Confusion Matrix
    # ========================================================

    cm = confusion_matrix(
        y_test,
        predictions
    )


    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=iris.target_names
    )


    display.plot()

    plt.title(
        "Iris Confusion Matrix"
    )

    plt.tight_layout()


    confusion_plot_path = (
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_plot_path
    )

    plt.close()


    mlflow.log_artifact(
        confusion_plot_path,
        artifact_path="plots"
    )

    print(
        "Confusion matrix logged successfully!"
    )


    # ========================================================
    # 12. Model Signature
    # ========================================================

    signature = infer_signature(
        X_train_df,
        model.predict(X_train_df)
    )


    # ========================================================
    # 13. Input Example
    # ========================================================

    input_example = X_train_df.head(3)


    # ========================================================
    # 14. Log trained model manually
    # ========================================================

    mlflow.sklearn.log_model(
        sk_model=model,
        name="random_forest_model",
        signature=signature,
        input_example=input_example
    )


    print(
        "Model logged successfully!"
    )


print(
    "Run completed successfully!"
)