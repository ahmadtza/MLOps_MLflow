import mlflow
import mlflow.sklearn

from mlflow import MlflowClient

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# Configuration
# ============================================================

TRACKING_URI = "http://127.0.0.1:5000"

MODEL_NAME = "Iris Random Forest"

CHAMPION_ALIAS = "champion"
CANDIDATE_ALIAS = "candidate"

MINIMUM_ACCURACY = 0.95
MINIMUM_F1 = 0.95


# ============================================================
# MLflow configuration
# ============================================================

mlflow.set_tracking_uri(TRACKING_URI)

mlflow.set_experiment(
    "Model Validation"
)

client = MlflowClient()


# ============================================================
# Resolve model versions
# ============================================================

champion_version = client.get_model_version_by_alias(
    MODEL_NAME,
    CHAMPION_ALIAS
)

candidate_version = client.get_model_version_by_alias(
    MODEL_NAME,
    CANDIDATE_ALIAS
)


print(
    "Champion version:",
    champion_version.version
)

print(
    "Candidate version:",
    candidate_version.version
)

if champion_version.version == candidate_version.version:
    raise ValueError(
        "Champion and candidate point to the same model version. "
        "Validation requires two different versions."
    )

# ============================================================
# Load models
# ============================================================

champion_model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
)

candidate_model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}@{CANDIDATE_ALIAS}"
)


# ============================================================
# Validation dataset
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target


_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# Evaluation function
# ============================================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    return {
        "accuracy": accuracy_score(
            y,
            predictions
        ),

        "precision": precision_score(
            y,
            predictions,
            average="weighted"
        ),

        "recall": recall_score(
            y,
            predictions,
            average="weighted"
        ),

        "f1": f1_score(
            y,
            predictions,
            average="weighted"
        )
    }


# ============================================================
# Evaluate both models
# ============================================================

champion_metrics = evaluate_model(
    champion_model,
    X_test,
    y_test
)

candidate_metrics = evaluate_model(
    candidate_model,
    X_test,
    y_test
)


print("\nChampion metrics:")
print(champion_metrics)

print("\nCandidate metrics:")
print(candidate_metrics)


# ============================================================
# Quality Gate
# ============================================================

passes_thresholds = (

    candidate_metrics["accuracy"]
    >= MINIMUM_ACCURACY

    and

    candidate_metrics["f1"]
    >= MINIMUM_F1
)


not_worse_than_champion = (

    candidate_metrics["accuracy"]
    >= champion_metrics["accuracy"]

    and

    candidate_metrics["f1"]
    >= champion_metrics["f1"]
)


promotion_passed = (
    passes_thresholds
    and
    not_worse_than_champion
)


# ============================================================
# Validation Audit Run
# ============================================================

with mlflow.start_run():

    # -------------------------
    # Parameters
    # -------------------------

    mlflow.log_params({

        "model_name":
            MODEL_NAME,

        "champion_version":
            champion_version.version,

        "candidate_version":
            candidate_version.version,

        "minimum_accuracy":
            MINIMUM_ACCURACY,

        "minimum_f1":
            MINIMUM_F1
    })


    # -------------------------
    # Champion Metrics
    # -------------------------

    mlflow.log_metrics({

        "champion_accuracy":
            champion_metrics["accuracy"],

        "champion_precision":
            champion_metrics["precision"],

        "champion_recall":
            champion_metrics["recall"],

        "champion_f1":
            champion_metrics["f1"]
    })


    # -------------------------
    # Candidate Metrics
    # -------------------------

    mlflow.log_metrics({

        "candidate_accuracy":
            candidate_metrics["accuracy"],

        "candidate_precision":
            candidate_metrics["precision"],

        "candidate_recall":
            candidate_metrics["recall"],

        "candidate_f1":
            candidate_metrics["f1"]
    })


    # -------------------------
    # Promotion
    # -------------------------

    if promotion_passed:

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias=CHAMPION_ALIAS,
            version=candidate_version.version
        )

        mlflow.set_tag(
            "promotion_result",
            "passed"
        )

        print("\nPROMOTION PASSED")

        print(
            f"Version "
            f"{candidate_version.version} "
            f"is now champion."
        )

    else:

        mlflow.set_tag(
            "promotion_result",
            "failed"
        )

        print("\nPROMOTION FAILED")

        print(
            f"Version "
            f"{champion_version.version} "
            f"remains champion."
        )


print(
    "\nValidation pipeline completed."
)