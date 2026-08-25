import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Iris Autolog Experiment")


# Enable autologging
mlflow.sklearn.autolog()


# Load data
iris = load_iris()

X = iris.data
y = iris.target


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    random_state=42
)


with mlflow.start_run():

    model.fit(
        X_train,
        y_train
    )

    score = model.score(
        X_test,
        y_test
    )

    print("Test accuracy:", score)


print("Autolog run completed successfully!")